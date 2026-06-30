// lib/useChatStream.ts
// Streams an answer from POST /conversations/{id}/chat over SSE, with a smooth
// typewriter reveal.
//
// Why not EventSource: it only does GET. We need POST with a JSON body, so we
// use fetch + ReadableStream and parse SSE frames ourselves. The parser buffers
// across reads because a single network chunk may contain a partial event or
// several events — splitting only on the blank-line frame delimiter ("\n\n").
//
// Reveal pacing: tokens arrive from Gemini in staggered BURSTS (one network
// chunk often carries ~10 tokens at once). Appending them straight to state
// makes the answer jump forward in big steps. Instead we accumulate received
// text into a "target", and a requestAnimationFrame loop advances a "revealed"
// cursor a few characters per frame — a steady typewriter effect decoupled from
// network batching. The full text is never lost; we only pace its display.

import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL } from "./api";
import type {
  Tone, Source, EntityImage, MetaEvent, TokenEvent, SourcesEvent, ImagesEvent,
} from "./types";

export interface StreamState {
  streaming: boolean;
  text: string;        // the revealed (displayed) text
  tone: Tone | null;
  sources: Source[];
  images: EntityImage[];
  error: string | null;
}

const INITIAL: StreamState = {
  streaming: false,
  text: "",
  tone: null,
  sources: [],
  images: [],
  error: null,
};

// Characters revealed per animation frame (~60fps). Higher = faster reveal.
const CHARS_PER_FRAME = 3;

interface ParsedEvent {
  event: string;
  data: string;
}

function parseFrame(frame: string): ParsedEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}

export function useChatStream() {
  const [state, setState] = useState<StreamState>(INITIAL);

  const abortRef = useRef<AbortController | null>(null);
  // Reveal machinery
  const targetRef = useRef("");        // full text received so far
  const revealedRef = useRef(0);       // chars currently shown
  const rafRef = useRef<number | null>(null);
  const streamDoneRef = useRef(false); // network finished?
  const onDoneRef = useRef<(() => void) | null>(null);
  // Sources/images are received mid-stream but should only appear once the
  // text has finished revealing, so they don't pop in above half-typed text.
  const pendingSourcesRef = useRef<Source[]>([]);
  const pendingImagesRef = useRef<EntityImage[]>([]);

  const stopRaf = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const tick = useCallback(() => {
    const target = targetRef.current;
    if (revealedRef.current < target.length) {
      revealedRef.current = Math.min(target.length, revealedRef.current + CHARS_PER_FRAME);
      const shown = target.slice(0, revealedRef.current);
      setState((s) => ({ ...s, text: shown }));
      rafRef.current = requestAnimationFrame(tick);
    } else if (!streamDoneRef.current) {
      // Caught up but more may arrive — keep polling lightly.
      rafRef.current = requestAnimationFrame(tick);
    } else {
      // Fully revealed AND stream finished → reveal sources/images, then settle.
      stopRaf();
      setState((s) => ({
        ...s,
        streaming: false,
        sources: pendingSourcesRef.current,
        images: pendingImagesRef.current,
      }));
      onDoneRef.current?.();
      onDoneRef.current = null;
    }
  }, [stopRaf]);

  const reset = useCallback(() => {
    stopRaf();
    targetRef.current = "";
    revealedRef.current = 0;
    streamDoneRef.current = false;
    pendingSourcesRef.current = [];
    pendingImagesRef.current = [];
    setState(INITIAL);
  }, [stopRaf]);

  const send = useCallback(
    async (
      conversationId: string,
      question: string,
      opts?: { k?: number; tone?: Tone; onDone?: () => void }
    ) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;

      // reset reveal machinery
      stopRaf();
      targetRef.current = "";
      revealedRef.current = 0;
      streamDoneRef.current = false;
      pendingSourcesRef.current = [];
      pendingImagesRef.current = [];
      onDoneRef.current = opts?.onDone ?? null;
      setState({ ...INITIAL, streaming: true });

      // start the reveal loop
      rafRef.current = requestAnimationFrame(tick);

      try {
        const res = await fetch(
          `${API_URL}/conversations/${conversationId}/chat`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, k: opts?.k ?? 8, tone: opts?.tone ?? null }),
            signal: ac.signal,
          }
        );

        if (!res.ok || !res.body) throw new Error(`Request failed (${res.status})`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let idx: number;
          while ((idx = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);
            const parsed = parseFrame(frame);
            if (!parsed) continue;
            handleEvent(parsed, {
              appendText: (t) => { targetRef.current += t; },
              setMeta: (tone) => setState((s) => ({ ...s, tone })),
              setSources: (sources) => { pendingSourcesRef.current = sources; },
              setImages: (images) => { pendingImagesRef.current = images; },
              setError: (message) => {
                streamDoneRef.current = true;
                setState((s) => ({ ...s, streaming: false, error: message }));
              },
              markDone: () => { streamDoneRef.current = true; },
            });
          }
        }
        // Network finished. The reveal loop will drain remaining chars, then
        // settle + fire onDone.
        streamDoneRef.current = true;
      } catch (err: unknown) {
        if ((err as Error).name === "AbortError") return;
        const message = err instanceof Error ? err.message : "Stream failed";
        streamDoneRef.current = true;
        stopRaf();
        setState((s) => ({ ...s, streaming: false, error: message }));
      }
    },
    [tick, stopRaf]
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    streamDoneRef.current = true;
    stopRaf();
    setState((s) => ({ ...s, streaming: false }));
  }, [stopRaf]);

  // Cleanup on unmount.
  useEffect(() => () => stopRaf(), [stopRaf]);

  return { ...state, send, cancel, reset };
}

interface Handlers {
  appendText: (t: string) => void;
  setMeta: (tone: Tone) => void;
  setSources: (s: Source[]) => void;
  setImages: (i: EntityImage[]) => void;
  setError: (m: string) => void;
  markDone: () => void;
}

function handleEvent({ event, data }: ParsedEvent, h: Handlers) {
  let payload: unknown;
  try {
    payload = JSON.parse(data);
  } catch {
    return;
  }
  switch (event) {
    case "meta":
      h.setMeta((payload as MetaEvent).tone);
      break;
    case "token":
      h.appendText((payload as TokenEvent).text);
      break;
    case "sources":
      h.setSources((payload as SourcesEvent).sources);
      break;
    case "images":
      h.setImages((payload as ImagesEvent).images);
      break;
    case "done":
      h.markDone();
      break;
    case "error":
      h.setError((payload as { message: string }).message);
      break;
  }
}