// components/ChatView.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useChatStream } from "@/lib/useChatStream";
import type { Tone } from "@/lib/types";
import { MessageItem, UserMessage, AssistantMessage } from "./Message";
import { ChatInput } from "./ChatInput";
import { VoiceToggle } from "./VoiceToggle";
import { SigilBackground } from "./SigilBackground";

function MenuIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M4 6 H20 M4 12 H20 M4 18 H20" stroke={color} strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function ChatView({
  conversationId,
  sidebarOpen,
  onOpenSidebar,
}: {
  conversationId: string;
  sidebarOpen: boolean;
  onOpenSidebar: () => void;
}) {
  const qc = useQueryClient();
  const [tone, setTone] = useState<Tone>("scholar");
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const stream = useChatStream();
  const scrollRef = useRef<HTMLDivElement>(null);

  // Load the conversation + its messages.
  const { data: conv } = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => api.getConversation(conversationId),
  });

  // Keep the local tone in sync with the conversation's stored tone.
  useEffect(() => {
    if (conv?.tone) setTone(conv.tone);
  }, [conv?.tone]);

  // Auto-scroll to the bottom as content grows.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [conv?.messages?.length, stream.text]);

  const handleSend = (question: string) => {
    // The user message + streamed answer are persisted server-side; on done we
    // refetch the conversation to pull the canonical, saved messages.
    setPendingQuestion(question);
    stream.send(conversationId, question, {
      tone,
      onDone: () => {
        qc.invalidateQueries({ queryKey: ["conversation", conversationId] });
        qc.invalidateQueries({ queryKey: ["conversations"] });
        setPendingQuestion(null);
        stream.reset();
      },
    });
  };

  const messages = conv?.messages ?? [];

  return (
    <main style={{ flex: 1, display: "flex", flexDirection: "column", position: "relative", minWidth: 0 }}>
      <SigilBackground shifted={sidebarOpen} />

      {/* Top-left: show-sidebar button when hidden */}
      {!sidebarOpen && (
        <div style={{ position: "absolute", top: 18, left: 22, zIndex: 3 }}>
          <button
            onClick={onOpenSidebar}
            aria-label="Show sidebar"
            style={{ background: "var(--ash)", border: "1px solid color-mix(in srgb, var(--gold-dim) 55%, transparent)", borderRadius: 8, padding: 7, cursor: "pointer", display: "flex" }}
          >
            <MenuIcon color="var(--gold)" />
          </button>
        </div>
      )}

      {/* Top-right: voice toggle */}
      <div style={{ position: "absolute", top: 18, right: 26, zIndex: 3 }}>
        <VoiceToggle tone={tone} onChange={setTone} />
      </div>

      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "40px 0", position: "relative" }}>
        <div style={{ maxWidth: 720, margin: "0 auto", padding: "0 32px", position: "relative", zIndex: 1 }}>
          {messages.map((m) => (
            <MessageItem key={m.id} message={m} />
          ))}

          {/* Live streaming answer (not yet persisted/refetched) */}
          {stream.streaming && (
            <>
              {pendingQuestion && <UserMessage content={pendingQuestion} />}
              <AssistantMessage
                content={stream.text}
                sources={stream.sources}
                images={stream.images}
                streaming
              />
            </>
          )}

          {stream.error && (
            <div style={{ color: "#c97a5a", margin: "12px 0", fontSize: 15 }}>
              {stream.error}
            </div>
          )}
        </div>
      </div>

      <ChatInput onSend={handleSend} disabled={stream.streaming} />
    </main>
  );
}