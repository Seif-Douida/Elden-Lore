// components/Message.tsx
"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message as Msg, Source, EntityImage } from "@/lib/types";

function RuneGlyph() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" style={{ flexShrink: 0 }}>
      <path
        d="M12 2 L12 22 M12 8 L7 4 M12 8 L17 4 M12 14 L6 11 M12 14 L18 11"
        stroke="var(--gold)" strokeWidth="1.2" strokeLinecap="round" opacity="0.9"
      />
      <circle cx="12" cy="12" r="1.6" fill="var(--gold)" />
    </svg>
  );
}

function SourceCards({ sources }: { sources: Source[] }) {
  if (!sources?.length) return null;
  const seen = new Set<string>();
  const unique = sources.filter((s) => (seen.has(s.url) ? false : (seen.add(s.url), true)));
  return (
    <div style={{ marginTop: 18, display: "flex", flexWrap: "wrap", gap: 8 }}>
      {unique.map((s) => (
        <a
          key={s.url}
          href={s.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{
            border: "1px solid color-mix(in srgb, var(--gold-dim) 44%, transparent)",
            borderRadius: "var(--radius)",
            padding: "6px 12px",
            fontSize: 13,
            color: "var(--parchment-dim)",
            background: "color-mix(in srgb, var(--ash) 80%, transparent)",
            textDecoration: "none",
          }}
        >
          <span style={{ color: "var(--gold)" }}>{s.title}</span>
          {s.section && <span style={{ opacity: 0.5 }}> · {s.section}</span>}
        </a>
      ))}
    </div>
  );
}

function ImageGallery({ images }: { images: EntityImage[] }) {
  // Drop entries with no usable image URL (some wiki pages have null/empty).
  const valid = (images ?? []).filter(
    (im) => im.image_url && im.image_url.trim().length > 0
  );
  // Track images that fail to load at runtime, so we can hide them.
  const [broken, setBroken] = useState<Set<string>>(new Set());
  const shown = valid.filter((im) => !broken.has(im.image_url));
  if (!shown.length) return null;

  return (
    <div
      style={{
        marginBottom: 18,
        display: "grid",
        gridTemplateColumns: shown.length === 1 ? "1fr" : "repeat(auto-fill, minmax(160px, 1fr))",
        gap: 12,
      }}
    >
      {shown.map((im) => (
        <figure key={im.image_url} style={{ margin: 0 }}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={im.image_url}
            alt={im.title}
            onError={() =>
              setBroken((prev) => {
                const next = new Set(prev);
                next.add(im.image_url);
                return next;
              })
            }
            style={{
              width: "100%",
              borderRadius: "var(--radius)",
              border: "1px solid color-mix(in srgb, var(--gold-dim) 44%, transparent)",
              display: "block",
            }}
          />
          <figcaption
            className="font-display"
            style={{ fontSize: 11, letterSpacing: "0.08em", color: "var(--gold-dim)", marginTop: 6, textAlign: "center" }}
          >
            {im.title}
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

export function UserMessage({ content }: { content: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-end", margin: "28px 0" }}>
      <div
        style={{
          background: "var(--ash-hi)",
          border: "1px solid color-mix(in srgb, var(--gold-dim) 33%, transparent)",
          borderRadius: "var(--radius)",
          padding: "12px 18px",
          fontSize: 17,
          maxWidth: "75%",
        }}
      >
        {content}
      </div>
    </div>
  );
}

export function AssistantMessage({
  content,
  sources,
  images,
  streaming,
}: {
  content: string;
  sources?: Source[] | null;
  images?: EntityImage[] | null;
  streaming?: boolean;
}) {
  return (
    <div style={{ margin: "0 0 36px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <RuneGlyph />
        <div style={{ height: 1, flex: 1, background: "linear-gradient(90deg, var(--gold-dim), transparent)" }} />
      </div>

      {images && images.length > 0 && <ImageGallery images={images} />}

      <div className="answer" style={{ fontSize: 17.5 }}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        {streaming && <span className="cursor-pulse" />}
      </div>

      {sources && sources.length > 0 && <SourceCards sources={sources} />}
    </div>
  );
}

export function MessageItem({ message }: { message: Msg }) {
  if (message.role === "user") return <UserMessage content={message.content} />;
  return (
    <AssistantMessage
      content={message.content}
      sources={message.sources}
      images={message.images}
    />
  );
}