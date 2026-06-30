// components/ChatInput.tsx
"use client";

import { useState, KeyboardEvent } from "react";

function ArrowIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M12 19 L12 5 M6 11 L12 5 L18 11" stroke="var(--abyss)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function ChatInput({
  onSend,
  disabled,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
}) {
  const [value, setValue] = useState("");

  const submit = () => {
    const t = value.trim();
    if (!t || disabled) return;
    onSend(t);
    setValue("");
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div style={{ borderTop: "1px solid color-mix(in srgb, var(--gold-dim) 22%, transparent)", padding: "20px 0", position: "relative", zIndex: 2, background: "var(--abyss)" }}>
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "0 32px", display: "flex", gap: 12 }}>
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          placeholder="What would you like to know"
          style={{
            flex: 1,
            background: "var(--ash)",
            border: "1px solid color-mix(in srgb, var(--gold-dim) 44%, transparent)",
            borderRadius: "var(--radius)",
            padding: "13px 18px",
            fontSize: 16,
            color: "var(--parchment)",
            fontFamily: "var(--font-body), serif",
            outline: "none",
          }}
        />
        <button
          onClick={submit}
          disabled={disabled}
          aria-label="Send"
          style={{
            background: "var(--gold)",
            border: "none",
            borderRadius: "var(--radius)",
            width: 50,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: disabled ? "default" : "pointer",
            opacity: disabled ? 0.5 : 1,
          }}
        >
          <ArrowIcon />
        </button>
      </div>
    </div>
  );
}