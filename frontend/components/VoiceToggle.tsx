// components/VoiceToggle.tsx
"use client";

import type { Tone } from "@/lib/types";

export function VoiceToggle({
  tone,
  onChange,
}: {
  tone: Tone;
  onChange: (t: Tone) => void;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span className="font-display" style={{ fontSize: 10, letterSpacing: "0.16em", color: "var(--gold-dim)" }}>
        VOICE
      </span>
      <div style={{ display: "flex", border: "1px solid var(--gold-dim)", borderRadius: "var(--radius)", overflow: "hidden" }}>
        {(["scholar", "cryptic"] as Tone[]).map((t) => (
          <button
            key={t}
            onClick={() => onChange(t)}
            className="font-display"
            style={{
              padding: "6px 16px",
              background: tone === t ? "var(--gold)" : "transparent",
              color: tone === t ? "var(--abyss)" : "var(--parchment)",
              border: "none",
              fontSize: 11,
              letterSpacing: "0.08em",
              cursor: "pointer",
              textTransform: "capitalize",
            }}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}