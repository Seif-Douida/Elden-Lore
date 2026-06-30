// components/Sidebar.tsx
"use client";

import type { Conversation } from "@/lib/types";

function MenuIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M4 6 H20 M4 12 H20 M4 18 H20" stroke={color} strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function Sidebar({
  open,
  conversations,
  activeId,
  onSelect,
  onNew,
  onClose,
  onDelete,
}: {
  open: boolean;
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onClose: () => void;
  onDelete: (id: string) => void;
}) {
  return (
    <aside
      style={{
        width: open ? 270 : 0,
        background: "var(--ash)",
        borderRight: open ? "1px solid color-mix(in srgb, var(--gold-dim) 33%, transparent)" : "none",
        display: "flex",
        flexDirection: "column",
        padding: open ? "22px 0" : 0,
        overflow: "hidden",
        transition: "width 0.22s ease, padding 0.22s ease",
        flexShrink: 0,
      }}
    >
      <div style={{ padding: "0 22px 20px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <h1 className="font-display" style={{ fontSize: 22, letterSpacing: "0.18em", color: "var(--gold)", margin: 0, whiteSpace: "nowrap" }}>
          ELDEN PATH
        </h1>
        <button onClick={onClose} style={{ background: "none", border: "none", cursor: "pointer", padding: 4 }} aria-label="Hide sidebar">
          <MenuIcon color="var(--gold-dim)" />
        </button>
      </div>

      <button
        onClick={onNew}
        className="font-display"
        style={{
          margin: "0 18px 22px",
          padding: "11px 14px",
          background: "transparent",
          border: "1px solid var(--gold-dim)",
          color: "var(--parchment)",
          fontSize: 12,
          letterSpacing: "0.1em",
          cursor: "pointer",
          borderRadius: "var(--radius)",
          whiteSpace: "nowrap",
        }}
      >
        + NEW CHAT
      </button>

      <div className="font-display" style={{ padding: "0 22px 8px", fontSize: 11, letterSpacing: "0.16em", color: "var(--gold-dim)", textTransform: "uppercase" }}>
        Chats
      </div>

      <nav style={{ flex: 1, overflowY: "auto" }}>
        {conversations.map((c) => (
          <div
            key={c.id}
            onClick={() => onSelect(c.id)}
            className="chat-row"
            style={{
              margin: "2px 12px",
              padding: "9px 14px",
              fontSize: 15,
              color: c.id === activeId ? "var(--gold)" : "var(--parchment)",
              background: c.id === activeId ? "color-mix(in srgb, var(--gold) 8%, transparent)" : "transparent",
              borderRadius: "var(--radius)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 8,
            }}
          >
            <span style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {c.title}
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
              aria-label="Delete chat"
              style={{ background: "none", border: "none", color: "var(--gold-dim)", cursor: "pointer", fontSize: 16, lineHeight: 1, opacity: 0.6 }}
            >
              ×
            </button>
          </div>
        ))}
      </nav>
    </aside>
  );
}