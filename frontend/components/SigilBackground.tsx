// components/SigilBackground.tsx
"use client";

// Uses the real uploaded sigil image from /public/sigil.jpg as a dim, fixed
// watermark behind the conversation. Drop your sigil .jpg into frontend/public/
// and name it sigil.jpg (or change the path below).

export function SigilBackground({ shifted }: { shifted: boolean }) {
  return (
    <div
      aria-hidden
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        pointerEvents: "none",
        zIndex: 0,
        backgroundImage: "url('/sigil.jpg')",
        backgroundRepeat: "no-repeat",
        backgroundPosition: `calc(50% + ${shifted ? 135 : 0}px) center`,
        backgroundSize: "520px auto",
        opacity: 0.12,
        transition: "background-position 0.22s ease",
      }}
    />
  );
}