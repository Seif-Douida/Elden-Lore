// app/page.tsx
"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar";
import { ChatView } from "@/components/ChatView";

function EmptyState({ onNew }: { onNew: () => void }) {
  return (
    <main style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", padding: 32 }}>
      <h2 className="font-display" style={{ color: "var(--gold)", fontSize: 28, letterSpacing: "0.12em", margin: "0 0 12px" }}>
        ELDEN PATH
      </h2>
      <p style={{ color: "var(--parchment-dim)", fontSize: 17, maxWidth: 420, lineHeight: 1.6, margin: "0 0 24px" }}>
        Seek knowledge of the Lands Between — its bosses, its lore, its hidden paths.
      </p>
      <button
        onClick={onNew}
        className="font-display"
        style={{ padding: "12px 24px", background: "var(--gold)", color: "var(--abyss)", border: "none", borderRadius: "var(--radius)", fontSize: 13, letterSpacing: "0.1em", cursor: "pointer" }}
      >
        BEGIN A CHAT
      </button>
    </main>
  );
}

export default function Home() {
  const qc = useQueryClient();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeId, setActiveId] = useState<string | null>(null);

  const { data: conversations = [] } = useQuery({
    queryKey: ["conversations"],
    queryFn: api.listConversations,
  });

  const createMut = useMutation({
    mutationFn: () => api.createConversation("scholar"),
    onSuccess: (conv) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      setActiveId(conv.id);
    },
  });

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.deleteConversation(id),
    onSuccess: (_res, id) => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      if (activeId === id) setActiveId(null);
    },
  });

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={() => createMut.mutate()}
        onClose={() => setSidebarOpen(false)}
        onDelete={(id) => deleteMut.mutate(id)}
      />
      {activeId ? (
        <ChatView
          key={activeId}
          conversationId={activeId}
          sidebarOpen={sidebarOpen}
          onOpenSidebar={() => setSidebarOpen(true)}
        />
      ) : (
        <EmptyState onNew={() => createMut.mutate()} />
      )}
    </div>
  );
}