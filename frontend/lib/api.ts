// lib/api.ts
// Typed client for the conversation CRUD endpoints.

import type { Conversation, ConversationWithMessages, Tone } from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3000";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.error?.message ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listConversations: () => req<Conversation[]>("/conversations"),

  createConversation: (tone: Tone = "scholar") =>
    req<Conversation>("/conversations", {
      method: "POST",
      body: JSON.stringify({ tone }),
    }),

  getConversation: (id: string) =>
    req<ConversationWithMessages>(`/conversations/${id}`),

  renameConversation: (id: string, title?: string, tone?: Tone) =>
    req<Conversation>(`/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title, tone }),
    }),

  deleteConversation: (id: string) =>
    req<{ deleted: boolean }>(`/conversations/${id}`, { method: "DELETE" }),
};

export const API_URL = API;