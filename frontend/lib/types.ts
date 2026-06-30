// lib/types.ts
// Types mirroring the backend API contract.

export type Tone = "scholar" | "cryptic";

export interface Source {
  title: string;
  url: string;
  section: string;
  image_url: string | null;
  category: string;
}

export interface EntityImage {
  image_url: string;
  title: string;
  url: string;
}

export interface Conversation {
  id: string;
  title: string;
  tone: Tone;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null;
  images: EntityImage[] | null;
  tone: Tone | null;
  created_at: string;
}

export interface ConversationWithMessages extends Conversation {
  messages: Message[];
}

// ── SSE event payloads (from the /chat stream) ──
export interface MetaEvent {
  tone: Tone;
  used_retrieval: boolean;
  entity_fallback: boolean;
}
export interface TokenEvent {
  text: string;
}
export interface SourcesEvent {
  sources: Source[];
}
export interface ImagesEvent {
  images: EntityImage[];
}
export interface ErrorEvent {
  message: string;
}