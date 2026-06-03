export const RAG_CONFIG = {
  chunk_size: 640,
  overlap_ratio: 0.2,
  top_k: 7,
} as const;

export const EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small";
export const CHAT_MODEL = "4UHRUIN-gpt-5-mini";
export const RETRIEVAL_CANDIDATE_K = Math.min(RAG_CONFIG.top_k * 3, 30);
