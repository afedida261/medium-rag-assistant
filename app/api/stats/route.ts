import { RAG_CONFIG } from "@/lib/rag-config";

export async function GET() {
  return Response.json(RAG_CONFIG);
}
