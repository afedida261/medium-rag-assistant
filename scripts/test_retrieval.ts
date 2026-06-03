import dotenv from "dotenv";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { EMBEDDING_MODEL, RAG_CONFIG } from "../lib/rag-config";

dotenv.config({ path: ".env.local" });

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing environment variable: ${name}`);
  }
  return value;
}

async function main() {
  const question =
    "Find an article about education and provide the title and author.";

  const openai = new OpenAI({
    apiKey: requireEnv("LLMOD_API_KEY"),
    baseURL: requireEnv("LLMOD_BASE_URL"),
  });

  const pc = new Pinecone({
    apiKey: requireEnv("PINECONE_API_KEY"),
  });

  const index = pc.index(requireEnv("PINECONE_INDEX_NAME"));

  const embeddingResponse = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: question,
  });

  const queryVector = embeddingResponse.data[0].embedding;

  const results = await index.query({
    vector: queryVector,
    topK: RAG_CONFIG.top_k,
    includeMetadata: true,
  });

  console.log(`Question: ${question}`);
  console.log("\nTop retrieved chunks:\n");

  for (const [i, match] of results.matches.entries()) {
    const metadata = match.metadata as {
      article_id?: string;
      title?: string;
      authors?: string;
      chunk?: string;
      url?: string;
    };

    console.log(`Result ${i + 1}`);
    console.log(`Score: ${match.score}`);
    console.log(`Article ID: ${metadata.article_id}`);
    console.log(`Title: ${metadata.title}`);
    console.log(`Authors: ${metadata.authors}`);
    console.log(`URL: ${metadata.url}`);
    console.log(`Chunk preview: ${metadata.chunk?.slice(0, 500)}...`);
    console.log("-".repeat(80));
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
