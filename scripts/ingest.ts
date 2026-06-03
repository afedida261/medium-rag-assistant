import dotenv from "dotenv";
import fs from "fs";
import readline from "readline";
import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

dotenv.config({ path: ".env.local" });

const DEFAULT_CHUNKS_PATH = "data/chunks_full.jsonl";
const EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small";
const BATCH_SIZE = 50;

type ChunkRecord = {
  id: string;
  article_id: string;
  chunk_index: number;
  title: string;
  url: string;
  authors: string;
  timestamp: string;
  tags: string;
  chunk: string;
};

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing environment variable: ${name}`);
  }
  return value;
}

async function readJsonl(path: string): Promise<ChunkRecord[]> {
  const records: ChunkRecord[] = [];

  const rl = readline.createInterface({
    input: fs.createReadStream(path, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (!line.trim()) continue;
    records.push(JSON.parse(line));
  }

  return records;
}

function getChunksPath(): string {
  return process.env.CHUNKS_PATH || DEFAULT_CHUNKS_PATH;
}

function buildEmbeddingInput(record: ChunkRecord): string {
  const parts = [
    `Title: ${record.title}`,
    `Tags: ${record.tags}`,
    `Authors: ${record.authors}`,
    `Chunk: ${record.chunk}`,
  ];

  return parts.join("\n");
}

async function main() {
  const llmodApiKey = requireEnv("LLMOD_API_KEY");
  const llmodBaseUrl = requireEnv("LLMOD_BASE_URL");
  const pineconeApiKey = requireEnv("PINECONE_API_KEY");
  const pineconeIndexName = requireEnv("PINECONE_INDEX_NAME");

  const openai = new OpenAI({
    apiKey: llmodApiKey,
    baseURL: llmodBaseUrl,
  });

  const pc = new Pinecone({
    apiKey: pineconeApiKey,
  });

  const index = pc.index(pineconeIndexName);
  const chunksPath = getChunksPath();

  const records = await readJsonl(chunksPath);

  console.log(`Loaded ${records.length} chunks from ${chunksPath}`);

  for (let i = 0; i < records.length; i += BATCH_SIZE) {
    const batch = records.slice(i, i + BATCH_SIZE);

    const embeddingResponse = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: batch.map(buildEmbeddingInput),
    });

    const vectors = batch.map((record, j) => ({
      id: record.id,
      values: embeddingResponse.data[j].embedding,
      metadata: {
        article_id: record.article_id,
        chunk_index: record.chunk_index,
        title: record.title,
        url: record.url,
        authors: record.authors,
        timestamp: record.timestamp,
        tags: record.tags,
        chunk: record.chunk,
      },
    }));

    await index.upsert({
        records: vectors,
    });

    console.log(`Upserted ${Math.min(i + BATCH_SIZE, records.length)} / ${records.length}`);
  }

  const stats = await index.describeIndexStats();
  console.log("Done. Pinecone stats:");
  console.log(JSON.stringify(stats, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
