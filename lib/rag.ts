import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { SYSTEM_PROMPT, buildUserPrompt } from "./prompts";

const EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small";
const CHAT_MODEL = "4UHRUIN-gpt-5-mini";
const TOP_K = 5;

type RetrievedContext = {
  article_id: string;
  title: string;
  chunk: string;
  score: number;
};

function requireEnv(name: string): string {
  const value = process.env[name];

  if (!value) {
    throw new Error(`Missing environment variable: ${name}`);
  }

  return value;
}

function formatContext(matches: RetrievedContext[]): string {
  return matches
    .map((item, index) => {
      return `Context ${index + 1}
Article ID: ${item.article_id}
Title: ${item.title}
Score: ${item.score}
Chunk:
${item.chunk}`;
    })
    .join("\n\n---\n\n");
}

export async function answerQuestion(question: string) {
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

  const pineconeResults = await index.query({
    vector: queryVector,
    topK: TOP_K,
    includeMetadata: true,
  });

  const context: RetrievedContext[] = (pineconeResults.matches ?? []).map(
    (match) => {
      const metadata = match.metadata as {
        article_id?: string;
        title?: string;
        chunk?: string;
      };

      return {
        article_id: String(metadata.article_id ?? ""),
        title: String(metadata.title ?? ""),
        chunk: String(metadata.chunk ?? ""),
        score: match.score ?? 0,
      };
    }
  );

  const contextText = formatContext(context);
  const userPrompt = buildUserPrompt(question, contextText);

  const chatResponse = await openai.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      {
        role: "system",
        content: SYSTEM_PROMPT,
      },
      {
        role: "user",
        content: userPrompt,
      },
    ],
  });

  const response =
    chatResponse.choices[0]?.message?.content ??
    "I don’t know based on the provided Medium articles data.";

  return {
    response,
    context,
    Augmented_prompt: {
      System: SYSTEM_PROMPT,
      User: userPrompt,
    },
  };
}