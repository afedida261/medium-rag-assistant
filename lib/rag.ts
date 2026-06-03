import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";
import { SYSTEM_PROMPT, buildUserPrompt } from "./prompts";
import {
  CHAT_MODEL,
  EMBEDDING_MODEL,
  RAG_CONFIG,
  RETRIEVAL_CANDIDATE_K,
} from "./rag-config";

type RetrievedContext = {
  article_id: string;
  title: string;
  authors: string;
  tags: string;
  chunk: string;
  score: number;
};

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "about",
  "are",
  "as",
  "at",
  "article",
  "articles",
  "be",
  "by",
  "can",
  "exactly",
  "find",
  "for",
  "from",
  "give",
  "i",
  "if",
  "in",
  "is",
  "it",
  "list",
  "me",
  "of",
  "on",
  "or",
  "provide",
  "question",
  "recommend",
  "return",
  "titles",
  "title",
  "summarise",
  "summarize",
  "that",
  "the",
  "their",
  "them",
  "to",
  "want",
  "which",
  "with",
  "why",
]);

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
Authors: ${item.authors}
Tags: ${item.tags}
Score: ${item.score}
Chunk:
${item.chunk}`;
    })
    .join("\n\n---\n\n");
}

function getRequestedTitleCount(question: string): number | null {
  const match = question.toLowerCase().match(/list exactly\s+(\d+)/);
  return match ? Number.parseInt(match[1], 10) : null;
}

function sanitizeModelResponse(response: string): string {
  const fallback = "I don't know based on the provided Medium articles data.";
  const trimmed = response.trim();

  if (trimmed === fallback) {
    return trimmed;
  }

  const fallbackIndex = trimmed.indexOf(fallback);

  if (fallbackIndex <= 0) {
    return trimmed;
  }

  const beforeFallback = trimmed.slice(0, fallbackIndex).trim();

  return beforeFallback || fallback;
}

function formatTitleOnlyResponse(question: string, response: string, context: RetrievedContext[]): string {
  const requestedCount = getRequestedTitleCount(question);

  if (!requestedCount) {
    return response;
  }

  const lines = response
    .split(/\r?\n/)
    .map((line) => line.replace(/^[-*]\s*/, "").trim())
    .filter(Boolean);

  const collected = new Set<string>();

  for (const line of lines) {
    if (/^(explanation|these\b|because\b)/i.test(line)) {
      break;
    }

    collected.add(line);

    if (collected.size === requestedCount) {
      return Array.from(collected).join("\n");
    }
  }

  for (const item of context) {
    collected.add(item.title.trim());

    if (collected.size === requestedCount) {
      break;
    }
  }

  return Array.from(collected).slice(0, requestedCount).join("\n");
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 2 && !STOP_WORDS.has(token));
}

function getQuestionTokens(question: string): string[] {
  const lowerQuestion = question.toLowerCase();

  if (lowerQuestion.includes("list exactly")) {
    const topicMatch = lowerQuestion.match(/articles?\s+(?:about|on)\s+(.+?)(?:[.?!]|$)/);

    if (topicMatch?.[1]) {
      const topicTokens = tokenize(topicMatch[1]);

      if (topicTokens.length > 0) {
        return topicTokens;
      }
    }
  }

  return tokenize(question);
}

function computeLexicalScore(questionTokens: string[], item: RetrievedContext): number {
  if (questionTokens.length === 0) {
    return 0;
  }

  const titleTokens = new Set(tokenize(item.title));
  const tagTokens = new Set(tokenize(item.tags));
  const chunkTokens = new Set(tokenize(item.chunk).slice(0, 120));

  let score = 0;

  for (const token of questionTokens) {
    if (titleTokens.has(token)) score += 3;
    if (tagTokens.has(token)) score += 2.5;
    if (chunkTokens.has(token)) score += 1;
  }

  return score / questionTokens.length;
}

function rerankAndDedupe(question: string, matches: RetrievedContext[]): RetrievedContext[] {
  const questionTokens = getQuestionTokens(question);

  const ranked = matches
    .map((item) => {
      const lexicalScore = computeLexicalScore(questionTokens, item);
      const combinedScore = item.score + lexicalScore * 0.08;

      return {
        ...item,
        score: combinedScore,
      };
    })
    .sort((a, b) => b.score - a.score);

  const uniqueArticles = new Map<string, RetrievedContext>();

  for (const item of ranked) {
    if (!uniqueArticles.has(item.article_id)) {
      uniqueArticles.set(item.article_id, item);
    }
  }

  return Array.from(uniqueArticles.values()).slice(0, RAG_CONFIG.top_k);
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
    topK: RETRIEVAL_CANDIDATE_K,
    includeMetadata: true,
  });

  const retrievedMatches: RetrievedContext[] = (pineconeResults.matches ?? []).map(
    (match) => {
      const metadata = match.metadata as {
        article_id?: string;
        title?: string;
        chunk?: string;
        authors?: string;
        tags?: string;
      };

      return {
        article_id: String(metadata.article_id ?? ""),
        title: String(metadata.title ?? ""),
        authors: String(metadata.authors ?? ""),
        tags: String(metadata.tags ?? ""),
        chunk: String(metadata.chunk ?? ""),
        score: match.score ?? 0,
      };
    }
  );

  const context = rerankAndDedupe(question, retrievedMatches);

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
    formatTitleOnlyResponse(
      question,
      sanitizeModelResponse(
        chatResponse.choices[0]?.message?.content ??
          "I don't know based on the provided Medium articles data."
      ),
      context
    );

  return {
    response,
    context,
    Augmented_prompt: {
      System: SYSTEM_PROMPT,
      User: userPrompt,
    },
  };
}
