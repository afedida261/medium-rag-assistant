# Medium RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions about a provided dataset of roughly 7,600 English-language Medium articles.

Live app: https://medium-rag-assistant-seven.vercel.app/
GitHub: https://github.com/afedida261/medium-rag-assistant

## Overview

The assistant answers questions using only the provided Medium article dataset. Each article includes:

```text
title, text, url, authors, timestamp, tags
```

The system supports the required assignment query types:

1. Finding a specific article from semantic criteria
2. Listing up to 3 relevant article titles for a topic
3. Summarizing the key idea of a relevant article
4. Recommending an article with justification from retrieved evidence

If the answer is not supported by the retrieved Medium article context, the assistant is instructed to respond:

```text
I don't know based on the provided Medium articles data.
```

## RAG Configuration

The current configuration is exposed at:

```text
GET /api/stats
```

Current values:

```json
{
  "chunk_size": 640,
  "overlap_ratio": 0.2,
  "top_k": 7
}
```

Models used:

```text
Embedding: 4UHRUIN-text-embedding-3-small
Chat: 4UHRUIN-gpt-5-mini
```

## How It Works

The Medium articles are cleaned and split into overlapping chunks. Each chunk is embedded and stored in Pinecone with metadata such as article ID, title, authors, tags, URL, timestamp, and chunk text.

When a question is submitted, the system:

1. Embeds the user question
2. Retrieves candidate chunks from Pinecone
3. Reranks results using a lightweight lexical score over titles, tags, and chunk text
4. Deduplicates results by article ID
5. Sends the final retrieved context to the chat model
6. Returns the answer, retrieved context, and augmented prompt

## API Endpoints

### `GET /api/stats`

Returns the current RAG hyperparameters.

Example:

```bash
curl https://medium-rag-assistant-seven.vercel.app/api/stats
```

### `POST /api/prompt`

Queries the assistant.

Example:

```bash
curl -X POST "https://medium-rag-assistant-seven.vercel.app/api/prompt" \
  -H "Content-Type: application/json" \
  -d '{"question":"List exactly 3 articles about education. Return only the titles."}'
```

Input:

```json
{
  "question": "Your question here"
}
```

Output:

```json
{
  "response": "Final answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Article title",
      "authors": "Author name",
      "tags": "Article tags",
      "chunk": "Retrieved article passage",
      "score": 0.1234
    }
  ],
  "Augmented_prompt": {
    "System": "System prompt used for the chat model",
    "User": "User prompt with retrieved context"
  }
}
```

## Local Setup

Install dependencies:

```bash
npm install
```

Create a `.env.local` file:

```text
LLMOD_API_KEY=
LLMOD_BASE_URL=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
```

Run locally:

```bash
npm run dev
```

Build:

```bash
npm run build
```

## Data Processing

Create chunks from the Medium CSV:

```bash
python scripts/chunk_articles.py
```

Upload chunks to Pinecone:

```bash
npm run ingest
```

Run the evaluation questions:

```bash
npm run eval
```

## Test Questions

```text
Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author.
```

```text
List exactly 3 articles about education. Return only the titles.
```

```text
Find an article that argues past pandemics such as the bubonic plague can spur innovation and recovery, and summarize its central argument.
```

```text
I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?
```

Unsupported questions should not be answered from outside knowledge. For example:

```text
Who won the 2024 US presidential election?
```

Expected behavior:

```text
I don't know based on the provided Medium articles data.
```
