import json
import pandas as pd
from pathlib import Path


CSV_PATH = Path("data") / "medium-english-50mb.csv"
OUTPUT_PATH = Path("data") / "chunks_sample.jsonl"

CHUNK_SIZE = 512
OVERLAP_RATIO = 0.15    
OVERLAP = int(CHUNK_SIZE * OVERLAP_RATIO)
STEP = CHUNK_SIZE - OVERLAP

MAX_ARTICLES = 100  # start small for testing


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def chunk_text(text: str):
    words = text.split()
    chunks = []

    for start in range(0, len(words), STEP):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]

        if len(chunk_words) < 50:
            continue

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

    return chunks


def main():
    df = pd.read_csv(CSV_PATH)
    df = df.reset_index().rename(columns={"index": "article_id"})

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 200]

    if MAX_ARTICLES:
        df = df.head(MAX_ARTICLES)

    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with output_file.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            article_id = str(row["article_id"])
            chunks = chunk_text(row["text"])

            for i, chunk in enumerate(chunks):
                record = {
                    "id": f"{article_id}_{i}",
                    "article_id": article_id,
                    "chunk_index": i,
                    "title": row["title"],
                    "url": row["url"],
                    "authors": row["authors"],
                    "timestamp": row["timestamp"],
                    "tags": row["tags"],
                    "chunk": chunk,
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Created {total_chunks} chunks at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()