import dotenv from "dotenv";
import { Pinecone } from "@pinecone-database/pinecone";

dotenv.config({ path: ".env.local" });

async function main() {
  const apiKey = process.env.PINECONE_API_KEY;
  const indexName = process.env.PINECONE_INDEX_NAME;

  if (!apiKey) throw new Error("Missing PINECONE_API_KEY");
  if (!indexName) throw new Error("Missing PINECONE_INDEX_NAME");

  const pc = new Pinecone({ apiKey });
  const index = pc.index(indexName);

  const stats = await index.describeIndexStats();

  console.log("Connected to Pinecone.");
  console.log(JSON.stringify(stats, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});