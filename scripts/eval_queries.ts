import dotenv from "dotenv";
import { answerQuestion } from "../lib/rag";

dotenv.config({ path: ".env.local" });

const QUESTIONS = [
  "Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable. Provide the title and author.",
  "List exactly 3 articles about education. Return only the titles.",
  "Find an article that argues past pandemics such as the bubonic plague can spur innovation and recovery, and summarise its central argument.",
  "I want practical, beginner-friendly advice on building habits that actually stick. Which article would you recommend, and why?",
];

async function main() {
  for (const question of QUESTIONS) {
    const result = await answerQuestion(question);

    console.log(`QUESTION: ${question}`);
    console.log(`RESPONSE: ${result.response}`);
    console.log("CONTEXT:");

    for (const item of result.context) {
      console.log(
        `- ${item.article_id} | ${item.title} | ${item.authors} | ${item.score}`
      );
    }

    console.log("---");
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
