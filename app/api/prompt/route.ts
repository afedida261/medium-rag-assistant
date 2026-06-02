import { answerQuestion } from "@/lib/rag";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const question = body.question;

    if (!question || typeof question !== "string") {
      return Response.json(
        { error: "Missing or invalid question field." },
        { status: 400 }
      );
    }

    const result = await answerQuestion(question);

    return Response.json(result);
  } catch (error) {
    console.error(error);

    return Response.json(
      { error: "Failed to process prompt." },
      { status: 500 }
    );
  }
}