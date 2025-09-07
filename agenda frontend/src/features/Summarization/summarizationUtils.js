export async function summarizeText(text) {
  // Fake delay to simulate AI processing
  await new Promise((res) => setTimeout(res, 1000));

  // Simple mock: take first and last sentence
  const sentences = text.split(".");
  if (sentences.length > 2) {
    return (
      sentences[0].trim() +
      " ... " +
      sentences[sentences.length - 2].trim()
    );
  }

  // If text is too short, return a "mock summary"
  return "Summary: Focus on the key points and avoid extra details.";
}
