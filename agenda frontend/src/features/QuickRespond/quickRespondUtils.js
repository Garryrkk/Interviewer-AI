// Later, replace this with real AI API call
export async function sendQuickReply(userInput) {
  await new Promise((res) => setTimeout(res, 800)); // simulate delay

  const suggestions = [
    "Keep your answer concise and highlight key achievements.",
    "Use the STAR method: Situation, Task, Action, Result.",
    "Don’t forget to show confidence and enthusiasm.",
  ];

  return suggestions[Math.floor(Math.random() * suggestions.length)];
}
