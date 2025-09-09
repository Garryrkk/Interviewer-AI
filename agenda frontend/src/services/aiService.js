
const API_URL = "http://localhost:11434/api/generate";
const MODEL = "nous-hermes"; // the name you registered in Ollama

// Core function: send prompt to local model
async function queryModel(prompt) {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        prompt: prompt,
      }),
    });

    const data = await response.json();
    return data.response || "⚠️ No response from AI";
  } catch (err) {
    console.error("AI Service Error:", err);
    return "⚠️ Error connecting to local AI";
  }
}

// ----------------------
// Feature APIs
// ----------------------

// Quick Respond
export async function quickRespond(question) {
  const prompt = `You are helping in a live meeting. Answer this question clearly and concisely:\n\n${question}`;
  return await queryModel(prompt);
}

// Summarization
export async function summarizeText(text) {
  const prompt = `Summarize the following text in 3-4 bullet points:\n\n${text}`;
  return await queryModel(prompt);
}

// Key Insights
export async function extractKeyInsights(text) {
  const prompt = `From the following text, extract key insights and main takeaways:\n\n${text}`;
  return await queryModel(prompt);
}

// Hands Free (works with voice input later)
export async function handsFreeCommand(command) {
  const prompt = `You are in hands-free mode. User said: "${command}". 
  Interpret and respond naturally as if you are assisting live.`;
  return await queryModel(prompt);
}
