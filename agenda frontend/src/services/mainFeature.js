// src/services/mainFeature.js
import api from "./apiConfig";

/**
 * Fetch a hidden AI-generated answer visible only to the user.
 * @param {Object} payload - { question: string, context: string }
 */
export async function HiddenAnswers(payload) {
  const { data } = await api.post("/main/hidden-answer", payload);
  return data; // { answer: "private AI suggestion" }
}

/**
 * Optional: Poll for updates or live suggestions.
 */
export async function pollHiddenSuggestions(meetingId) {
  const { data } = await api.get(`/main/hidden-suggestions/${meetingId}`);
  return data; // { suggestions: [...] }
}
   