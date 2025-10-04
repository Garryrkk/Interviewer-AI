// use environment variable from Vite
const BACKEND_BASE =
  import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, "") ?? "/api/v1";

/**
 * Generic JSON POST helper
 * @param {string} path - API route after the base URL
 * @param {Object} body - Request payload
 */
async function postJSON(path, body) {
  const url = `${BACKEND_BASE}${path}`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Request failed ${resp.status}: ${text}`);
  }
  return await resp.json();
}

/* ------------------------------------------------------------------
   🟢  Feature-specific API functions
   Each one uses postJSON but stays isolated so you can import them
   individually inside your React components.
-------------------------------------------------------------------*/

/**
 * Quick Respond
 * @param {string} prompt - user prompt or question
 */
export async function QuickRespond(prompt) {
  return postJSON("/ai/quick-respond", { prompt });
}

/**
 * Summarization
 * @param {string} text - full transcript or meeting notes
 */
export async function Summarization(text) {
  return postJSON("/ai/summarize", { text });
}

/**
 * Key Insights
 * @param {string} text - content to extract insights from
 */
export async function KeyInsights(text) {
  return postJSON("/ai/key-insights", { text });
}

/** 
 * Hands Free   
 * @param {Object} context - { meetingId, action, ... }
 * Designed for triggering hands-free AI actions
 */
export async function HandsFreeMode(context) {
  return postJSON("/ai/hands-free", context);
}

// Export the generic helper too, if needed elsewhere
export { postJSON };
