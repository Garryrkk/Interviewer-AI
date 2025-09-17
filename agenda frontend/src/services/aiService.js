// use environment variable from Vite
const BACKEND_BASE = import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, '') ?? '/api/v1';

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

export { postJSON };
