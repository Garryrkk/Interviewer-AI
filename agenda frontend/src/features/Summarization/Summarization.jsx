import { useState } from "react";
import { summarizeText } from "./summarizationUtils";

export default function Summarization() {
  const [input, setInput] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSummarize() {
    if (!input.trim()) return;
    setLoading(true);
    const result = await summarizeText(input);
    setSummary(result);
    setLoading(false);
  }

  return (
    <div>
      <textarea
        placeholder="Paste your long answer here..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />

      <button onClick={handleSummarize} disabled={loading}>
        {loading ? "Summarizing..." : "Summarize"}
      </button>

      {summary && (
        <div>
          <p>{summary}</p>
        </div>
      )}
    </div>
  );
}
