import { useState } from "react";
import { sendQuickReply } from "./quickRespondUtils";

export default function QuickRespond() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSend() {
    if (!input.trim()) return;
    setLoading(true);
    const reply = await sendQuickReply(input);
    setResponse(reply);
    setLoading(false);
  }

  return (
    <div>
      <textarea
        placeholder="Type your answer..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />

      <button onClick={handleSend} disabled={loading}>
        {loading ? "Thinking..." : "Respond"}
      </button>

      {response && (
        <div>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}
