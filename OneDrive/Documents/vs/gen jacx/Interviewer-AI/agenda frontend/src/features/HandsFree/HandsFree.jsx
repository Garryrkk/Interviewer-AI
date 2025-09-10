import { useEffect, useState } from "react";
import { startListening } from "./handsFreeUtils";
import { sendQuickReply } from "../QuickRespond/quickRespondUtils";

export default function HandsFreeMode() {
  const [messages, setMessages] = useState([]); // chat history
  const [listening, setListening] = useState(false);

  useEffect(() => {
    if (listening) {
      // Start speech recognition
      startListening(async (transcript) => {
        // Add interviewer’s text
        setMessages((prev) => [...prev, { from: "interviewer", text: transcript }]);

        // AI generates quick response
        const aiReply = await sendQuickReply(transcript);
        setMessages((prev) => [...prev, { from: "ai", text: aiReply }]);
      });
    }
  }, [listening]);

  return (
    <div>
      <button onClick={() => setListening(!listening)}>
        {listening ? "Stop Hands-Free Mode" : "Start Hands-Free Mode"}
      </button>

      <div style={{ marginTop: "20px" }}>
        {messages.map((msg, index) => (
          <div key={index} style={{ margin: "6px 0" }}>
            <strong>{msg.from === "ai" ? "AI" : "Interviewer"}:</strong> {msg.text}
          </div>
        ))}
      </div>
    </div>
  );
}
