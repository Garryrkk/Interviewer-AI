import React, { useState } from "react";

// ✅ Base backend URL
const BACKEND_URL = "http://127.0.0.1:8000";

function ExampleFeature() {
  const [inputText, setInputText] = useState("");
  const [response, setResponse] = useState("");

  const callBackend = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/ping`); // Replace /ping with your real endpoint
      const data = await res.json();
      setResponse(JSON.stringify(data));
      console.log("Backend response:", data);
    } catch (err) {
      console.error("API call failed:", err);
      setResponse("Error connecting to backend. Check console.");
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-4">Example Feature</h2>

      <input
        className="border p-2 mr-2"
        placeholder="Enter text"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      />
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded"
        onClick={callBackend}
      >
        Call Backend
      </button>

      <p className="mt-4">Response: {response}</p>
    </div>
  );
}

export default ExampleFeature;
