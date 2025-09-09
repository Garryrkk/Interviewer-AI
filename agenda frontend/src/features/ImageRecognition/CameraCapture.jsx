import React, { useEffect, useRef, useState } from "react";
import {
  captureCameraFrame,
  blobToBase64,
  detectExpression,
} from "./cameraCaptureUtils";
import { sendQuickReply } from "../QuickRespond/quickRespondUtils";
import { summarizeText } from "../Summarization/summarizationUtils"; 

export default function CameraCapture() {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [running, setRunning] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [expression, setExpression] = useState(null);
  const [messages, setMessages] = useState([]);
  const detectorLockRef = useRef(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => stopStream();
  }, []);

  async function startStream() {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play().catch(() => {});
      }
      setRunning(true);

      // Run expression detection every 2s
      intervalRef.current = setInterval(() => {
        if (!detectorLockRef.current) detectFromCamera();
      }, 2000);
    } catch (err) {
      console.error("Camera access failed:", err);
      alert("Unable to access camera: " + (err.message || err));
    }
  }

  function stopStream() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
    setStream(null);
    setRunning(false);
    clearInterval(intervalRef.current);
  }

  async function detectFromCamera() {
    if (detectorLockRef.current) return;
    detectorLockRef.current = true;
    setProcessing(true);

    try {
      const frameBlob = await captureCameraFrame(videoRef.current);
      const b64 = await blobToBase64(frameBlob);

      // Expression detection (mock for now)
      const expr = await detectExpression(b64);
      setExpression(expr);

      // Add to chat log
      setMessages((m) => [
        ...m,
        { id: crypto.randomUUID(), from: "system", text: `Expression detected: ${expr.label} (${expr.confidence})` },
      ]);

      // If confused → simplify answer
      if (expr.label === "confused" && expr.confidence > 0.6) {
        const latestAI = messages.filter((x) => x.from === "ai").slice(-1)[0];
        if (latestAI) {
          const simpler = await summarizeText(latestAI.text, { simplify: true });
          const reply = await sendQuickReply(simpler);

          setMessages((m) => [
            ...m,
            { id: crypto.randomUUID(), from: "ai", text: reply },
          ]);
        }
      }
    } catch (err) {
      console.error("Camera detect error:", err);
    } finally {
      detectorLockRef.current = false;
      setProcessing(false);
    }
  }

  return (
    <div>
      <div style={{ marginBottom: 8 }}>
        <button onClick={running ? stopStream : startStream}>
          {running ? "Stop Camera" : "Start Camera"}
        </button>
        {processing && <span style={{ marginLeft: 8 }}>Detecting…</span>}
      </div>

      <video ref={videoRef} style={{ width: 320, height: 240, border: "1px solid #ddd" }} muted playsInline />

      <div style={{ marginTop: 12 }}>
        <h4>Detected Expression</h4>
        {expression ? (
          <div>
            {expression.label} ({expression.confidence})
          </div>
        ) : (
          <div>No expression yet</div>
        )}
      </div>

      <div style={{ marginTop: 12, border: "1px solid #eee", padding: 8, borderRadius: 6, maxHeight: 240, overflowY: "auto" }}>
        <h4>AI Chat</h4>
        {messages.length === 0 && <div>No messages yet</div>}
        {messages.map((m) => (
          <div key={m.id} style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 12, color: "#888" }}>{m.from.toUpperCase()}</div>
            <div style={{ background: m.from === "ai" ? "#eef6ff" : "#f7f7f7", padding: 6, borderRadius: 6 }}>
              {m.text}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
