import React, { useEffect, useRef, useState } from "react";
import {
  captureScreenFrame,
  blobToBase64,
  recognizeImage,
  generateInsightFromRecognition,
  formatRecognitionResult,
} from "./screenCaptureUtils";
import { sendQuickReply } from "../QuickRespond/quickRespondUtils";

/**
 * Hello Bimari Gamari read from here
 * ScreenCapture component
 * - Starts/stops screen sharing
 * - Captures frames (single or interval)
 * - Runs image recognition on captured frames
 * - Converts recognition => AI quick response and shows in chat UI
 */
export default function ScreenCapture() {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [running, setRunning] = useState(false);
  const [autoDetect, setAutoDetect] = useState(true);
  const [lastSnapshot, setLastSnapshot] = useState(null);
  const [recognitions, setRecognitions] = useState([]); // history of recognition results
  const [messages, setMessages] = useState([]); // chat-like UI of interviewer/ai
  const [processing, setProcessing] = useState(false);
  const detectorLockRef = useRef(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => {
      stopStream();
      clearInterval(intervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startStream() {
    try {
      const mediaStream = await navigator.mediaDevices.getDisplayMedia({
        video: { cursor: "always" },
        audio: false,
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play().catch(() => {});
      }
      setRunning(true);

      // auto-capture periodically if autoDetect enabled
      if (autoDetect) {
        intervalRef.current = setInterval(() => {
          if (!detectorLockRef.current) doCaptureAndRecognize();
        }, 1500); // throttle to once every 1.5s
      }
    } catch (err) {
      console.error("Screen capture failed:", err);
      alert("Unable to start screen capture: " + (err.message || err));
    }
  }

  async function stopStream() {
    try {
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    } finally {
      setStream(null);
      setRunning(false);
      clearInterval(intervalRef.current);
    }
  }

  async function doCaptureAndRecognize({ manualTrigger = false } = {}) {
    // avoid concurrent processing
    if (detectorLockRef.current) return;
    detectorLockRef.current = true;
    setProcessing(true);

    try {
      const frameBlob = await captureScreenFrame(videoRef.current);
      if (!frameBlob) {
        detectorLockRef.current = false;
        setProcessing(false);
        return;
      }
      setLastSnapshot(frameBlob);

      // Convert to base64 for sending to backend or to show inline
      const b64 = await blobToBase64(frameBlob);

      // Recognize image (mock or real API inside recognizeImage)
      const recognition = await recognizeImage(b64);

      // Create an insight-like object (similar to KeyInsights style)
      const insight = generateInsightFromRecognition(recognition);

      // Add to recognition history
      setRecognitions((s) => [insight, ...s].slice(0, 20));

      // Format result for AI quick response (send to quick respond)
      const formatted = formatRecognitionResult(insight);

      // Add interviewer-like message (what the UI shows as detected question context)
      setMessages((m) => [
        ...m,
        { id: crypto.randomUUID(), from: "system", text: "Detected: " + (insight.title || "screen content") },
      ]);

      // Ask QuickRespond (or your chat service) for an immediate answer
      const aiReply = await sendQuickReply(formatted);
      setMessages((m) => [
        ...m,
        { id: crypto.randomUUID(), from: "ai", text: aiReply },
      ]);

    } catch (err) {
      console.error("Capture/recognize error:", err);
    } finally {
      detectorLockRef.current = false;
      setProcessing(false);
    }
  }

  // Manual snapshot handler
  async function handleSnapshot() {
    await doCaptureAndRecognize({ manualTrigger: true });
  }

  // Toggle auto-detect on the fly
  function toggleAutoDetect() {
    setAutoDetect((v) => {
      const next = !v;
      if (next) {
        // start interval if running
        if (running && !intervalRef.current) {
          intervalRef.current = setInterval(() => {
            if (!detectorLockRef.current) doCaptureAndRecognize();
          }, 1500);
        }
      } else {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return next;
    });
  }

  // Copy the last AI message or recognition
  function copyLatest() {
    const latest = messages.length ? messages[messages.length - 1] : null;
    const text = latest ? `${latest.from.toUpperCase()}: ${latest.text}` : "No messages";
    navigator.clipboard?.writeText(text).then(() => {
      // optional: add a small system message
      setMessages((m) => [...m, { id: crypto.randomUUID(), from: "system", text: "Copied latest to clipboard" }]);
    });
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <button onClick={running ? stopStream : startStream}>
          {running ? "Stop Screen Sharing" : "Start Screen Sharing"}
        </button>

        <button onClick={handleSnapshot} disabled={!running || processing}>
          {processing ? "Processing…" : "Capture Now"}
        </button>

        <button onClick={toggleAutoDetect}>
          {autoDetect ? "Auto-Detect: ON" : "Auto-Detect: OFF"}
        </button>

        <button onClick={copyLatest} disabled={!messages.length}>
          Copy Latest
        </button>
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ flex: 1, minWidth: 320 }}>
          {/* Video element shows the shared screen stream */}
          <div style={{ border: "1px solid #ddd", borderRadius: 6, overflow: "hidden" }}>
            <video ref={videoRef} style={{ width: "100%", height: 240 }} muted playsInline />
          </div>

          {/* Last snapshot preview */}
          <div style={{ marginTop: 8 }}>
            <strong>Last Snapshot:</strong>
            {lastSnapshot ? (
              <img
                alt="snapshot"
                src={URL.createObjectURL(lastSnapshot)}
                style={{ display: "block", maxWidth: "100%", marginTop: 6, borderRadius: 6 }}
              />
            ) : (
              <div style={{ color: "#666", marginTop: 6 }}>No snapshots yet</div>
            )}
          </div>
        </div>

        {/* Right column: recognitions history & chat UI */}
        <div style={{ width: 420 }}>
          <div style={{ border: "1px solid #eee", padding: 8, borderRadius: 6, maxHeight: 420, overflowY: "auto" }}>
            <h4>Recognition History</h4>
            {recognitions.length === 0 && <div style={{ color: "#666" }}>No detections yet</div>}
            {recognitions.map((r) => (
              <div key={r.id} style={{ padding: 8, borderBottom: "1px solid #fafafa" }}>
                <div style={{ fontWeight: 600 }}>{r.title}</div>
                <div style={{ fontSize: 13, color: "#333" }}>{r.summary}</div>
                <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>
                  Confidence: {(r.confidence || 0).toFixed(2)} · Category: {r.category}
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 12, border: "1px solid #eee", padding: 8, borderRadius: 6, maxHeight: 300, overflowY: "auto" }}>
            <h4>AI Chat</h4>
            {messages.length === 0 && <div style={{ color: "#666" }}>No messages yet</div>}
            {messages.map((m) => (
              <div key={m.id} style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 12, color: "#888" }}>{m.from.toUpperCase()}</div>
                <div style={{ background: m.from === "ai" ? "#eef6ff" : "#f7f7f7", padding: 8, borderRadius: 6 }}>
                  {m.text}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
