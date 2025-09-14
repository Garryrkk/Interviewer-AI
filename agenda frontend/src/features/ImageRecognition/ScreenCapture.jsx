import React, { useEffect, useRef, useState } from "react";
import {
  captureScreenFrame,
  blobToBase64,
  recognizeImage,
  generateInsightFromRecognition,
  formatRecognitionResult,
} from "./ScreenCaptureUtils";
import { sendQuickReply } from "../QuickRespond/quickRespondUtils";
import { 
  Monitor, 
  Play, 
  Square, 
  Camera, 
  Activity, 
  Eye, 
  EyeOff, 
  Copy,
  Circle,
  Zap
} from 'lucide-react';

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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-slate-100 mb-2 flex items-center space-x-3">
              <Monitor className="text-blue-500" size={40} />
              <span>Screen Capture</span>
            </h1>
            <p className="text-slate-400">Real-time screen monitoring and AI analysis</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium ${
              running ? 'bg-green-600 text-white' : 'bg-slate-700 text-slate-300'
            }`}>
              <Circle size={8} className={`fill-current ${running ? 'text-green-300' : 'text-slate-500'}`} />
              <span>{running ? 'Active' : 'Standby'}</span>
            </div>
          </div>
        </div>

        {/* Control Panel */}
        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-xl font-semibold mb-4 text-slate-200">Control Panel</h3>
          <div className="flex flex-wrap gap-4">
            <button 
              onClick={running ? stopStream : startStream}
              className={`flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all ${
                running 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {running ? <Square size={20} /> : <Play size={20} />}
              <span>{running ? "Stop Screen Sharing" : "Start Screen Sharing"}</span>
            </button>

            <button 
              onClick={handleSnapshot} 
              disabled={!running || processing}
              className={`flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all ${
                !running || processing 
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed' 
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              <Camera size={20} />
              <span>{processing ? "Processing…" : "Capture Now"}</span>
              {processing && <Activity size={16} className="animate-spin" />}
            </button>

            <button 
              onClick={toggleAutoDetect}
              className={`flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all ${
                autoDetect 
                  ? 'bg-purple-600 hover:bg-purple-700 text-white' 
                  : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
              }`}
            >
              {autoDetect ? <Eye size={20} /> : <EyeOff size={20} />}
              <span>{autoDetect ? "Auto-Detect: ON" : "Auto-Detect: OFF"}</span>
            </button>

            <button 
              onClick={copyLatest} 
              disabled={!messages.length}
              className={`flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all ${
                !messages.length 
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed' 
                  : 'bg-orange-600 hover:bg-orange-700 text-white'
              }`}
            >
              <Copy size={20} />
              <span>Copy Latest</span>
            </button>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Left Column - Video and Snapshot */}
          <div className="xl:col-span-2 space-y-6">
            {/* Video Stream */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                <Monitor className="mr-2" size={20} />
                Live Screen Stream
              </h3>
              <div className="bg-slate-900/80 rounded-xl overflow-hidden border-2 border-slate-700">
                <video 
                  ref={videoRef} 
                  className="w-full h-64 object-cover" 
                  muted 
                  playsInline 
                  style={{ backgroundColor: '#1e293b' }}
                />
              </div>
            </div>

            {/* Last Snapshot */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                <Camera className="mr-2" size={20} />
                Latest Snapshot
              </h3>
              <div className="bg-slate-900/80 rounded-xl overflow-hidden border-2 border-dashed border-slate-600 min-h-48">
                {lastSnapshot ? (
                  <img
                    alt="Latest snapshot"
                    src={URL.createObjectURL(lastSnapshot)}
                    className="w-full h-auto object-cover rounded-lg"
                  />
                ) : (
                  <div className="flex items-center justify-center h-48 text-slate-400">
                    <div className="text-center">
                      <Camera size={48} className="mx-auto mb-2 text-slate-500" />
                      <p>No snapshots captured yet</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column - Recognition History and Chat */}
          <div className="space-y-6">
            {/* Recognition History */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                <Zap className="mr-2" size={20} />
                Recognition History
              </h3>
              <div className="bg-slate-900/80 rounded-lg p-4 max-h-80 overflow-y-auto">
                {recognitions.length === 0 ? (
                  <div className="text-center text-slate-400 py-8">
                    <Activity size={48} className="mx-auto mb-2 text-slate-500" />
                    <p>No detections yet</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {recognitions.map((r) => (
                      <div key={r.id} className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
                        <div className="font-semibold text-slate-200 mb-2">{r.title}</div>
                        <div className="text-sm text-slate-300 mb-3">{r.summary}</div>
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-slate-400">
                            Confidence: <span className="text-blue-400">{(r.confidence || 0).toFixed(2)}</span>
                          </span>
                          <span className="px-2 py-1 bg-slate-700 text-slate-300 rounded-full">
                            {r.category}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* AI Chat */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                <Activity className="mr-2" size={20} />
                AI Chat
              </h3>
              <div className="bg-slate-900/80 rounded-lg p-4 max-h-64 overflow-y-auto">
                {messages.length === 0 ? (
                  <div className="text-center text-slate-400 py-8">
                    <div className="text-center">
                      <Circle size={32} className="mx-auto mb-2 text-slate-500" />
                      <p>No messages yet</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {messages.map((m) => (
                      <div key={m.id} className="space-y-1">
                        <div className="text-xs font-medium text-slate-400 uppercase tracking-wide">
                          {m.from}
                        </div>
                        <div className={`p-3 rounded-lg text-sm ${
                          m.from === "ai" 
                            ? "bg-blue-600/20 border border-blue-600/30 text-blue-100" 
                            : m.from === "system"
                            ? "bg-purple-600/20 border border-purple-600/30 text-purple-100"
                            : "bg-slate-700/50 border border-slate-600 text-slate-200"
                        }`}>
                          {m.text}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}