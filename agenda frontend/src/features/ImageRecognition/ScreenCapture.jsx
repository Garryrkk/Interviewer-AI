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
  Zap,
  Wifi,
  WifiOff,
  AlertCircle
} from 'lucide-react';
import { ScreenCapture } from "../../services/imageService";
// Backend Communication Layer
const screenCaptureAPI = {
  startCapture: async (config) => {
    const response = await fetch('/api/screen-capture/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return response.json();
  },
  stopCapture: async (sessionId) => {
    return fetch(`/api/screen-capture/stop/${sessionId}`, { method: 'POST' });
  },
  sendFrame: async (frameData, sessionId) => {
    return fetch('/api/screen-capture/frame', {
      method: 'POST',
      body: frameData,
      headers: { 'X-Session-ID': sessionId }
    });
  }
};

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
  const wsRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [running, setRunning] = useState(false);
  const [autoDetect, setAutoDetect] = useState(true);
  const [lastSnapshot, setLastSnapshot] = useState(null);
  const [recognitions, setRecognitions] = useState([]); // history of recognition results
  const [messages, setMessages] = useState([]); // chat-like UI of interviewer/ai
  const [processing, setProcessing] = useState(false);
  
  // Backend Integration State
  const [sessionId, setSessionId] = useState(null);
  const [backendStatus, setBackendStatus] = useState('disconnected'); // connected, processing, error
  const [permissions, setPermissions] = useState({
    screen: null,
    camera: null,
    microphone: null
  });
  
  const detectorLockRef = useRef(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    return () => {
      stopStream();
      clearInterval(intervalRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Permission Handling System
  async function requestPermissions() {
    try {
      // Request screen capture permission
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { cursor: "always" },
        audio: false
      });
      
      setPermissions(prev => ({...prev, screen: 'granted'}));
      return stream;
    } catch (error) {
      setPermissions(prev => ({...prev, screen: 'denied'}));
      throw error;
    }
  }

  // Backend Service Activation
  async function activateBackendServices() {
    try {
      setBackendStatus('connecting');
      
      const response = await fetch('/api/services/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          services: ['screen_capture', 'image_recognition', 'ai_analysis'],
          config: {
            captureInterval: 1500,
            analysisMode: 'realtime',
            autoDetect: autoDetect
          }
        })
      });
      
      if (!response.ok) throw new Error('Failed to activate backend services');
      
      const { sessionId: newSessionId } = await response.json();
      setSessionId(newSessionId);
      setBackendStatus('connected');
      
      return newSessionId;
    } catch (error) {
      setBackendStatus('error');
      throw error;
    }
  }

  // Real-time Backend Synchronization
  function establishWebSocketConnection(sessionId) {
    wsRef.current = new WebSocket(`ws://localhost:8000/ws/screen-capture/${sessionId}`);
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'recognition_result':
          setRecognitions(prev => [data.payload, ...prev].slice(0, 20));
          break;
        case 'ai_response':
          setMessages(prev => [...prev, {
            id: crypto.randomUUID(),
            from: 'ai',
            text: data.payload.response
          }]);
          break;
        case 'backend_status':
          setBackendStatus(data.payload.status);
          break;
        default:
          break;
      }
    };

    wsRef.current.onclose = () => {
      setBackendStatus('disconnected');
    };

    wsRef.current.onerror = () => {
      setBackendStatus('error');
    };
  }

  function startAutoCaptureLoop(sessionId) {
    if (autoDetect) {
      intervalRef.current = setInterval(() => {
        if (!detectorLockRef.current) doCaptureAndRecognize();
      }, 1500); // throttle to once every 1.5s
    }
  }

  async function handleStartupError(error) {
    setBackendStatus('error');
    setMessages(prev => [...prev, {
      id: crypto.randomUUID(),
      from: 'system',
      text: `Startup Error: ${error.message}`
    }]);
  }

  // Enhanced Start/Stop Flow
  async function startStream() {
    try {
      // 1. Request permissions
      const mediaStream = await requestPermissions();
      
      // 2. Activate backend services
      const sessionId = await activateBackendServices();
      
      // 3. Establish WebSocket connection
      establishWebSocketConnection(sessionId);
      
      // 4. Start local stream
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }
      setRunning(true);
      
      // 5. Notify backend that stream is ready
      await fetch('/api/screen-capture/ready', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, status: 'ready' })
      });
      
      // 6. Start auto-capture if enabled
      if (autoDetect) {
        startAutoCaptureLoop(sessionId);
      }
      
    } catch (error) {
      console.error("Failed to start screen capture:", error);
      await handleStartupError(error);
    }
  }

  async function stopStream() {
    try {
      // 1. Stop local stream
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      
      // 2. Close WebSocket
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      // 3. Deactivate backend services
      if (sessionId) {
        await fetch('/api/services/deactivate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sessionId })
        });
      }
      
      // 4. Clean up state
      setStream(null);
      setSessionId(null);
      setRunning(false);
      setBackendStatus('disconnected');
      clearInterval(intervalRef.current);
      
    } catch (error) {
      console.error("Error stopping stream:", error);
    }
  }

  function handleImmediateResult(result) {
    if (result.recognition) {
      setRecognitions(prev => [result.recognition, ...prev].slice(0, 20));
    }
    if (result.aiResponse) {
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        from: 'ai',
        text: result.aiResponse
      }]);
    }
  }

  // Frame Transmission to Backend
  async function doCaptureAndRecognize({ manualTrigger = false } = {}) {
    if (detectorLockRef.current || !sessionId) return;
    
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

      // Create FormData for multipart upload
      const formData = new FormData();
      formData.append('frame', frameBlob, 'screenshot.png');
      formData.append('sessionId', sessionId);
      formData.append('timestamp', Date.now().toString());
      formData.append('manualTrigger', manualTrigger.toString());

      // Send to backend for processing
      const response = await fetch('/api/screen-capture/process-frame', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Backend processing failed: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Backend will send results via WebSocket, but we can also handle immediate response
      if (result.immediate) {
        handleImmediateResult(result);
      }

      // Fallback to local processing if backend fails
      if (!result.success) {
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
      }

    } catch (error) {
      console.error("Frame processing error:", error);
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        from: 'system',
        text: `Error: ${error.message}`
      }]);
    } finally {
      detectorLockRef.current = false;
      setProcessing(false);
    }
  }

  // Manual snapshot handler
  async function handleSnapshot() {
    await doCaptureAndRecognize({ manualTrigger: true });
  }

  // Configuration Sync with Backend
  async function updateBackendConfig(config) {
    if (!sessionId) return;
    
    try {
      await fetch('/api/screen-capture/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          config: {
            autoDetect,
            captureInterval: 1500,
            recognitionThreshold: 0.7,
            ...config
          }
        })
      });
    } catch (error) {
      console.error('Failed to update backend config:', error);
    }
  }

  // Toggle auto-detect on the fly
  function toggleAutoDetect() {
    setAutoDetect((v) => {
      const next = !v;
      updateBackendConfig({ autoDetect: next }); // Sync with backend
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

        {/* Backend Status Monitor */}
        <div className="bg-slate-800/50 backdrop-blur p-4 rounded-xl border border-slate-700">
          <h4 className="text-sm font-semibold text-slate-200 mb-2">Backend Status</h4>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 text-sm ${
                backendStatus === 'connected' ? 'text-green-400' : 
                backendStatus === 'connecting' ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {backendStatus === 'connected' ? <Wifi size={16} /> : 
                 backendStatus === 'connecting' ? <Activity size={16} className="animate-spin" /> :
                 <WifiOff size={16} />}
                <span className="capitalize">{backendStatus}</span>
              </div>
              {sessionId && (
                <span className="text-xs text-slate-400">
                  Session: {sessionId.slice(0, 8)}...
                </span>
              )}
            </div>
            <div className="flex items-center space-x-2">
              <div className={`flex items-center space-x-1 text-xs ${
                permissions.screen === 'granted' ? 'text-green-400' : 
                permissions.screen === 'denied' ? 'text-red-400' : 'text-slate-400'
              }`}>
                {permissions.screen === 'granted' ? <Circle size={6} className="fill-current" /> :
                 permissions.screen === 'denied' ? <AlertCircle size={12} /> :
                 <Circle size={6} className="text-slate-500" />}
                <span>Screen</span>
              </div>
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
              className={`flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all`}
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
              onClick={() => {
                console.log("copyLatest clicked");
                copyLatest();
              }}
              className={`flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all `}
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