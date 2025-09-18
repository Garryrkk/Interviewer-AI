import React, { useEffect, useRef, useState } from "react";
import {
  captureCameraFrame,
  blobToBase64,
  detectExpression,
} from "./cameraCaptureUtils";
import { sendQuickReply } from "../QuickRespond/quickRespondUtils";
import { summarizeText } from "../Summarization/summarizationUtils";
import { Camera, Play, Square, Activity, MessageSquare, Eye, Brain, Zap } from 'lucide-react';

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
      // First, send request to backend
      const backendResponse = await fetch('/api/camera/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: 'someId' })
      });
      
      if (!backendResponse.ok) {
        throw new Error('Backend denied camera access');
      }

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

  async function stopStream() {
    try {
      // Send request to backend to stop camera
      await fetch('/api/camera/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: 'someId' })
      });
    } catch (err) {
      console.error("Backend stop request failed:", err);
    }

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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
              <Camera size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-100">Camera Expression Detection</h1>
              <p className="text-slate-400">Real-time facial expression analysis and AI response system</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${
              running ? 'bg-green-600/20 text-green-400 border border-green-600/30' : 'bg-slate-700/50 text-slate-400 border border-slate-600'
            }`}>
              <div className={`w-2 h-2 rounded-full ${running ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`}></div>
              <span className="text-sm font-medium">{running ? 'Camera Active' : 'Camera Inactive'}</span>
            </div>
            
            {processing && (
              <div className="flex items-center space-x-2 px-4 py-2 bg-blue-600/20 text-blue-400 border border-blue-600/30 rounded-full">
                <Activity size={16} className="animate-spin" />
                <span className="text-sm font-medium">Processing...</span>
              </div>
            )}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          
          {/* Camera Feed Section */}
          <div className="xl:col-span-2 grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* Camera Controls & Feed */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-slate-200 flex items-center">
                  <Camera className="mr-3" size={20} />
                  Camera Feed
                </h3>
                <div className={`px-3 py-1 text-xs font-medium rounded-full ${
                  running ? 'bg-green-600 text-white' : 'bg-slate-600 text-slate-300'
                }`}>
                  {running ? 'LIVE' : 'OFFLINE'}
                </div>
              </div>
              
              {/* Video Container */}
              <div className="bg-slate-900/80 p-4 rounded-lg mb-6 relative overflow-hidden">
                <video 
                  ref={videoRef} 
                  className="w-full h-48 bg-slate-800 rounded-lg object-cover"
                  muted 
                  playsInline 
                  style={{ 
                    display: running ? 'block' : 'none',
                  }}
                />
                {!running && (
                  <div className="w-full h-48 bg-slate-800 rounded-lg flex items-center justify-center">
                    <div className="text-center">
                      <Camera size={48} className="text-slate-600 mx-auto mb-3" />
                      <p className="text-slate-500 text-sm">Camera feed will appear here</p>
                    </div>
                  </div>
                )}
                
                {/* Processing Overlay */}
                {processing && (
                  <div className="absolute inset-0 bg-blue-600/10 rounded-lg flex items-center justify-center">
                    <div className="bg-slate-900/90 px-4 py-2 rounded-lg flex items-center space-x-2">
                      <Activity size={16} className="text-blue-400 animate-spin" />
                      <span className="text-blue-400 text-sm font-medium">Analyzing Expression...</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Camera Controls */}
              <div className="grid grid-cols-1 gap-3">
                <button 
                  onClick={running ? stopStream : startStream}
                  className={`flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all ${
                    running 
                      ? 'bg-red-600 hover:bg-red-700 text-white' 
                      : 'bg-blue-600 hover:bg-blue-700 text-white'
                  }`}
                >
                  {running ? (
                    <>
                      <Square size={18} />
                      <span>Stop Camera</span>
                    </>
                  ) : (
                    <>
                      <Play size={18} />
                      <span>Start Camera</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Expression Analysis */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-xl font-semibold mb-6 text-slate-200 flex items-center">
                <Brain className="mr-3" size={20} />
                Expression Analysis
              </h3>
              
              <div className="bg-slate-900/80 p-6 rounded-lg">
                {expression ? (
                  <div className="space-y-4">
                    <div className="text-center">
                      <div className="w-20 h-20 bg-gradient-to-r from-purple-600 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Eye size={32} className="text-white" />
                      </div>
                      <h4 className="text-2xl font-bold text-slate-200 capitalize mb-2">
                        {expression.label}
                      </h4>
                      <div className="flex items-center justify-center space-x-2 mb-4">
                        <span className="text-slate-400 text-sm">Confidence:</span>
                        <span className="text-slate-200 font-semibold">{expression.confidence}</span>
                      </div>
                    </div>
                    
                    {/* Confidence Bar */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">Accuracy</span>
                        <span className="text-slate-300">{(parseFloat(expression.confidence) * 100).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-slate-700 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${parseFloat(expression.confidence) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Eye size={32} className="text-slate-600" />
                    </div>
                    <p className="text-slate-500">No expression detected yet</p>
                    <p className="text-slate-600 text-sm mt-2">Start camera to begin analysis</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Chat/Messages Panel */}
          <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
            <h3 className="text-xl font-semibold mb-6 text-slate-200 flex items-center justify-between">
              <div className="flex items-center">
                <MessageSquare className="mr-3" size={20} />
                AI Response Log
              </div>
              <div className="px-3 py-1 bg-slate-700 text-slate-300 text-xs rounded-full">
                {messages.length} messages
              </div>
            </h3>
            
            <div className="bg-slate-900/80 rounded-lg min-h-96 max-h-96 overflow-y-auto p-4 space-y-3">
              {messages.length === 0 ? (
                <div className="text-center py-12">
                  <Zap size={48} className="text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-500">No messages yet</p>
                  <p className="text-slate-600 text-sm mt-2">AI responses will appear here</p>
                </div>
              ) : (
                messages.map((message) => (
                  <div key={message.id} className="space-y-2">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${
                        message.from === 'ai' ? 'bg-blue-500' : 
                        message.from === 'system' ? 'bg-purple-500' : 'bg-slate-500'
                      }`}></div>
                      <span className={`text-xs font-medium uppercase tracking-wider ${
                        message.from === 'ai' ? 'text-blue-400' :
                        message.from === 'system' ? 'text-purple-400' : 'text-slate-400'
                      }`}>
                        {message.from}
                      </span>
                    </div>
                    <div className={`p-4 rounded-lg ${
                      message.from === 'ai' ? 'bg-blue-600/10 border border-blue-600/20' :
                      message.from === 'system' ? 'bg-purple-600/10 border border-purple-600/20' :
                      'bg-slate-700/50 border border-slate-600/50'
                    }`}>
                      <p className="text-slate-200 text-sm leading-relaxed">{message.text}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Status Bar */}
        <div className="bg-slate-800/30 backdrop-blur p-4 rounded-xl border border-slate-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${running ? 'bg-green-500' : 'bg-slate-500'}`}></div>
                <span className="text-slate-300 text-sm">Camera Status: {running ? 'Active' : 'Inactive'}</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${processing ? 'bg-blue-500 animate-pulse' : 'bg-slate-500'}`}></div>
                <span className="text-slate-300 text-sm">Processing: {processing ? 'Active' : 'Standby'}</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-purple-500"></div>
                <span className="text-slate-300 text-sm">AI Assistant: Ready</span>
              </div>
            </div>
            <div className="text-slate-400 text-sm">
              Detection Interval: 2.0s
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}