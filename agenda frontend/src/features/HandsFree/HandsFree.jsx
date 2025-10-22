import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Video, VideoOff, Pause, Play, Square, Settings, Activity, Eye, Brain, Heart, Upload, RefreshCw, Database } from 'lucide-react';

import { API_BASE_URL, callEndpoint } from "../../services/apiConfig";
// api-configuration.js
const ENV = "development"; // or "production"

const BASE_URLS = {
  development: "http://localhost:8000",
  production: "https://api.myapp.com",
};

export const API_BASE_URL = BASE_URLS[ENV];


const HandsFreeInterviewSystem = () => {
  const [apiMode, setApiMode] = useState('original'); // 'original' or 'handsfree'
  const [sessionId, setSessionId] = useState(null);
  const [sessionStatus, setSessionStatus] = useState('inactive');
  const [handsFreeActive, setHandsFreeActive] = useState(false);
  const [micConfigured, setMicConfigured] = useState(false);
  const [aiReady, setAiReady] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [currentResponse, setCurrentResponse] = useState('');
  const [confidenceScore, setConfidenceScore] = useState(0);
  const [keyInsights, setKeyInsights] = useState([]);
  const [facialAnalysis, setFacialAnalysis] = useState(null);
  const [confidenceTips, setConfidenceTips] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  const [sessionInsights, setSessionInsights] = useState(null);
  const [detailedSessionStatus, setDetailedSessionStatus] = useState(null);
  const [activeWebsockets, setActiveWebsockets] = useState(0);
  const [selectedMicId, setSelectedMicId] = useState('0');
  const [showDataModal, setShowDataModal] = useState(false);
  const [modalContent, setModalContent] = useState(null);
  const [settings, setSettings] = useState({
    auto_response_enabled: true,
    response_delay: 2.0,
    confidence_coaching_enabled: true,
    facial_analysis_enabled: true,
    key_insights_only: true,
    voice_feedback_enabled: false,
    sensitivity_level: 0.7
  });

  const audioWsRef = useRef(null);
  const videoWsRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const videoIntervalRef = useRef(null);


  // ==========================================================================
  // ORIGINAL API ENDPOINTS
  // ==========================================================================
const context = { meetingId: "123", action: "start" };

  const createSessionOriginal = async () => {
    try {
      const params = new URLSearchParams({
        user_id: 'user123',
        default_mic_id: selectedMicId,
        interview_type: 'general',
        company_info: 'Tech Company',
        job_role: 'Software Developer'
      });

      const response = await fetch(`${API_BASE}/sessions/create?${params.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
        console.log('✅ [ORIGINAL] POST /sessions/create:', data);
        return data.session_id;
      } else {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create session');
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to create session:', error);
      alert('Failed to create session: ' + error.message);
    }
  };

  const configureAudioOriginal = async (sid) => {
    try {
      const params = new URLSearchParams({ mic_id: selectedMicId });
      const response = await fetch(`${API_BASE}/sessions/${sid}/configure-audio?${params.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        setMicConfigured(data.success);
        console.log('✅ [ORIGINAL] POST /configure-audio:', data);
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to configure audio:', error);
    }
  };

  const initializeAIOriginal = async (sid) => {
    try {
      const response = await fetch(`${API_BASE}/sessions/${sid}/initialize-ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        setAiReady(data.success);
        console.log('✅ [ORIGINAL] POST /initialize-ai:', data);
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to initialize AI:', error);
    }
  };

  const activateHandsFreeOriginal = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/sessions/${sessionId}/activate`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        setHandsFreeActive(true);
        setSessionStatus('hands_free_active');
        console.log('✅ [ORIGINAL] POST /activate:', data);
        
        await initializeWebSockets();
        await startVideoStream();
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to activate:', error);
      alert('Failed to activate: ' + error.message);
    }
  };

  const pauseSessionOriginal = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/sessions/${sessionId}/pause`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        setSessionStatus('paused');
        console.log('✅ [ORIGINAL] POST /pause:', data);
        
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.pause();
        }
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to pause:', error);
    }
  };

  const resumeSessionOriginal = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/sessions/${sessionId}/resume`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        setSessionStatus('hands_free_active');
        console.log('✅ [ORIGINAL] POST /resume:', data);
        
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'paused') {
          mediaRecorderRef.current.resume();
        }
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to resume:', error);
    }
  };

  const stopSessionOriginal = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/sessions/${sessionId}/stop`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        setSessionInsights(data.summary);
        console.log('✅ [ORIGINAL] POST /stop:', data);
      }
      cleanupResources();
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to stop:', error);
      cleanupResources();
    }
  };

  const getSessionStatusOriginal = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/sessions/${sessionId}/status`);
      if (response.ok) {
        const data = await response.json();
        setDetailedSessionStatus(data);
        console.log('✅ [ORIGINAL] GET /status:', data);
        if (data.status) setSessionStatus(data.status);
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to get status:', error);
    }
  };

  const getSessionInsightsOriginal = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/sessions/${sessionId}/insights`);
      if (response.ok) {
        const data = await response.json();
        console.log('✅ [ORIGINAL] GET /insights:', data);
        return data;
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to get insights:', error);
    }
  };

  const updateSettingsOriginal = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/sessions/${sessionId}/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ [ORIGINAL] PUT /settings:', data);
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to update settings:', error);
    }
  };

  const generateManualResponseOriginal = async () => {
    if (!sessionId || !currentQuestion) return;
    try {
      const params = new URLSearchParams({
        question: currentQuestion,
        context: 'Manual generation request'
      });

      const response = await fetch(`${API_BASE}/sessions/${sessionId}/generate-response?${params.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });

      if (response.ok) {
        const data = await response.json();
        setCurrentResponse(data.response_text || data.response);
        setKeyInsights(data.key_insights || []);
        setConfidenceScore(data.confidence_score || 0);
        console.log('✅ [ORIGINAL] POST /generate-response:', data);
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to generate response:', error);
    }
  };

  const analyzeFacialExpressionOriginal = async (file) => {
    if (!sessionId || !file) return;
    try {
      const formData = new FormData();
      formData.append('frame', file);

      const response = await fetch(`${API_BASE}/sessions/${sessionId}/analyze-facial`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setFacialAnalysis(data.analysis);
        setConfidenceTips(data.confidence_tips?.tips || []);
        console.log('✅ [ORIGINAL] POST /analyze-facial:', data);
      }
    } catch (error) {
      console.error('❌ [ORIGINAL] Failed to analyze facial:', error);
    }
  };

  // ==========================================================================
  // HANDS-FREE API ENDPOINTS
  // ==========================================================================

  const createSessionHandsFree = async () => {
    try {
      const response = await fetch(`${API_BASE}/hands-free/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'user123',
          default_mic_id: selectedMicId,
          interview_type: 'general',
          company_info: 'Tech Company',
          job_role: 'Software Developer'
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
        setMicConfigured(data.mic_configured);
        setAiReady(data.ai_ready);
        setSessionStatus(data.status);
        console.log('✅ [HANDS-FREE] POST /session/start:', data);
        return data.session_id;
      }
    } catch (error) {
      console.error('❌ [HANDS-FREE] Failed to create session:', error);
      alert('Failed to create session: ' + error.message);
    }
  };

  const activateHandsFreeMode = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/hands-free/session/${sessionId}/activate`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        setHandsFreeActive(true);
        setSessionStatus(data.status);
        console.log('✅ [HANDS-FREE] POST /activate:', data);
        
        await initializeWebSockets();
        await startVideoStream();
      }
    } catch (error) {
      console.error('Failed to activate hands-free mode:', error);
    }
  };

  // Initialize WebSocket connections
  const initializeWebSockets = async () => {
    // Audio WebSocket
    audioWsRef.current = new WebSocket(`ws://localhost:8000/hands-free/session/${sessionId}/audio-stream`);
    
    audioWsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'automated_response') {
        setCurrentQuestion(data.question);
        setCurrentResponse(data.response);
        setKeyInsights(data.key_insights || []);
        setConfidenceScore(data.confidence_score);
      } else if (data.type === 'status_update') {
        setIsListening(data.listening);
        setIsProcessing(data.processing);
        setAudioLevel(data.audio_level);
      }
    };

    // Video WebSocket
    videoWsRef.current = new WebSocket(`ws://localhost:8000/hands-free/session/${sessionId}/video-analysis`);
    
    videoWsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'facial_analysis_result') {
        setFacialAnalysis(data.analysis);
        setConfidenceTips(data.confidence_tips?.tips || []);
      } else if (data.type === 'error') {
        console.error('Video WebSocket error:', data.error);
      }
    };

    videoWsRef.current.onerror = (error) => {
      console.error('❌ Video WebSocket error:', error);
    };

    videoWsRef.current.onclose = () => {
      console.log('Video WebSocket disconnected');
      setActiveWebsockets(prev => Math.max(0, prev - 1));
    };

    await startAudioCapture();
  };

  const startAudioCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && audioWsRef.current?.readyState === WebSocket.OPEN) {
          audioWsRef.current.send(event.data);
        }
      };

      mediaRecorder.start(100);
      mediaRecorderRef.current = mediaRecorder;
      console.log('Audio capture started');
    } catch (error) {
      console.error('Failed to access microphone:', error);
      alert('Failed to access microphone. Please check permissions.');
    }
  };

  const startVideoStream = async () => {
    if (!settings.facial_analysis_enabled) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        } 
      });
      
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      const sendFrame = () => {
        if (videoRef.current && videoWsRef.current?.readyState === WebSocket.OPEN) {
          canvas.width = videoRef.current.videoWidth;
          canvas.height = videoRef.current.videoHeight;
          ctx.drawImage(videoRef.current, 0, 0);
          
          canvas.toBlob((blob) => {
            if (blob && videoWsRef.current?.readyState === WebSocket.OPEN) {
              videoWsRef.current.send(blob);
            }
          }, 'image/jpeg', 0.8);
        }
      };

      videoIntervalRef.current = setInterval(sendFrame, 1000);
    } catch (error) {
      console.error('Failed to access camera:', error);
    }
  };

  const getSystemStatusData = async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data);
        console.log('✅ GET /status:', data);
      }
    } catch (error) {
      console.error('❌ Failed to get system status:', error);
    }
  };

  const checkSystemHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      if (response.ok) {
        const data = await response.json();
        setSystemHealth({
          overall_status: data.status === 'operational' ? 'healthy' : 'degraded',
          active_sessions: data.active_sessions,
          active_websockets: data.active_websockets
        });
      }
    } catch (error) {
      console.error('Failed to resume session:', error);
    }
  };

  // Stop session
  const stopSession = async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(`${API_BASE}/session/${sessionId}/stop`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        setSessionInsights(data.session_summary);
      }

      // Cleanup
      audioWsRef.current?.close();
      videoWsRef.current?.close();
      streamRef.current?.getTracks().forEach(track => track.stop());
      
      setSessionId(null);
      setHandsFreeActive(false);
      setSessionStatus('inactive');
    } catch (error) {
      console.error('Failed to stop session:', error);
    }
  };

  // Check system health
  const checkSystemHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/system/health`);
      if (response.ok) {
        const health = await response.json();
        setSystemHealth(health);
      }
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  useEffect(() => {
    checkSystemHealth();
    getSystemStatusData();
    
    const interval = setInterval(() => {
      checkSystemHealth();
      getSystemStatusData();
      if (sessionId) {
        getSessionStatus();
      }
    }, 30000);
    
    return () => clearInterval(interval);
  }, [sessionId]);

  useEffect(() => {
    return () => {
      cleanupResources();
    };
  }, []);

  useEffect(() => {
    if (sessionId && handsFreeActive) {
      updateSettings();
    }
  }, [settings.auto_response_enabled, settings.facial_analysis_enabled]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'hands_free_active': return 'text-green-400';
      case 'active': return 'text-blue-400';
      case 'paused': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const confidenceColor = (score) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <div className="container mx-auto px-6 py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-white">Hands-Free Interview System</h1>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 bg-gray-800 rounded-lg px-4 py-2 border border-gray-700">
              <span className="text-sm text-gray-400">API Mode:</span>
              <button
                onClick={() => setApiMode('original')}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  apiMode === 'original' 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                }`}
              >
                Original
              </button>
              <button
                onClick={() => setApiMode('handsfree')}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  apiMode === 'handsfree' 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                }`}
              >
                Hands-Free
              </button>
            </div>
            <div className={`flex items-center space-x-2 ${getStatusColor(sessionStatus)}`}>
              <Activity className="w-5 h-5" />
              <span className="font-medium">{sessionStatus.replace('_', ' ').toUpperCase()}</span>
            </div>
            {systemHealth && (
              <div className={`flex items-center space-x-2 ${systemHealth.overall_status === 'healthy' ? 'text-green-400' : 'text-yellow-400'
                }`}>
                <Heart className="w-5 h-5" />
                <span>System {systemHealth.overall_status}</span>
              </div>
            )}
            {systemStatus && (
              <div className="flex items-center space-x-2 text-blue-400">
                <Activity className="w-5 h-5" />
                <span>{systemStatus.active_sessions || 0} sessions</span>
              </div>
            )}
            <button
              onClick={() => {
                checkSystemHealth();
                getSystemStatusData();
                if (sessionId) getSessionStatus();
              }}
              className="text-purple-400 hover:text-purple-300"
              title="Refresh status"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4 text-white">Session Control</h2>
              <div className="space-y-4">
                {!sessionId ? (
                  <>
                    <div className="mb-4">
                      <label className="block text-sm text-gray-400 mb-2">Microphone ID</label>
                      <input
                        type="text"
                        value={selectedMicId}
                        onChange={(e) => setSelectedMicId(e.target.value)}
                        className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:border-purple-500 focus:outline-none"
                        placeholder="Enter microphone ID"
                      />
                    </div>
                    <button
                      onClick={startSession}
                      className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-xl font-medium transition-colors duration-200 flex items-center justify-center space-x-2"
                    >
                      <Play className="w-5 h-5" />
                      <span>Start Session ({apiMode === 'original' ? 'Original API' : 'Hands-Free API'})</span>
                    </button>
                  </>
                ) : !handsFreeActive ? (
                  <button
                    onClick={activateHandsFree}
                    disabled={!micConfigured || !aiReady}
                    className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-3 px-6 rounded-xl font-medium transition-colors duration-200 flex items-center justify-center space-x-2"
                  >
                    <Brain className="w-5 h-5" />
                    <span>Activate Hands-Free</span>
                  </button>
                ) : (
                  <div className="space-y-3">
                    <button
                      onClick={sessionStatus === 'paused' ? resumeSession : pauseSession}
                      className="w-full bg-yellow-600 hover:bg-yellow-700 text-white py-3 px-6 rounded-xl font-medium transition-colors duration-200 flex items-center justify-center space-x-2"
                    >
                      {sessionStatus === 'paused' ? (
                        <>
                          <Play className="w-5 h-5" />
                          <span>Resume</span>
                        </>
                      ) : (
                        <>
                          <Pause className="w-5 h-5" />
                          <span>Pause</span>
                        </>
                      )}
                    </button>
                    <button
                      onClick={stopSession}
                      className="w-full bg-red-600 hover:bg-red-700 text-white py-3 px-6 rounded-xl font-medium transition-colors duration-200 flex items-center justify-center space-x-2"
                    >
                      <Square className="w-5 h-5" />
                      <span>Stop Session</span>
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4 text-white">System Status</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">API Mode</span>
                  <span className={`text-sm font-medium ${apiMode === 'original' ? 'text-blue-400' : 'text-purple-400'}`}>
                    {apiMode === 'original' ? 'ORIGINAL' : 'HANDS-FREE'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Session ID</span>
                  <span className="text-sm text-gray-300 font-mono">
                    {sessionId ? sessionId.substring(0, 8) + '...' : 'N/A'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Microphone</span>
                  <div className={`flex items-center space-x-2 ${micConfigured ? 'text-green-400' : 'text-red-400'}`}>
                    {micConfigured ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
                    <span>{micConfigured ? 'Ready' : 'Not Ready'}</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">AI Systems</span>
                  <div className={`flex items-center space-x-2 ${aiReady ? 'text-green-400' : 'text-red-400'}`}>
                    <Brain className="w-4 h-4" />
                    <span>{aiReady ? 'Ready' : 'Not Ready'}</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">WebSockets</span>
                  <div className={`flex items-center space-x-2 ${activeWebsockets > 0 ? 'text-green-400' : 'text-gray-400'}`}>
                    <Activity className="w-4 h-4" />
                    <span>{activeWebsockets} active</span>
                  </div>
                </div>
                {handsFreeActive && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Listening</span>
                      <div className={`flex items-center space-x-2 ${isListening ? 'text-green-400' : 'text-gray-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${isListening ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`}></div>
                        <span>{isListening ? 'Active' : 'Inactive'}</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Processing</span>
                      <div className={`flex items-center space-x-2 ${isProcessing ? 'text-yellow-400' : 'text-gray-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-yellow-400 animate-pulse' : 'bg-gray-400'}`}></div>
                        <span>{isProcessing ? 'Active' : 'Inactive'}</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {handsFreeActive && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h2 className="text-xl font-semibold mb-4 text-white">Audio Level</h2>
                <div className="w-full bg-gray-700 rounded-full h-4">
                  <div 
                    className="bg-purple-600 h-4 rounded-full transition-all duration-200"
                    style={{ width: `${audioLevel * 100}%` }}
                  ></div>
                </div>
                <p className="text-sm text-gray-400 mt-2">{Math.round(audioLevel * 100)}%</p>
              </div>
            )}

            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4 text-white">Settings</h2>
              <div className="space-y-3">
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-gray-400">Facial Analysis</span>
                  <input
                    type="checkbox"
                    checked={settings.facial_analysis_enabled}
                    onChange={(e) => setSettings({...settings, facial_analysis_enabled: e.target.checked})}
                    className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                  />
                </label>
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-gray-400">Auto Response</span>
                  <input
                    type="checkbox"
                    checked={settings.auto_response_enabled}
                    onChange={(e) => setSettings({...settings, auto_response_enabled: e.target.checked})}
                    className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                  />
                </label>
                <label className="flex items-center justify-between cursor-pointer">
                  <span className="text-gray-400">Confidence Coaching</span>
                  <input
                    type="checkbox"
                    checked={settings.confidence_coaching_enabled}
                    onChange={(e) => setSettings({...settings, confidence_coaching_enabled: e.target.checked})}
                    className="w-5 h-5 text-purple-600 rounded focus:ring-purple-500"
                  />
                </label>
              </div>
            </div>

            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <div className="flex items-center space-x-2 mb-4">
                <Database className="w-5 h-5 text-purple-400" />
                <h2 className="text-xl font-semibold text-white">Data Viewer</h2>
              </div>
              <div className="space-y-2">
                <button
                  onClick={async () => {
                    const data = await getAllInterviewResponses();
                    setModalContent({ title: 'Interview Responses', data });
                    setShowDataModal(true);
                  }}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg text-sm"
                >
                  Interview Responses
                </button>
                <button
                  onClick={async () => {
                    const data = await getAllFacialAnalyses();
                    setModalContent({ title: 'Facial Analyses', data });
                    setShowDataModal(true);
                  }}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg text-sm"
                >
                  Facial Analyses
                </button>
                <button
                  onClick={async () => {
                    const data = await getAllAudioStreamResults();
                    setModalContent({ title: 'Audio Stream Results', data });
                    setShowDataModal(true);
                  }}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg text-sm"
                >
                  Audio Results
                </button>
                <button
                  onClick={async () => {
                    const data = await getAllSessionSummaries();
                    setModalContent({ title: 'Session Summaries', data });
                    setShowDataModal(true);
                  }}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg text-sm"
                >
                  Session Summaries
                </button>
                <button
                  onClick={async () => {
                    const data = await getAllWebSocketMessages();
                    setModalContent({ title: 'WebSocket Messages', data });
                    setShowDataModal(true);
                  }}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg text-sm"
                >
                  WS Messages
                </button>
                <button
                  onClick={async () => {
                    const data = await getAllAutomatedResponses();
                    setModalContent({ title: 'Automated Responses', data });
                    setShowDataModal(true);
                  }}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg text-sm"
                >
                  Auto Responses
                </button>
                <button
                  onClick={apiMode === 'handsfree' ? getHandsFreeHealth : checkSystemHealth}
                  className="w-full bg-purple-700 hover:bg-purple-600 text-white py-2 px-4 rounded-lg text-sm font-medium"
                >
                  Check System Health
                </button>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            {handsFreeActive && settings.facial_analysis_enabled && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-white">Video Analysis</h2>
                  <div className="flex items-center space-x-2">
                    <Video className="w-6 h-6 text-purple-400" />
                    <label className="cursor-pointer bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg text-sm flex items-center space-x-2">
                      <Upload className="w-4 h-4" />
                      <span>Upload Frame</span>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                    </label>
                  </div>
                </div>
                <div className="relative bg-black rounded-xl overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    className="w-full h-64 object-cover"
                  />
                  {facialAnalysis && (
                    <div className="absolute top-4 right-4 bg-black bg-opacity-75 rounded-lg p-3 text-sm backdrop-blur-sm">
                      <div className={`font-medium ${confidenceColor(facialAnalysis.confidence_score || 0)}`}>
                        Confidence: {Math.round((facialAnalysis.confidence_score || 0) * 100)}%
                      </div>
                      {facialAnalysis.primary_emotion && (
                        <div className="text-gray-300">
                          Emotion: {facialAnalysis.primary_emotion}
                        </div>
                      )}
                      {facialAnalysis.engagement_level && (
                        <div className="text-gray-300">
                          Engagement: {facialAnalysis.engagement_level}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}

            {handsFreeActive && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h2 className="text-xl font-semibold mb-4 text-white">Current Interview</h2>

                {currentQuestion && (
                  <div className="mb-6">
                    <h3 className="text-lg font-medium text-purple-400 mb-2">Question:</h3>
                    <p className="text-gray-200 bg-gray-700 rounded-lg p-4">{currentQuestion}</p>
                  </div>
                )}

                {isProcessing && (
                  <div className="mb-6">
                    <div className="flex items-center space-x-3 text-yellow-400">
                      <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                      <span>Generating response...</span>
                    </div>
                  </div>
                )}

                {currentResponse && (
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-lg font-medium text-green-400">Response:</h3>
                      <div className={`font-medium ${confidenceColor(confidenceScore)}`}>
                        {Math.round(confidenceScore * 100)}% confidence
                      </div>
                    </div>
                    <p className="text-gray-200 bg-gray-700 rounded-lg p-4 mb-4">{currentResponse}</p>
                    
                    {keyInsights.length > 0 && (
                      <div>
                        <h4 className="text-md font-medium text-blue-400 mb-2">Key Insights:</h4>
                        <ul className="space-y-2">
                          {keyInsights.map((insight, index) => (
                            <li key={index} className="flex items-start space-x-2">
                              <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                              <span className="text-gray-300">{insight.point || insight}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {currentQuestion && (
                      <button
                        onClick={generateManualResponse}
                        className="mt-4 bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded-lg text-sm flex items-center space-x-2"
                      >
                        <RefreshCw className="w-4 h-4" />
                        <span>Regenerate ({apiMode === 'original' ? 'Original' : 'Hands-Free'})</span>
                      </button>
                    )}
                  </div>
                )}

                {!currentQuestion && !currentResponse && !isProcessing && (
                  <div className="text-center text-gray-400 py-8">
                    <Eye className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Listening for interview questions...</p>
                    <p className="text-sm mt-2">The system will automatically detect and respond to questions</p>
                  </div>
                )}
              </div>
            )}

            {confidenceTips.length > 0 && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h2 className="text-xl font-semibold mb-4 text-white">Confidence Tips</h2>
                <div className="space-y-3">
                  {confidenceTips.map((tip, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border-l-4 ${
                        tip.priority === 'high' ? 'bg-red-900 border-red-500' :
                        tip.priority === 'medium' ? 'bg-yellow-900 border-yellow-500' :
                        'bg-green-900 border-green-500'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="font-medium text-white">
                          {tip.tip_type ? tip.tip_type.replace('_', ' ').toUpperCase() : 'TIP'}
                        </div>
                        {tip.priority && (
                          <span className={`text-xs px-2 py-1 rounded ${
                            tip.priority === 'high' ? 'bg-red-600' :
                            tip.priority === 'medium' ? 'bg-yellow-600' :
                            'bg-green-600'
                          }`}>
                            {tip.priority.toUpperCase()}
                          </span>
                        )}
                      </div>
                      <p className="text-gray-200">{tip.message}</p>
                      {tip.immediate_action && (
                        <div className="text-sm text-yellow-400 mt-2 flex items-center space-x-1">
                          <span>⚡</span>
                          <span>Immediate action recommended</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {sessionId && !sessionInsights && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-white">Session Information</h2>
                  <button
                    onClick={async () => {
                      await getSessionStatus();
                      await getSessionInsights();
                    }}
                    className="text-purple-400 hover:text-purple-300 text-sm flex items-center space-x-1"
                  >
                    <RefreshCw className="w-4 h-4" />
                    <span>Refresh</span>
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Session ID</div>
                    <div className="text-lg font-medium text-white font-mono mt-1">
                      {sessionId.substring(0, 16)}...
                    </div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400">Status</div>
                    <div className={`text-lg font-medium mt-1 ${getStatusColor(sessionStatus)}`}>
                      {sessionStatus.replace('_', ' ')}
                    </div>
                  </div>
                  {systemStatus && (
                    <>
                      <div className="bg-gray-700 rounded-lg p-4">
                        <div className="text-sm text-gray-400">Active Sessions</div>
                        <div className="text-lg font-medium text-blue-400 mt-1">
                          {systemStatus.active_sessions || 0}
                        </div>
                      </div>
                      <div className="bg-gray-700 rounded-lg p-4">
                        <div className="text-sm text-gray-400">Active WebSockets</div>
                        <div className="text-lg font-medium text-purple-400 mt-1">
                          {systemStatus.active_websockets || 0}
                        </div>
                      </div>
                    </>
                  )}
                </div>
                {detailedSessionStatus && (
                  <div className="mt-4 bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-2">Detailed Status</div>
                    <pre className="text-xs text-gray-300 overflow-auto max-h-40">
                      {JSON.stringify(detailedSessionStatus, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {showDataModal && modalContent && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-gray-800 rounded-2xl p-6 max-w-4xl w-full max-h-96 overflow-y-auto border border-gray-700">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-white">{modalContent.title}</h2>
                <button
                  onClick={() => setShowDataModal(false)}
                  className="text-gray-400 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>
              <pre className="text-sm text-gray-300 bg-gray-900 rounded-lg p-4 overflow-auto">
                {JSON.stringify(modalContent.data, null, 2)}
              </pre>
              <button
                onClick={() => setShowDataModal(false)}
                className="mt-4 w-full bg-purple-600 hover:bg-purple-700 text-white py-2 px-4 rounded-xl font-medium"
              >
                Close
              </button>
            </div>
          </div>
        )}

        {sessionInsights && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-gray-800 rounded-2xl p-6 max-w-2xl w-full max-h-96 overflow-y-auto border border-gray-700">
              <h2 className="text-2xl font-bold text-white mb-4">Session Complete</h2>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-2xl font-bold text-purple-400">
                      {sessionInsights.total_duration ? Math.round(sessionInsights.total_duration) : 0} min
                    </div>
                    <div className="text-gray-400">Duration</div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-2xl font-bold text-green-400">
                      {sessionInsights.questions_handled || 0}
                    </div>
                    <div className="text-gray-400">Questions Handled</div>
                  </div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-400">
                    {sessionInsights.average_response_quality ? Math.round(sessionInsights.average_response_quality * 100) : 0}%
                  </div>
                  <div className="text-gray-400">Average Response Quality</div>
                </div>
                {sessionInsights.key_topics && sessionInsights.key_topics.length > 0 && (
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-sm text-gray-400 mb-2">Key Topics Discussed</div>
                    <div className="flex flex-wrap gap-2">
                      {sessionInsights.key_topics.map((topic, index) => (
                        <span key={index} className="bg-purple-600 text-white px-3 py-1 rounded-full text-sm">
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <button
                  onClick={() => setSessionInsights(null)}
                  className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-xl font-medium transition-colors duration-200"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HandsFreeInterviewSystem;