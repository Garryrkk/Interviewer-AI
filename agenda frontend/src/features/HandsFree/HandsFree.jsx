import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Video, VideoOff, Pause, Play, Square, Settings, Activity, Eye, Brain, Heart } from 'lucide-react';

const HandsFreeInterviewSystem = () => {
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
  const [confidenceTips, setConfenceTips] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [sessionInsights, setSessionInsights] = useState(null);
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

  const API_BASE = 'http://localhost:8000/hands-free';

  // Initialize system
  const startSession = async () => {
    try {
      const response = await fetch(`${API_BASE}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'user123',
          default_mic_id: '0',
          interview_type: 'general',
          company_info: 'Tech Company',
          job_role: 'Software Developer'
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
        setSessionStatus(data.status);
        setMicConfigured(data.mic_configured);
        setAiReady(data.ai_ready);
      }
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  };

  // Activate hands-free mode
  const activateHandsFree = async () => {
    if (!sessionId) return;

    try {
      const response = await fetch(`${API_BASE}/session/${sessionId}/activate`, {
        method: 'POST'
      });

      if (response.ok) {
        setHandsFreeActive(true);
        setSessionStatus('hands_free_active');
        await initializeWebSockets();
        await startVideoStream();
      }
    } catch (error) {
      console.error('Failed to activate hands-free mode:', error);
    }
  };

  // Get session status
  const getSessionStatus = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/session/${sessionId}/status`);
      if (response.ok) {
        const status = await response.json();
        setSessionStatus(status.status); // or set other details if needed
      }
    } catch (error) {
      console.error("Failed to fetch session status:", error);
    }
  };

  // Update settings
  const updateSettings = async (newSettings) => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/session/${sessionId}/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newSettings),
      });
      if (response.ok) {
        const result = await response.json();
        setSettings(result.settings);
      }
    } catch (error) {
      console.error("Failed to update settings:", error);
    }
  };

  // Get insights manually (without stopping session)
  const fetchInsights = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${API_BASE}/session/${sessionId}/insights`);
      if (response.ok) {
        const insights = await response.json();
        setSessionInsights(insights);
      }
    } catch (error) {
      console.error("Failed to fetch insights:", error);
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
      setFacialAnalysis(data.facial_analysis);
      setConfenceTips(data.confidence_tips?.tips || []);
    };

    // Start audio capture
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (event) => {
        if (audioWsRef.current?.readyState === WebSocket.OPEN) {
          audioWsRef.current.send(event.data);
        }
      };

      mediaRecorder.start(100); // Send data every 100ms
    } catch (error) {
      console.error('Failed to access microphone:', error);
    }
  };

  // Start video stream
  const startVideoStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Send video frames for analysis
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

      setInterval(sendFrame, 1000); // Send frame every second
    } catch (error) {
      console.error('Failed to access camera:', error);
    }
  };

  // Emergency pause
  const emergencyPause = async () => {
    if (!sessionId) return;

    try {
      await fetch(`${API_BASE}/session/${sessionId}/emergency-pause`, {
        method: 'POST'
      });
      setHandsFreeActive(false);
      setSessionStatus('paused');
    } catch (error) {
      console.error('Failed to pause session:', error);
    }
  };

  // Resume hands-free
  const resumeHandsFree = async () => {
    if (!sessionId) return;

    try {
      await fetch(`${API_BASE}/session/${sessionId}/resume`, {
        method: 'POST'
      });
      setHandsFreeActive(true);
      setSessionStatus('hands_free_active');
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
    const interval = setInterval(checkSystemHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

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
    <div className="min-h-screen bg-gray-900 text-gray-100 font-roboto">
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-white">Hands-Free Interview System</h1>
          <div className="flex items-center space-x-4">
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
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Control Panel */}
          <div className="lg:col-span-1 space-y-6">
            {/* Session Controls */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4 text-white">Session Control</h2>
              <div className="space-y-4">
                {!sessionId ? (
                  <button
                    onClick={startSession}
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-xl font-medium transition-colors duration-200 flex items-center justify-center space-x-2"
                  >
                    <Play className="w-5 h-5" />
                    <span>Start Session</span>
                  </button>
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
                      onClick={sessionStatus === 'paused' ? resumeHandsFree : emergencyPause}
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
                          <span>Emergency Pause</span>
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

            {/* System Status */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <h2 className="text-xl font-semibold mb-4 text-white">System Status</h2>
              <div className="space-y-3">
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
                {handsFreeActive && (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Listening</span>
                      <div className={`flex items-center space-x-2 ${isListening ? 'text-green-400' : 'text-gray-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${isListening ? 'bg-green-400' : 'bg-gray-400'}`}></div>
                        <span>{isListening ? 'Active' : 'Inactive'}</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-400">Processing</span>
                      <div className={`flex items-center space-x-2 ${isProcessing ? 'text-yellow-400' : 'text-gray-400'}`}>
                        <div className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-yellow-400' : 'bg-gray-400'}`}></div>
                        <span>{isProcessing ? 'Active' : 'Inactive'}</span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Audio Level */}
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
          </div>

          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Video Feed */}
            {handsFreeActive && settings.facial_analysis_enabled && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-white">Video Analysis</h2>
                  <Video className="w-6 h-6 text-purple-400" />
                </div>
                <div className="relative bg-black rounded-xl overflow-hidden">
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    className="w-full h-64 object-cover"
                  />
                  {facialAnalysis && (
                    <div className="absolute top-4 right-4 bg-black bg-opacity-50 rounded-lg p-3 text-sm">
                      <div className={`font-medium ${confidenceColor(facialAnalysis.confidence_score)}`}>
                        Confidence: {Math.round(facialAnalysis.confidence_score * 100)}%
                      </div>
                      <div className="text-gray-300">
                        Emotion: {facialAnalysis.primary_emotion}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Current Interview */}
            {handsFreeActive && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h2 className="text-xl font-semibold mb-4 text-white">Current Interview</h2>

                {currentQuestion && (
                  <div className="mb-6">
                    <h3 className="text-lg font-medium text-purple-400 mb-2">Question:</h3>
                    <p className="text-gray-200 bg-gray-700 rounded-lg p-4">{currentQuestion}</p>
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
                              <div className="w-2 h-2 bg-blue-400 rounded-full mt-2"></div>
                              <span className="text-gray-300">{insight.point}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {!currentQuestion && !currentResponse && (
                  <div className="text-center text-gray-400 py-8">
                    <Eye className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Listening for interview questions...</p>
                  </div>
                )}
              </div>
            )}

            {/* Confidence Tips */}
            {confidenceTips.length > 0 && (
              <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                <h2 className="text-xl font-semibold mb-4 text-white">Confidence Tips</h2>
                <div className="space-y-3">
                  {confidenceTips.map((tip, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg border-l-4 ${tip.priority === 'high' ? 'bg-red-900 border-red-500' :
                          tip.priority === 'medium' ? 'bg-yellow-900 border-yellow-500' :
                            'bg-green-900 border-green-500'
                        }`}
                    >
                      <div className="font-medium text-white mb-1">{tip.tip_type.replace('_', ' ').toUpperCase()}</div>
                      <p className="text-gray-200">{tip.message}</p>
                      {tip.immediate_action && (
                        <div className="text-sm text-yellow-400 mt-1">⚡ Immediate action recommended</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Session Insights Modal */}
        {sessionInsights && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-gray-800 rounded-2xl p-6 max-w-2xl w-full max-h-96 overflow-y-auto">
              <h2 className="text-2xl font-bold text-white mb-4">Session Complete</h2>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-2xl font-bold text-purple-400">{Math.round(sessionInsights.total_duration)} min</div>
                    <div className="text-gray-400">Duration</div>
                  </div>
                  <div className="bg-gray-700 rounded-lg p-4">
                    <div className="text-2xl font-bold text-green-400">{sessionInsights.questions_handled}</div>
                    <div className="text-gray-400">Questions Handled</div>
                  </div>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="text-2xl font-bold text-blue-400">{Math.round(sessionInsights.average_response_quality * 100)}%</div>
                  <div className="text-gray-400">Average Response Quality</div>
                </div>
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