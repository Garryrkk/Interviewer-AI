import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Play, Square, Volume2, Settings, Loader, CheckCircle, AlertCircle } from 'lucide-react';

// API Service Class
class VoiceProcessingAPI {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async startSession(userId) {
    const response = await fetch(`${this.baseUrl}/api/voice/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId })
    });
    return response.json();
  }

  async checkMicrophoneStatus(sessionId) {
    const response = await fetch(`${this.baseUrl}/api/voice/microphone/status/${sessionId}`);
    return response.json();
  }

  async getAudioDevices(sessionId) {
    const response = await fetch(`${this.baseUrl}/api/voice/devices/list/${sessionId}`);
    return response.json();
  }

  async selectDevice(sessionId, deviceId) {
    const response = await fetch(`${this.baseUrl}/api/voice/device/select`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, device_id: deviceId })
    });
    return response.json();
  }

  async toggleMicrophone(sessionId, turnOn, deviceId = null) {
    const response = await fetch(`${this.baseUrl}/api/voice/microphone/toggle`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, turn_on: turnOn, device_id: deviceId })
    });
    return response.json();
  }

  async processAudio(sessionId, audioFile) {
    const formData = new FormData();
    formData.append('audio_file', audioFile);
    
    const response = await fetch(`${this.baseUrl}/api/voice/audio/process?session_id=${sessionId}`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  }

  async getAIResponse(sessionId, question, responseFormat = 'summary', context = null) {
    const response = await fetch(`${this.baseUrl}/api/voice/ai/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        question: question,
        response_format: responseFormat,
        context: context
      })
    });
    return response.json();
  }

  async analyzeVoice(sessionId, audioData) {
    const response = await fetch(`${this.baseUrl}/api/voice/analyze/voice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        audio_data: audioData
      })
    });
    return response.json();
  }

  async endSession(sessionId) {
    const response = await fetch(`${this.baseUrl}/api/voice/session/${sessionId}`, {
      method: 'DELETE'
    });
    return response.json();
  }
}

// Main Voice Processing Component
const VoiceProcessingApp = () => {
  const [sessionId, setSessionId] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [micEnabled, setMicEnabled] = useState(false);
  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [voiceAnalysis, setVoiceAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [responseFormat, setResponseFormat] = useState('summary');
  
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const api = useRef(new VoiceProcessingAPI());

  // Initialize session on component mount
  useEffect(() => {
    initializeSession();
    return () => {
      if (sessionId) {
        api.current.endSession(sessionId);
      }
    };
  }, []);

  const initializeSession = async () => {
    try {
      setLoading(true);
      const userId = `user_${Date.now()}`;
      const response = await api.current.startSession(userId);
      
      if (response.success) {
        setSessionId(response.session_id);
        await loadAudioDevices(response.session_id);
      } else {
        setError('Failed to start session');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadAudioDevices = async (sessionId) => {
    try {
      const response = await api.current.getAudioDevices(sessionId);
      if (response.success) {
        setDevices(response.devices);
        if (response.default_device) {
          setSelectedDevice(response.default_device);
        }
      }
    } catch (err) {
      console.error('Failed to load devices:', err);
    }
  };

  const toggleMicrophone = async () => {
    if (!sessionId) return;

    try {
      setLoading(true);
      const response = await api.current.toggleMicrophone(
        sessionId, 
        !micEnabled, 
        selectedDevice?.id
      );

      if (response.success) {
        setMicEnabled(response.microphone_on);
        if (response.connected_device) {
          setSelectedDevice({ name: response.connected_device });
        }
      } else {
        setError(response.message);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    if (!micEnabled) {
      setError('Please enable microphone first');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        await processRecording(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      setError('Failed to start recording: ' + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processRecording = async (audioBlob) => {
    if (!sessionId) return;

    try {
      setLoading(true);
      const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
      
      // Process audio
      const processResponse = await api.current.processAudio(sessionId, audioFile);
      
      if (processResponse.success) {
        setTranscription(processResponse.transcription);
        
        // Get AI response
        const aiResponse = await api.current.getAIResponse(
          sessionId, 
          processResponse.transcription, 
          responseFormat
        );
        
        if (aiResponse.success) {
          setAiResponse(aiResponse.response);
        }

        // Analyze voice (convert blob to base64)
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            const base64Audio = reader.result.split(',')[1];
            const voiceAnalysis = await api.current.analyzeVoice(sessionId, base64Audio);
            if (voiceAnalysis.success) {
              setVoiceAnalysis(voiceAnalysis);
            }
          } catch (err) {
            console.error('Voice analysis failed:', err);
          }
        };
        reader.readAsDataURL(audioBlob);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const selectAudioDevice = async (device) => {
    if (!sessionId) return;

    try {
      setLoading(true);
      const response = await api.current.selectDevice(sessionId, device.id);
      
      if (response.success) {
        setSelectedDevice(device);
      } else {
        setError('Failed to select device');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-roboto">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-4">Voice Processing Assistant</h1>
          <p className="text-gray-300">Speak naturally and get AI-powered responses</p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-900 border border-red-600 rounded-lg p-4 mb-6 flex items-center">
            <AlertCircle className="mr-3 text-red-400" size={20} />
            <span>{error}</span>
            <button 
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-300"
            >
              ×
            </button>
          </div>
        )}

        {/* Session Status */}
        <div className="bg-gray-800 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Session Status</h2>
            <div className="flex items-center space-x-2">
              {sessionId ? (
                <CheckCircle className="text-green-400" size={20} />
              ) : (
                <AlertCircle className="text-yellow-400" size={20} />
              )}
              <span className="text-sm">
                {sessionId ? 'Active' : 'Disconnected'}
              </span>
            </div>
          </div>

          {/* Device Selection */}
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Audio Device</label>
              <select
                value={selectedDevice?.id || ''}
                onChange={(e) => {
                  const device = devices.find(d => d.id === e.target.value);
                  if (device) selectAudioDevice(device);
                }}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 focus:outline-none focus:border-purple-500"
              >
                <option value="">Select device...</option>
                {devices.map(device => (
                  <option key={device.id} value={device.id}>
                    {device.name} {device.is_default ? '(Default)' : ''}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Response Format</label>
              <select
                value={responseFormat}
                onChange={(e) => setResponseFormat(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 focus:outline-none focus:border-purple-500"
              >
                <option value="summary">Summary</option>
                <option value="detailed">Detailed</option>
                <option value="key_insights">Key Insights</option>
                <option value="bullet_points">Bullet Points</option>
              </select>
            </div>
          </div>
        </div>

        {/* Recording Controls */}
        <div className="bg-gray-800 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-center space-x-4">
            {/* Microphone Toggle */}
            <button
              onClick={toggleMicrophone}
              disabled={loading || !sessionId}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                micEnabled 
                  ? 'bg-green-600 hover:bg-green-700 text-white' 
                  : 'bg-gray-600 hover:bg-gray-700 text-gray-200'
              } disabled:opacity-50`}
            >
              {micEnabled ? <Mic size={20} /> : <MicOff size={20} />}
              <span>{micEnabled ? 'Microphone On' : 'Enable Microphone'}</span>
            </button>

            {/* Recording Button */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!micEnabled || loading}
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                isRecording 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-purple-600 hover:bg-purple-700 text-white'
              } disabled:opacity-50`}
            >
              {loading ? (
                <Loader className="animate-spin" size={20} />
              ) : isRecording ? (
                <Square size={20} />
              ) : (
                <Play size={20} />
              )}
              <span>
                {loading ? 'Processing...' : isRecording ? 'Stop Recording' : 'Start Recording'}
              </span>
            </button>
          </div>

          {/* Recording Indicator */}
          {isRecording && (
            <div className="flex items-center justify-center mt-4">
              <div className="animate-pulse flex items-center space-x-2 text-red-400">
                <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                <span>Recording...</span>
              </div>
            </div>
          )}
        </div>

        {/* Results Display */}
        <div className="space-y-6">
          {/* Transcription */}
          {transcription && (
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3">Transcription</h3>
              <p className="text-gray-200 bg-gray-700 rounded-lg p-4">{transcription}</p>
            </div>
          )}

          {/* AI Response */}
          {aiResponse && (
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3">AI Response</h3>
              <div className="text-gray-200 bg-gray-700 rounded-lg p-4 whitespace-pre-wrap">
                {aiResponse}
              </div>
            </div>
          )}

          {/* Voice Analysis */}
          {voiceAnalysis && (
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-3">Voice Analysis</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-medium mb-2">Confidence Rating</h4>
                  <div className="flex items-center space-x-2">
                    <div className="w-full bg-gray-700 rounded-full h-3">
                      <div 
                        className="bg-purple-600 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${(voiceAnalysis.confidence_rating / 10) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium">
                      {voiceAnalysis.confidence_rating}/10
                    </span>
                  </div>
                </div>
                <div>
                  <h4 className="font-medium mb-2">Voice Characteristics</h4>
                  <div className="space-y-1 text-sm text-gray-300">
                    <div>Volume: {(voiceAnalysis.voice_characteristics?.volume_level * 100).toFixed(0)}%</div>
                    <div>Speech Rate: {voiceAnalysis.voice_characteristics?.speech_rate.toFixed(0)} wpm</div>
                    <div>Clarity: {(voiceAnalysis.voice_characteristics?.clarity_score * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
              
              {voiceAnalysis.situational_tips && (
                <div className="mt-4">
                  <h4 className="font-medium mb-2">Tips</h4>
                  <ul className="space-y-1 text-sm text-gray-300">
                    {voiceAnalysis.situational_tips.map((tip, index) => (
                      <li key={index} className="flex items-start">
                        <span className="text-purple-400 mr-2">•</span>
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-400 text-sm">
          <p>Voice Processing Assistant - Powered by AI</p>
        </div>
      </div>
    </div>
  );
};

export default VoiceProcessingApp;