import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Play, Square, Volume2, Settings, Loader, CheckCircle, AlertCircle } from 'lucide-react';
import {VoiceProcessingAPI} from './VoicePreoputils'
import {VoiceRecognition} from '../../services/voiceService'
import { VoiceProcessing } from '../../services/voiceService';
// VoiceRecognition component
export function VoiceRecognition() { 
  const [transcript, setTranscript] = useState(""); 
  const recognitionRef = useRef(null);

  const startLiveVoiceRecognition = (setTranscript) => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    recognition.onresult = (event) => {
      let finalTranscript = '';
      let interimTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }
      
      setTranscript(finalTranscript + interimTranscript);
    };
    
    recognition.start();
    return recognition;
  };

  useEffect(() => { 
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) { 
      setTranscript("Browser does not support live recognition."); 
      return; 
    } 
    recognitionRef.current = startLiveVoiceRecognition(setTranscript); 
    return () => recognitionRef.current && recognitionRef.current.stop(); 
  }, []);

  return ( 
    <div className="p-4 bg-white shadow-md rounded-md"> 
      <h2 className="text-lg font-semibold mb-2">Live Voice Recognition</h2> 
      <p className="whitespace-pre-wrap">{transcript}</p> 
    </div> 
  ); 
}

// VoiceProcessing component
export function VoiceProcessing() { 
  const [recording, setRecording] = useState(false); 
  const [transcript, setTranscript] = useState(""); 
  const mediaRecorderRef = useRef(null); 
  const chunksRef = useRef([]);

  const transcribeAudio = async (audioBlob) => {
    // Mock transcription function - in real implementation, this would call an API
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({ transcript: "Mock transcription result" });
      }, 1000);
    });
  };

  const startRecording = async () => { 
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); 
    const mediaRecorder = new MediaRecorder(stream); 
    chunksRef.current = []; 
    mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data); 
    mediaRecorder.onstop = async () => { 
      const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" }); 
      const result = await transcribeAudio(audioBlob); 
      setTranscript(result.transcript || "(No text recognized)"); 
    }; 
    mediaRecorder.start(); 
    mediaRecorderRef.current = mediaRecorder; 
    setRecording(true); 
  };

  const stopRecording = () => { 
    if (mediaRecorderRef.current) { 
      mediaRecorderRef.current.stop(); 
      setRecording(false); 
    } 
  };

  return ( 
    <div className="p-4 bg-white shadow-md rounded-md"> 
      <h2 className="text-lg font-semibold mb-2">Voice Processing</h2> 
      <button onClick={recording ? stopRecording : startRecording} className={`px-4 py-2 rounded-md text-white ${ recording ? "bg-red-500" : "bg-green-500" }`} > {recording ? "Stop Recording" : "Start Recording"} </button> 
      {transcript && ( 
        <p className="mt-3 text-gray-700"> 
          <strong>Transcript:</strong> {transcript} 
        </p> 
      )} 
    </div> 
  ); 
}

// API Service Class - STILL HERE, NOT REMOVED!
class VoiceProcessingAPI {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  // Session Management
  async startSession(userId, meetingId = null) {
    const response = await fetch(`${this.baseUrl}/api/voice/session/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, meeting_id: meetingId })
    });
    return response.json();
  }

  async endSession(sessionId) {
    const response = await fetch(`${this.baseUrl}/api/voice/session/${sessionId}`, {
      method: 'DELETE'
    });
    return response.json();
  }

  // Microphone Management
  async checkMicrophoneStatus(sessionId) {
    const response = await fetch(`${this.baseUrl}/api/voice/microphone/status/${sessionId}`);
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

  // Device Management
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

  // Audio Processing
  async processAudio(sessionId, audioFile) {
    const formData = new FormData();
    formData.append('audio_file', audioFile);

    const response = await fetch(`${this.baseUrl}/api/voice/audio/process?session_id=${sessionId}`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  }

  async transcribeAudio(audioData, language = 'auto', modelSize = 'base') {
    const response = await fetch(`${this.baseUrl}/api/voice/transcribe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_data: audioData,
        language: language,
        model_size: modelSize
      })
    });
    return response.json();
  }

  async transcribeUpload(audioFile, language = 'auto', modelSize = 'base') {
    const formData = new FormData();
    formData.append('file', audioFile);
    formData.append('language', language);
    formData.append('model_size', modelSize);
    
    const response = await fetch(`${this.baseUrl}/api/voice/transcribe/upload`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  }

  // AI Response
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

  async simplifyResponse(originalResponse, simplificationLevel = 'basic') {
    const response = await fetch(`${this.baseUrl}/api/voice/ai/simplify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        original_response: originalResponse,
        simplification_level: simplificationLevel
      })
    });
    return response.json();
  }

  // Voice Analysis
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

  // Audio Calibration & Testing
  async calibrateAudio(duration = 3, sampleRate = 16000, channels = 1) {
    const response = await fetch(`${this.baseUrl}/api/v1/audio/calibrate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        duration: duration,
        sample_rate: sampleRate,
        channels: channels
      })
    });
    return response.json();
  }

  async testRecording(duration = 5, sampleRate = 16000, channels = 1, applyCalibration = true) {
    const response = await fetch(`${this.baseUrl}/api/v1/audio/test-record`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        duration: duration,
        sample_rate: sampleRate,
        channels: channels,
        apply_calibration: applyCalibration
      })
    });
    return response.json();
  }

  async getSupportedFormats() {
    const response = await fetch(`${this.baseUrl}/api/voice/supported-formats`);
    return response.json();
  }

  async calibrateAudio(settings) {
    const response = await fetch(`${this.baseUrl}/api/voice/calibrate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    return response.json();
  }

  async testAudioRecording(settings) {
    const response = await fetch(`${this.baseUrl}/api/voice/test-record`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    });
    return response.json();
  }

  async getCalibrationStatus() {
    const response = await fetch(`${this.baseUrl}/api/voice/calibration-status`);
    return response.json();
  }

  async resetCalibration() {
    const response = await fetch(`${this.baseUrl}/api/voice/reset-calibration`, {
      method: 'DELETE',
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
  const fileInputRef = useRef(null);
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

  const checkMicStatus = async () => {
    if (!sessionId) return;
    try {
      const response = await api.current.checkMicrophoneStatus(sessionId);
      if (response.success) {
        setMicEnabled(response.microphone_on);
      }
    } catch (err) {
      console.error('Failed to check mic status:', err);
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

  const handleCalibration = async () => {
    try {
      setIsCalibrating(true);
      setError(null);
      
      const response = await api.current.calibrateAudio(3, 16000, 1);
      
      if (response.success || response.noise_level !== undefined) {
        setCalibrationData(response);
        setError(null);
      } else {
        setError('Calibration failed');
      }
    } catch (err) {
      setError('Calibration error: ' + err.message);
    } finally {
      setIsCalibrating(false);
    }
  };

  const handleTestRecording = async () => {
    try {
      setIsTesting(true);
      setError(null);
      
      const response = await api.current.testRecording(5, 16000, 1, !!calibrationData);
      
      if (response.success || response.quality_score !== undefined) {
        setTestRecordingData(response);
        setError(null);
      } else {
        setError('Test recording failed');
      }
    } catch (err) {
      setError('Test recording error: ' + err.message);
    } finally {
      setIsTesting(false);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      setLoading(true);
      setError(null);
      
      const response = await api.current.transcribeUpload(file, uploadLanguage, uploadModelSize);
      
      if (response.success) {
        setTranscription(response.transcription || response.text);
        
        // Get AI response for the transcription
        if (sessionId) {
          const aiResp = await api.current.getAIResponse(
            sessionId,
            response.transcription || response.text,
            responseFormat
          );
          if (aiResp.success) {
            setAiResponse(aiResp.response);
          }
        }
      } else {
        setError('Transcription failed: ' + (response.message || 'Unknown error'));
      }
    } catch (err) {
      setError('File upload error: ' + err.message);
    } finally {
      setLoading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSimplifyResponse = async () => {
    if (!aiResponse) return;

    try {
      setLoading(true);
      const response = await api.current.simplifyResponse(aiResponse, simplificationLevel);
      
      if (response.success) {
        setAiResponse(response.simplified_response);
      } else {
        setError('Simplification failed');
      }
    } catch (err) {
      setError('Simplification error: ' + err.message);
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
    <div className="min-h-screen bg-gray-900 text-gray-100">
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
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowUpload(!showUpload)}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-700 hover:bg-blue-600 rounded-lg transition-colors"
              >
                <Upload size={16} />
                <span className="text-sm">Upload Audio</span>
              </button>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                <Settings size={16} />
                <span className="text-sm">Audio Setup</span>
              </button>
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
          </div>

          {/* Upload Audio Section */}
          {showUpload && (
            <div className="border-t border-gray-700 pt-4 mt-4">
              <h3 className="text-lg font-semibold mb-4">Upload Audio File</h3>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="grid md:grid-cols-3 gap-4 mb-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Language</label>
                    <select
                      value={uploadLanguage}
                      onChange={(e) => setUploadLanguage(e.target.value)}
                      className="w-full bg-gray-600 border border-gray-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="auto">Auto Detect</option>
                      <option value="en">English</option>
                      <option value="es">Spanish</option>
                      <option value="fr">French</option>
                      <option value="de">German</option>
                      <option value="hi">Hindi</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Model Size</label>
                    <select
                      value={uploadModelSize}
                      onChange={(e) => setUploadModelSize(e.target.value)}
                      className="w-full bg-gray-600 border border-gray-500 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                    >
                      <option value="tiny">Tiny (Fast)</option>
                      <option value="base">Base</option>
                      <option value="small">Small</option>
                      <option value="medium">Medium</option>
                      <option value="large">Large (Best)</option>
                    </select>
                  </div>
                  <div className="flex items-end">
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={loading}
                      className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50"
                    >
                      <FileAudio size={16} />
                      <span>Choose File</span>
                    </button>
                  </div>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <p className="text-xs text-gray-400 mt-2">
                  Supported formats: MP3, WAV, OGG, M4A, FLAC
                </p>
              </div>
            </div>
          )}

          {/* Audio Setup Section */}
          {showSettings && (
            <div className="border-t border-gray-700 pt-4 mt-4">
              <h3 className="text-lg font-semibold mb-4">Audio Setup & Testing</h3>
              
              {/* Calibration */}
              <div className="bg-gray-700 rounded-lg p-4 mb-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <Gauge size={20} className="text-blue-400" />
                    <h4 className="font-medium">Audio Calibration</h4>
                  </div>
                  <button
                    onClick={handleCalibration}
                    disabled={isCalibrating || !sessionId}
                    className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isCalibrating ? (
                      <>
                        <Loader className="animate-spin" size={16} />
                        <span>Calibrating...</span>
                      </>
                    ) : (
                      <>
                        <Gauge size={16} />
                        <span>Calibrate (3s)</span>
                      </>
                    )}
                  </button>
                </div>
                
                {calibrationData && (
                  <div className="mt-3 space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Noise Level:</span>
                      <span className="font-medium">{calibrationData.noise_level?.toFixed(2)} dB</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Recommended Gain:</span>
                      <span className="font-medium">{calibrationData.recommended_gain?.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Status:</span>
                      <span className="font-medium text-green-400">{calibrationData.calibration_status}</span>
                    </div>
                  </div>
                )}
              </div>

              {/* Test Recording */}
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <TestTube size={20} className="text-purple-400" />
                    <h4 className="font-medium">Test Recording</h4>
                  </div>
                  <button
                    onClick={handleTestRecording}
                    disabled={isTesting || !sessionId}
                    className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    {isTesting ? (
                      <>
                        <Loader className="animate-spin" size={16} />
                        <span>Testing...</span>
                      </>
                    ) : (
                      <>
                        <TestTube size={16} />
                        <span>Test Record (5s)</span>
                      </>
                    )}
                  </button>
                </div>
                
                {testRecordingData && (
                  <div className="mt-3 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-300">Quality Score:</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-32 bg-gray-600 rounded-full h-2">
                          <div 
                            className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${testRecordingData.quality_score * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-medium">{(testRecordingData.quality_score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                    
                    {testRecordingData.issues && testRecordingData.issues.length > 0 && (
                      <div>
                        <p className="text-sm text-yellow-400 mb-1">Issues Detected:</p>
                        <ul className="text-xs text-gray-300 space-y-1">
                          {testRecordingData.issues.map((issue, idx) => (
                            <li key={idx}>• {issue}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {testRecordingData.recommendations && testRecordingData.recommendations.length > 0 && (
                      <div>
                        <p className="text-sm text-blue-400 mb-1">Recommendations:</p>
                        <ul className="text-xs text-gray-300 space-y-1">
                          {testRecordingData.recommendations.map((rec, idx) => (
                            <li key={idx}>• {rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Device Selection */}
          <div className="grid md:grid-cols-2 gap-4 mt-4">
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
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${micEnabled
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
              className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${isRecording
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
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-lg font-semibold">AI Response</h3>
                <div className="flex items-center space-x-2">
                  <select
                    value={simplificationLevel}
                    onChange={(e) => setSimplificationLevel(e.target.value)}
                    className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-1 text-sm focus:outline-none focus:border-purple-500"
                  >
                    <option value="basic">Basic</option>
                    <option value="intermediate">Intermediate</option>
                    <option value="advanced">Advanced</option>
                  </select>
                  <button
                    onClick={handleSimplifyResponse}
                    disabled={loading}
                    className="flex items-center space-x-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded-lg text-sm transition-colors disabled:opacity-50"
                  >
                    {loading ? (
                      <Loader className="animate-spin" size={14} />
                    ) : (
                      <Volume2 size={14} />
                    )}
                    <span>Simplify</span>
                  </button>
                </div>
              </div>
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