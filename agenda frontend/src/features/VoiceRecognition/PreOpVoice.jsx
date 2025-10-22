import React, { useState, useEffect, useRef } from 'react';
import { AudioService } from './VoicePreoputils';
<<<<<<< HEAD
import { PreOpVoice } from '../../services/voiceService';
=======
import { API_BASE_URL, callEndpoint } from "../../services/apiConfig";
// api-configuration.js
const ENV = "development"; // or "production"

const BASE_URLS = {
  development: "http://127.0.0.1:8000",
  production: "https://api.myapp.com",
};

export const API_BASE_URL = BASE_URLS[ENV];



>>>>>>> dc48a3c9bb6b60e86081e19b5a58e753bc2d8ceb
// For demo purposes, we'll include the AudioService inline



class AudioService {
  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    this.apiPrefix = '/api/v1/audio';
    this.defaultTimeout = 10000;
    this.transcriptionTimeout = 30000;
  }

  async makeRequest(endpoint, options = {}) {
    const url = `${this.baseURL}${this.apiPrefix}${endpoint}`;
    const timeout = options.timeout || this.defaultTimeout;

    const defaultOptions = {
      headers: {
        'Accept': 'application/json',
        ...options.headers
      },
      ...options
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...defaultOptions,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }

      return await response.text();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        throw new Error('Request timed out');
      }

      if (error.message.includes('Failed to fetch')) {
        throw new Error('Unable to connect to audio service. Please check if the server is running.');
      }

      throw error;
    }
  }

  async calibrateAudio({ duration = 3, sampleRate = 16000, channels = 1 }) {
    const response = await this.makeRequest('/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        duration,
        sample_rate: sampleRate,
        channels
      }),
      timeout: (duration + 5) * 1000
    });
    return response;
  }

  async testRecording({ duration = 5, sampleRate = 16000, channels = 1, applyCalibration = true }) {
    const response = await this.makeRequest('/test-record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        duration,
        sample_rate: sampleRate,
        channels,
        apply_calibration: applyCalibration
      }),
      timeout: (duration + 10) * 1000
    });
    return response;
  }

  async transcribeAudio(audioFile, language = 'auto', modelSize = 'base') {
    if (!audioFile.type.startsWith('audio/')) {
      throw new Error('Please select a valid audio file');
    }

    const maxSize = 50 * 1024 * 1024;
    if (audioFile.size > maxSize) {
      throw new Error('Audio file is too large. Maximum size is 50MB.');
    }

    const formData = new FormData();
    formData.append('audio_file', audioFile);

    const queryParams = new URLSearchParams({
      language,
      model_size: modelSize
    });

    const response = await this.makeRequest(`/transcribe?${queryParams}`, {
      method: 'POST',
      body: formData,
      timeout: this.transcriptionTimeout,
      headers: {}
    });
    return response;
  }

  async transcribeTestRecording() {
    const response = await this.makeRequest('/transcribe-test', {
      method: 'POST',
      timeout: this.transcriptionTimeout
    });
    return response;
  }

  async getCalibrationStatus() {
    const response = await this.makeRequest('/calibration-status', {
      method: 'GET'
    });
    return response;
  }

  async resetCalibration() {
    const response = await this.makeRequest('/reset-calibration', {
      method: 'DELETE'
    });
    return response;
  }
}

const AudioTranscriptionApp = () => {
  const [activeTab, setActiveTab] = useState('transcribe');
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [calibrationStatus, setCalibrationStatus] = useState(null);
  const [transcriptionResult, setTranscriptionResult] = useState(null);
  const [testRecordingResult, setTestRecordingResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [supportedFormats, setSupportedFormats] = useState([]);
  const [error, setError] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [recordingTimer, setRecordingTimer] = useState(0);
  const [calibrationSettings, setCalibrationSettings] = useState({
    duration: 3,
    sampleRate: 16000,
    channels: 1
  });
  const [testSettings, setTestSettings] = useState({
    duration: 5,
    sampleRate: 16000,
    channels: 1,
    applyCalibration: true
  });
  const [transcriptionSettings, setTranscriptionSettings] = useState({
    language: 'auto',
    modelSize: 'base'
  });

  const fileInputRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const audioService = new AudioService();

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch supported formats from backend
        const formats = await api.current.getSupportedFormats();
        if (formats && formats.supported_formats) {
          setSupportedFormats(formats.supported_formats);
        }

        // Fetch calibration status from backend
        const status = await api.current.getCalibrationStatus();
        if (status) {
          setCalibrationStatus(status);
        }
      } catch (err) {
        console.error("Failed to fetch initial data:", err);
        setError("Failed to connect to backend for supported formats or calibration status");
      }
    };

    fetchData();

    // Cleanup timer on unmount
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, []);


  const checkCalibrationStatus = async () => {
    try {
      const status = await audioService.getCalibrationStatus();
      setCalibrationStatus(status);
      setIsCalibrated(status.is_calibrated);
    } catch (err) {
      console.error('Failed to check calibration status:', err);
    }
  };

  const startRecordingTimer = (duration) => {
    setRecordingTimer(duration);
    recordingTimerRef.current = setInterval(() => {
      setRecordingTimer(prev => {
        if (prev <= 1) {
          clearInterval(recordingTimerRef.current);
          setIsRecording(false);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleCalibrate = async () => {
    setLoading(true);
    setError(null);
    setIsRecording(true);

    try {
      startRecordingTimer(calibrationSettings.duration);
      const result = await api.current.calibrateAudio(calibrationSettings);
      setCalibrationStatus(result);
      setIsCalibrated(true);
    } catch (err) {
      setError(`Calibration failed: ${err.message}`);
      setIsRecording(false);
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleTestRecording = async () => {
    setLoading(true);
    setError(null);
    setIsRecording(true);

    try {
      startRecordingTimer(testSettings.duration);
      const result = await audioService.testRecording(testSettings);
      setTestRecordingResult(result);
    } catch (err) {
      setError(`Test recording failed: ${err.message}`);
      setIsRecording(false);
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
      setError(null);
    }
  };

  const handleTranscribeFile = async () => {
    if (!audioFile) {
      setError('Please select an audio file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await audioService.transcribeAudio(
        audioFile,
        transcriptionSettings.language,
        transcriptionSettings.modelSize
      );
      setTranscriptionResult(result);
    } catch (err) {
      setError(`Transcription failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleTranscribeTest = async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await audioService.transcribeTestRecording();
      setTranscriptionResult(result);
    } catch (err) {
      setError(`Test transcription failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleResetCalibration = async () => {
    try {
      await audioService.resetCalibration();
      setIsCalibrated(false);
      setCalibrationStatus(null);
    } catch (err) {
      setError(`Reset failed: ${err.message}`);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen p-6" style={{ backgroundColor: '#1E1E2F', color: '#F8FAFC', fontFamily: 'Roboto, sans-serif' }}>
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Audio Transcription Service</h1>
          <p className="text-gray-300">Professional speech-to-text conversion with audio calibration</p>
        </div>

        {/* Status Bar */}
        <div className="mb-6 p-4 rounded-lg bg-gray-800">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isCalibrated ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm">
                  {isCalibrated ? 'Calibrated' : 'Not Calibrated'}
                </span>
              </div>
              {calibrationStatus && (
                <span className="text-sm text-gray-300">
                  Noise Level: {calibrationStatus.noise_level?.toFixed(1)}dB
                </span>
              )}
            </div>
            {isRecording && (
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-mono">Recording: {recordingTimer}s</span>
              </div>
            )}
          </div>
        </div>

        {/* Progress Indicator */}
        {loading && (
          <div className="mb-6 p-4 bg-gray-800 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-500"></div>
              <div className="flex-1">
                <div className="text-sm font-medium text-gray-200">
                  {isRecording ? 'Recording in progress...' : 'Processing...'}
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full transition-all duration-300 animate-pulse"
                    style={{ width: '45%' }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
        {error && (
          <div className="mb-6 p-4 bg-red-900 border border-red-700 rounded-lg">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="flex space-x-1 mb-6">
          {[
            { id: 'transcribe', label: 'Transcribe Audio' },
            { id: 'calibrate', label: 'Audio Calibration' },
            { id: 'test', label: 'Test Recording' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${activeTab === tab.id
                ? 'text-white shadow-lg'
                : 'text-gray-300 hover:text-white hover:bg-gray-700'
                }`}
              style={{
                backgroundColor: activeTab === tab.id ? '#8F74D4' : 'transparent'
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {activeTab === 'transcribe' && (
              <div className="bg-gray-800 p-6 rounded-lg">
                <h2 className="text-xl font-semibold mb-4">Audio Transcription</h2>

                <div className="space-y-4">
                  {/* File Upload */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Upload Audio File</label>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="audio/*"
                      onChange={handleFileUpload}
                      className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-purple-600 file:text-white hover:file:bg-purple-700 file:cursor-pointer"
                    />
                    {audioFile && (
                      <p className="text-sm text-gray-300 mt-1">Selected: {audioFile.name}</p>
                    )}
                  </div>

                  {/* Transcription Settings */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Language</label>
                      <select
                        value={transcriptionSettings.language}
                        onChange={(e) => setTranscriptionSettings(prev => ({ ...prev, language: e.target.value }))}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
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
                      <label className="block text-sm font-medium mb-1">Model Size</label>
                      <select
                        value={transcriptionSettings.modelSize}
                        onChange={(e) => setTranscriptionSettings(prev => ({ ...prev, modelSize: e.target.value }))}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      >
                        <option value="tiny">Tiny (Fast)</option>
                        <option value="base">Base</option>
                        <option value="small">Small</option>
                        <option value="medium">Medium</option>
                        <option value="large">Large (Best Quality)</option>
                      </select>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex space-x-3">
                    <button
                      onClick={handleTranscribeFile}
                      disabled={loading || !audioFile}
                      className="flex-1 py-3 px-6 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg"
                      style={{ backgroundColor: '#8F74D4' }}
                    >
                      {loading ? 'Transcribing...' : 'Transcribe File'}
                    </button>
                    {testRecordingResult && (
                      <button
                        onClick={handleTranscribeTest}
                        disabled={loading}
                        className="flex-1 py-3 px-6 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-all duration-200 disabled:opacity-50"
                      >
                        Transcribe Test Recording
                      </button>
                    )}
                  </div>

                  {/* Quick Actions */}
                  <div className="mt-4 p-3 bg-gray-700 rounded-lg">
                    <h4 className="text-sm font-medium mb-2 text-gray-300">Quick Actions</h4>
                    <div className="flex flex-wrap gap-2 text-xs">
                      <button
                        onClick={async () => {
                          try {
                            const formats = await audioService.getSupportedFormats();
                            alert(`Supported formats: ${formats.supported_formats.join(', ')}`);
                          } catch (err) {
                            console.error('Failed to get formats:', err);
                          }
                        }}
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white"
                      >
                        View Supported Formats
                      </button>
                      {transcriptionResult && (
                        <>
                          <button
                            onClick={() => {
                              const success = navigator.clipboard?.writeText(transcriptionResult.text);
                              if (success) {
                                alert('Text copied to clipboard!');
                              }
                            }}
                            className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-white"
                          >
                            Copy Text
                          </button>
                          <button
                            onClick={() => {
                              const element = document.createElement('a');
                              const file = new Blob([transcriptionResult.text], { type: 'text/plain' });
                              element.href = URL.createObjectURL(file);
                              element.download = 'transcription.txt';
                              document.body.appendChild(element);
                              element.click();
                              document.body.removeChild(element);
                            }}
                            className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-white"
                          >
                            Download TXT
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'calibrate' && (
              <div className="bg-gray-800 p-6 rounded-lg">
                <h2 className="text-xl font-semibold mb-4">Audio Calibration</h2>

                <div className="space-y-4">
                  <p className="text-gray-300 text-sm">
                    Calibration measures background noise to optimize voice detection.
                    Stay quiet during calibration for best results.
                  </p>

                  {/* Calibration Settings */}
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Duration (s)</label>
                      <input
                        type="number"
                        min="1"
                        max="10"
                        value={calibrationSettings.duration}
                        onChange={(e) => setCalibrationSettings(prev => ({ ...prev, duration: parseInt(e.target.value) }))}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Sample Rate</label>
                      <select
                        value={calibrationSettings.sampleRate}
                        onChange={(e) => setCalibrationSettings(prev => ({ ...prev, sampleRate: parseInt(e.target.value) }))}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      >
                        <option value="8000">8000 Hz</option>
                        <option value="16000">16000 Hz</option>
                        <option value="22050">22050 Hz</option>
                        <option value="44100">44100 Hz</option>
                        <option value="48000">48000 Hz</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Channels</label>
                      <select
                        value={calibrationSettings.channels}
                        onChange={(e) => setCalibrationSettings(prev => ({ ...prev, channels: parseInt(e.target.value) }))}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      >
                        <option value="1">Mono</option>
                        <option value="2">Stereo</option>
                      </select>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex space-x-3">
                    <button
                      onClick={handleCalibrate}
                      disabled={loading || isRecording}
                      className="flex-1 py-3 px-6 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg"
                      style={{ backgroundColor: '#8F74D4' }}
                    >
                      {isRecording ? 'Calibrating...' : 'Start Calibration'}
                    </button>
                    {isCalibrated && (
                      <button
                        onClick={handleResetCalibration}
                        className="px-6 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-all duration-200"
                      >
                        Reset
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'test' && (
              <div className="bg-gray-800 p-6 rounded-lg">
                <h2 className="text-xl font-semibold mb-4">Test Recording</h2>

                <div className="space-y-4">
                  <p className="text-gray-300 text-sm">
                    Record a test clip to verify audio quality before transcription.
                  </p>

                  {/* Test Settings */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Duration (s)</label>
                      <input
                        type="number"
                        min="1"
                        max="30"
                        value={testSettings.duration}
                        onChange={(e) => setTestSettings(prev => ({ ...prev, duration: parseInt(e.target.value) }))}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1">Apply Calibration</label>
                      <select
                        value={testSettings.applyCalibration ? 'true' : 'false'}
                        onChange={(e) => setTestSettings(prev => ({ ...prev, applyCalibration: e.target.value === 'true' }))}
                        className="w-full p-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                      >
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                      </select>
                    </div>
                  </div>

                  <button
                    onClick={handleTestRecording}
                    disabled={loading || isRecording}
                    className="w-full py-3 px-6 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg"
                    style={{ backgroundColor: '#8F74D4' }}
                  >
                    {isRecording ? 'Recording...' : 'Start Test Recording'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Calibration Results */}
            {calibrationStatus && (
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="font-semibold mb-3">Calibration Status</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Status:</span>
                    <span className="text-green-400">{calibrationStatus.status}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Noise Level:</span>
                    <span>{calibrationStatus.noise_level?.toFixed(1)}dB</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Quality Score:</span>
                    <span>{(calibrationStatus.quality_score * 100)?.toFixed(0)}%</span>
                  </div>
                  {calibrationStatus.recommendations && (
                    <div className="mt-3">
                      <p className="font-medium text-yellow-400">Recommendations:</p>
                      <ul className="list-disc list-inside text-xs text-gray-300 mt-1">
                        {calibrationStatus.recommendations.map((rec, idx) => (
                          <li key={idx}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Test Recording Results */}
            {testRecordingResult && (
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="font-semibold mb-3">Test Recording Results</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Duration:</span>
                    <span>{testRecordingResult.duration?.toFixed(1)}s</span>
                  </div>
                  <div className="flex justify-between">
                    <span>File Size:</span>
                    <span>{(testRecordingResult.file_size / 1024)?.toFixed(1)}KB</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Peak Level:</span>
                    <span>{(testRecordingResult.peak_amplitude * 100)?.toFixed(0)}%</span>
                  </div>
                  {testRecordingResult.signal_to_noise_ratio && (
                    <div className="flex justify-between">
                      <span>SNR:</span>
                      <span>{testRecordingResult.signal_to_noise_ratio?.toFixed(1)}dB</span>
                    </div>
                  )}
                  {testRecordingResult.recommendations && (
                    <div className="mt-3">
                      <p className="font-medium text-yellow-400">Analysis:</p>
                      <ul className="list-disc list-inside text-xs text-gray-300 mt-1">
                        {testRecordingResult.recommendations.map((rec, idx) => (
                          <li key={idx}>{rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Transcription Results */}
            {transcriptionResult && (
              <div className="bg-gray-800 p-4 rounded-lg">
                <h3 className="font-semibold mb-3">Transcription Results</h3>
                <div className="space-y-3">
                  <div className="bg-gray-900 p-3 rounded text-sm">
                    <p className="font-medium text-green-400 mb-2">Transcribed Text:</p>
                    <p className="text-gray-200">{transcriptionResult.text}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-xs">
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span>Language:</span>
                        <span>{transcriptionResult.language}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Confidence:</span>
                        <span>{(transcriptionResult.confidence * 100).toFixed(0)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Duration:</span>
                        <span>{transcriptionResult.duration?.toFixed(1)}s</span>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between">
                        <span>Word Count:</span>
                        <span>{transcriptionResult.word_count}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Processing:</span>
                        <span>{transcriptionResult.processing_time?.toFixed(1)}s</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Quality:</span>
                        <span>{(transcriptionResult.audio_quality_score * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>

                  {transcriptionResult.segments && transcriptionResult.segments.length > 0 && (
                    <div className="mt-3">
                      <p className="font-medium text-blue-400 mb-2">Segments:</p>
                      <div className="max-h-32 overflow-y-auto space-y-1 text-xs">
                        {transcriptionResult.segments.map((segment, idx) => (
                          <div key={idx} className="bg-gray-900 p-2 rounded">
                            <div className="text-gray-400">
                              {formatTime(Math.floor(segment.start))} - {formatTime(Math.floor(segment.end))}
                            </div>
                            <div className="text-gray-200">{segment.text}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AudioTranscriptionApp;