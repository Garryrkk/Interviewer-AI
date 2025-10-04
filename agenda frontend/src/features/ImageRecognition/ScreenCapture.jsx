import React, { useState, useEffect } from 'react';
import { Play, Pause, Square, Eye, CheckCircle, XCircle, AlertTriangle, Monitor, Circle, Activity, Wifi, WifiOff, Camera, Zap } from 'lucide-react';

const ScreenRecordingApp = () => {
  const [userId, setUserId] = useState('user_001');
  const [recordingStatus, setRecordingStatus] = useState('not_recording');
  const [permissions, setPermissions] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recordingStats, setRecordingStats] = useState({ duration: 0, frame_count: 0 });
  const [backendStatus, setBackendStatus] = useState('disconnected');

  // API Base URL - adjust based on your backend
  const API_BASE = 'http://localhost:8000';

  // API Helper Function
  const apiCall = async (endpoint, method = 'GET', body = null) => {
    try {
      setError(null);
      const config = {
        method,
        headers: {
          'Content-Type': 'application/json',
        },
      };
      
      if (body) {
        config.body = JSON.stringify(body);
      }

      const response = await fetch(`${API_BASE}${endpoint}`, config);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'API request failed');
      }
      
      setBackendStatus('connected');
      return data;
    } catch (err) {
      setError(err.message);
      setBackendStatus('disconnected');
      throw err;
    }
  };

  // Check Permissions
  const checkPermissions = async () => {
    setLoading(true);
    setBackendStatus('connecting');
    try {
      const result = await apiCall('/screen_recording/permissions', 'POST', { user_id: userId });
      setPermissions(result);
    } catch (err) {
      console.error('Permission check failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Start Recording
  const startRecording = async () => {
    if (!permissions?.permissions_granted) {
      setError('Please check permissions first');
      return;
    }
    
    setLoading(true);
    try {
      const result = await apiCall('/screen_recording/start', 'POST', { 
        user_id: userId,
        permissions_granted: permissions.permissions_granted 
      });
      setRecordingStatus(result.status);
      if (result.status === 'recording_started') {
        pollRecordingStatus();
      }
    } catch (err) {
      console.error('Start recording failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Pause Recording
  const pauseRecording = async () => {
    setLoading(true);
    try {
      const result = await apiCall('/screen_recording/pause', 'POST', { user_id: userId });
      setRecordingStatus(result.status);
    } catch (err) {
      console.error('Pause recording failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Resume Recording
  const resumeRecording = async () => {
    setLoading(true);
    try {
      const result = await apiCall('/screen_recording/resume', 'POST', { user_id: userId });
      setRecordingStatus(result.status);
    } catch (err) {
      console.error('Resume recording failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Stop Recording
  const stopRecording = async () => {
    setLoading(true);
    try {
      const result = await apiCall('/screen_recording/stop', 'POST', { user_id: userId });
      setRecordingStatus(result.status);
      setRecordingStats({ 
        duration: result.duration || 0, 
        frame_count: result.frame_count || 0 
      });
    } catch (err) {
      console.error('Stop recording failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Get Recording Status
  const getRecordingStatus = async () => {
    try {
      const result = await apiCall(`/screen_recording/status/${userId}`);
      setRecordingStatus(result.status);
      setRecordingStats({
        duration: result.duration || 0,
        frame_count: result.frame_count || 0
      });
    } catch (err) {
      console.error('Status check failed:', err);
    }
  };

  // Poll Recording Status
  const pollRecordingStatus = () => {
    const interval = setInterval(() => {
      if (recordingStatus === 'recording_started' || recordingStatus === 'recording_paused' || recordingStatus === 'recording_resumed') {
        getRecordingStatus();
      } else {
        clearInterval(interval);
      }
    }, 2000);
    
    // Cleanup interval after 5 minutes
    setTimeout(() => clearInterval(interval), 300000);
  };

  // Analyze Recording
  const analyzeRecording = async () => {
    if (recordingStatus === 'not_recording') {
      setError('No recording available to analyze');
      return;
    }
    
    setLoading(true);
    try {
      const result = await apiCall('/screen_recording/analyze', 'POST', { 
        user_id: userId,
        question: question.trim() 
      });
      setAnalysisResult(result);
    } catch (err) {
      console.error('Analysis failed:', err);
    } finally {
      setLoading(false);
    }
  };

  // Status color helper
  const getStatusColor = (status) => {
    switch (status) {
      case 'recording_started':
      case 'recording_resumed':
        return 'text-red-400';
      case 'recording_paused':
        return 'text-yellow-400';
      case 'recording_stopped':
        return 'text-slate-400';
      default:
        return 'text-slate-500';
    }
  };

  // Get recording button state
  const isRecording = recordingStatus === 'recording_started' || recordingStatus === 'recording_resumed';
  const isPaused = recordingStatus === 'recording_paused';
  const isActive = isRecording || isPaused;

  useEffect(() => {
    checkPermissions();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-slate-100 mb-2 flex items-center space-x-3">
              <Monitor className="text-blue-500" size={40} />
              <span>Screen Recording</span>
            </h1>
            <p className="text-slate-400">Real-time screen recording and AI analysis</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium ${
              isActive ? 'bg-green-600 text-white' : 'bg-slate-700 text-slate-300'
            }`}>
              <Circle size={8} className={`fill-current ${isActive ? 'text-green-300' : 'text-slate-500'}`} />
              <span>{isActive ? 'Recording' : 'Standby'}</span>
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
              <span className="text-xs text-slate-400">
                User: {userId}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`flex items-center space-x-1 text-xs ${
                permissions?.permissions_granted === true ? 'text-green-400' : 
                permissions?.permissions_granted === false ? 'text-red-400' : 'text-slate-400'
              }`}>
                {permissions?.permissions_granted === true ? <Circle size={6} className="fill-current" /> :
                 permissions?.permissions_granted === false ? <AlertTriangle size={12} /> :
                 <Circle size={6} className="text-slate-500" />}
                <span>Permissions</span>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-600/20 border border-red-600/30 text-red-100 p-4 rounded-xl">
            <div className="flex items-center space-x-2">
              <XCircle size={20} className="text-red-400" />
              <span className="font-medium">Error:</span>
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Control Panel */}
        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-xl font-semibold mb-4 text-slate-200">Control Panel</h3>
          <div className="flex flex-wrap gap-4">
            <button 
              onClick={checkPermissions}
              disabled={loading}
              className="flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all bg-purple-600 hover:bg-purple-700 text-white disabled:opacity-50"
            >
              <Eye size={20} />
              <span>Check Permissions</span>
              {loading && <Activity size={16} className="animate-spin" />}
            </button>

            {!isRecording && !isPaused && (
              <button 
                onClick={startRecording}
                disabled={loading || !permissions?.permissions_granted}
                className="flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all bg-green-600 hover:bg-green-700 text-white disabled:opacity-50"
              >
                <Play size={20} />
                <span>Start Recording</span>
              </button>
            )}

            {isRecording && (
              <button 
                onClick={pauseRecording}
                disabled={loading}
                className="flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all bg-yellow-600 hover:bg-yellow-700 text-white disabled:opacity-50"
              >
                <Pause size={20} />
                <span>Pause Recording</span>
              </button>
            )}

            {isPaused && (
              <button 
                onClick={resumeRecording}
                disabled={loading}
                className="flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50"
              >
                <Play size={20} />
                <span>Resume Recording</span>
              </button>
            )}

            {isActive && (
              <button 
                onClick={stopRecording}
                disabled={loading}
                className="flex items-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all bg-red-600 hover:bg-red-700 text-white disabled:opacity-50"
              >
                <Square size={20} />
                <span>Stop Recording</span>
              </button>
            )}
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Left Column - Recording Status and Stats */}
          <div className="xl:col-span-2 space-y-6">
            {/* Recording Status */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                <Camera className="mr-2" size={20} />
                Recording Status
              </h3>
              <div className="bg-slate-900/80 rounded-xl p-6 border-2 border-slate-700">
                <div className="text-center space-y-4">
                  <div className={`text-3xl font-bold ${getStatusColor(recordingStatus)}`}>
                    {recordingStatus.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-center">
                    <div className="bg-slate-800/50 p-4 rounded-lg">
                      <div className="text-2xl font-bold text-blue-400">{recordingStats.duration}s</div>
                      <div className="text-sm text-slate-400">Duration</div>
                    </div>
                    <div className="bg-slate-800/50 p-4 rounded-lg">
                      <div className="text-2xl font-bold text-green-400">{recordingStats.frame_count}</div>
                      <div className="text-sm text-slate-400">Frames</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Permissions Status */}
            {permissions && (
              <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                  <Eye className="mr-2" size={20} />
                  Permissions Status
                </h3>
                <div className={`p-4 rounded-lg border-2 ${
                  permissions.permissions_granted 
                    ? 'bg-green-600/20 border-green-600/30' 
                    : 'bg-red-600/20 border-red-600/30'
                }`}>
                  <div className="flex items-center space-x-3">
                    {permissions.permissions_granted ? (
                      <CheckCircle className="text-green-400" size={24} />
                    ) : (
                      <XCircle className="text-red-400" size={24} />
                    )}
                    <div>
                      <div className={`font-medium ${permissions.permissions_granted ? 'text-green-100' : 'text-red-100'}`}>
                        {permissions.permissions_granted ? 'Permissions Granted' : 'Permissions Required'}
                      </div>
                      <div className={`text-sm ${permissions.permissions_granted ? 'text-green-200' : 'text-red-200'}`}>
                        {permissions.message}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Analysis */}
          <div className="space-y-6">
            {/* AI Analysis */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                <Zap className="mr-2" size={20} />
                AI Analysis
              </h3>
              <div className="space-y-4">
                <div>
                  <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a question about the recording..."
                    className="w-full p-3 bg-slate-900/80 border border-slate-600 rounded-lg text-slate-200 placeholder-slate-400 resize-none focus:outline-none focus:border-blue-500"
                    rows={3}
                  />
                </div>
                <button 
                  onClick={analyzeRecording}
                  disabled={loading || !question.trim() || recordingStatus === 'not_recording'}
                  className="w-full flex items-center justify-center space-x-3 py-3 px-6 rounded-lg font-medium transition-all bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50"
                >
                  <Activity size={20} />
                  <span>{loading ? 'Analyzing...' : 'Analyze Recording'}</span>
                  {loading && <Activity size={16} className="animate-spin" />}
                </button>
              </div>
            </div>

            {/* Analysis Results */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                <Activity className="mr-2" size={20} />
                Analysis Results
              </h3>
              <div className="bg-slate-900/80 rounded-lg p-4 max-h-64 overflow-y-auto">
                {!analysisResult ? (
                  <div className="text-center text-slate-400 py-8">
                    <div className="text-center">
                      <Circle size={32} className="mx-auto mb-2 text-slate-500" />
                      <p>No analysis results yet</p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="p-3 rounded-lg bg-blue-600/20 border border-blue-600/30 text-blue-100">
                      <div className="text-xs font-medium text-blue-200 uppercase tracking-wide mb-1">
                        Analysis Result
                      </div>
                      <div className="text-sm">
                        {typeof analysisResult === 'string' ? analysisResult : JSON.stringify(analysisResult, null, 2)}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScreenRecordingApp;