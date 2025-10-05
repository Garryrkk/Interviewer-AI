import React, { useState, useEffect } from 'react';
import { Play, Square, Eye, EyeOff, Settings, Shield, Activity, Download, Trash2, RefreshCw } from 'lucide-react';
import { InvisibilityService, SessionManager, ConfigurationValidator, UIStateManager } from './invisibilityService.js';
import { HiddenAnswers } from '../../services/mainFeature.js';
import { pollHiddenSuggestions } from '../../services/mainFeature.js';
import {InvisibilityService} from './invisibilityService.js';

invisibility.enableInvisibilityMode({
  recording: { screenRecording: true, voiceRecording: true },
  ui: { theme: 'dark' },
  security: { encrypt: true },
  sessionName: 'MySession',
  metadata: { project: 'Test' }
})
.then(response => {
  console.log("Invisibility Mode Enabled:", response);
})
.catch(err => {
  console.error("Enable Error:", err);
});

// Start recording
invisibility.startRecording({ realTimeInsights: true })
.then(response => {
  console.log("Recording Started:", response);
})
.catch(err => {
  console.error("Start Recording Error:", err);
});

// Stop recording
invisibility.stopRecording()
.then(response => {
  console.log("Recording Stopped:", response);
})
.catch(err => {
  console.error("Stop Recording Error:", err);
});

// Disable invisibility mode
invisibility.disableInvisibilityMode()
.then(response => {
  console.log("Invisibility Mode Disabled:", response);
})
.catch(err => {
  console.error("Disable Error:", err);
});

HiddenAnswers({ question: "What's the AI insight?", context: "meeting-notes" })
  .then(response => {
    console.log("HiddenAnswers API Response:", response);
  })
  .catch(err => {
    console.error("HiddenAnswers API Error:", err);
  });

// 🔹 Example 2 — calling pollHiddenSuggestions
pollHiddenSuggestions("abc123")
  .then(response => {
    console.log("pollHiddenSuggestions API Response:", response);
  })
  .catch(err => {
    console.error("pollHiddenSuggestions API Error:", err);
  });

// Legacy API wrapper for backward compatibility
class InvisibilityAPI {
  constructor() {}

  async enableMode(config) {
    const response = await fetch(`${this.baseURL}/mode/enable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return response.json();
  }

  async disableMode(sessionId) {
    const response = await fetch(`${this.baseURL}/mode/disable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    return response.json();
  }

  async startRecording(config) {
    const response = await fetch(`${this.baseURL}/recording/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return response.json();
  }

  async stopRecording(sessionId) {
    const response = await fetch(`${this.baseURL}/recording/stop`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
    return response.json();
  }

  async hideUI(sessionId, components, hideMode) {
    const response = await fetch(`${this.baseURL}/ui/hide`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        components_to_hide: components,
        hide_mode: hideMode
      })
    });
    return response.json();
  }

  async showUI(sessionId, components) {
    const response = await fetch(`${this.baseURL}/ui/show`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        components_to_show: components
      })
    });
    return response.json();
  }

  async getSessionStatus(sessionId) {
    const response = await fetch(`${this.baseURL}/session/${sessionId}/status`);
    return response.json();
  }

  async generateInsights(sessionId, insightTypes) {
    const response = await fetch(`${this.baseURL}/insights/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        insight_types: insightTypes,
        processing_options: {}
      })
    });
    return response.json();
  }

  async getInsights(sessionId) {
    const response = await fetch(`${this.baseURL}/insights/${sessionId}`);
    return response.json();
  }

  async getSecurityStatus(sessionId) {
    const response = await fetch(`${this.baseURL}/security/status/${sessionId}`);
    return response.json();
  }

  async cleanupSession(sessionId) {
    const response = await fetch(`${this.baseURL}/session/${sessionId}`, {
      method: 'DELETE'
    });
    return response.json();
  }

  async healthCheck() {
    const response = await fetch(`${this.baseURL}/health`);
    return response.json();
  }
}

const InvisibilityDashboard = () => {
  // Initialize services
  const [invisibilityService] = useState(() => new InvisibilityService());
  const [sessionManager] = useState(() => new SessionManager());
  const [configValidator] = useState(() => new ConfigurationValidator());
  const [uiStateManager] = useState(() => new UIStateManager());
  
  // State management
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [invisibilityEnabled, setInvisibilityEnabled] = useState(false);
  const [recording, setRecording] = useState(false);
  const [uiHidden, setUiHidden] = useState(false);
  const [sessionStatus, setSessionStatus] = useState(null);
  const [securityStatus, setSecurityStatus] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [systemConfig, setSystemConfig] = useState(null);
  const [hideModes, setHideModes] = useState([]);
  const [errors, setErrors] = useState([]);

  // Configuration states
  const [recordingConfig, setRecordingConfig] = useState({
    screen_recording: true,
    voice_recording: true,
    auto_notes: true,
    real_time_insights: false,
    recording_quality: 'medium',
    audio_format: 'mp3',
    video_format: 'mp4'
  });

  const [uiConfig, setUiConfig] = useState({
    hide_mode: 'minimize',
    components_to_hide: ['recording_indicator', 'ai_insights_panel'],
    keep_separate_window: false,
    minimize_to_tray: true,
    show_discrete_indicator: false
  });

  const [securityConfig, setSecurityConfig] = useState({
    local_processing_only: true,
    encrypt_data: true,
    auto_delete_after: 24,
    no_cloud_upload: true,
    secure_storage_path: null
  });

  // Event listeners setup
  useEffect(() => {
    // Set up service event listeners
    const handleSessionCreated = (data) => {
      setCurrentSessionId(data.sessionId);
      setInvisibilityEnabled(true);
    };

    const handleSessionEnded = () => {
      setCurrentSessionId(null);
      setInvisibilityEnabled(false);
      setRecording(false);
      setUiHidden(false);
      setSessionStatus(null);
    };

    const handleRecordingStarted = () => {
      setRecording(true);
    };

    const handleRecordingstopped = (data) => {
      setRecording(false);
      setSessionStatus(prev => ({
        ...prev,
        recording_duration: data.duration,
        data_size: data.dataSize
      }));
    };

    const handleUIHidden = () => {
      setUiHidden(true);
    };

    const handleUIShown = () => {
      setUiHidden(false);
    };

    const handleError = (data) => {
      setError(`${data.action}: ${data.error}`);
      setTimeout(() => setError(null), 5000);
    };

    const handleInsightsGenerated = (data) => {
      setInsights(data.insights);
    };

    // Register event listeners
    invisibilityService.on('session:created', handleSessionCreated);
    invisibilityService.on('session:ended', handleSessionEnded);
    invisibilityService.on('recording:started', handleRecordingStarted);
    invisibilityService.on('recording:stopped', handleRecordingStopped);
    invisibilityService.on('ui:hidden', handleUIHidden);
    invisibilityService.on('ui:shown', handleUIShown);
    invisibilityService.on('error', handleError);
    invisibilityService.on('insights:generated', handleInsightsGenerated);

    // Cleanup listeners on unmount
    return () => {
      invisibilityService.removeAllListeners();
    };
  }, [invisibilityService]);

  // Initialize available hide modes and system config
  useEffect(() => {
    const initializeData = async () => {
      try {
        // Get available hide modes
        const modes = await invisibilityService.getHideModes();
        setHideModes(modes);
        
        // Get system configuration
        const sysConfig = await invisibilityService.getSystemConfig();
        setSystemConfig(sysConfig);
      } catch (err) {
        console.error('Failed to initialize data:', err);
      }
    };

    initializeData();
  }, [invisibilityService]);

  // Session monitoring with additional data
  useEffect(() => {
    if (currentSessionId) {
      const interval = setInterval(async () => {
        try {
          const status = await invisibilityService.getSessionStatus();
          setSessionStatus(status);
          
          const healthCheck = await invisibilityService.healthCheck();
          setHealthStatus(healthCheck);
          
          // Get performance metrics
          const perfMetrics = await invisibilityService.getPerformanceMetrics(currentSessionId);
          setPerformanceMetrics(perfMetrics);
          
          // Get UI state
          const uiState = await invisibilityService.getUIState();
          setUiHidden(uiState.is_hidden || false);
        } catch (err) {
          console.error('Failed to fetch session status:', err);
        }
      }, 5000);

      return () => clearInterval(interval);
    }
  }, [currentSessionId, invisibilityService]);

  const enableInvisibilityMode = async () => {
    setLoading(true);
    try {
      // Validate configuration first
      const isValid = await configValidator.validateConfiguration({
        recording: recordingConfig,
        ui: uiConfig,
        security: securityConfig
      });

      if (!isValid.valid) {
        throw new Error(`Configuration error: ${isValid.errors.join(', ')}`);
      }

      // Enable invisibility mode using the service
      const response = await invisibilityService.enableInvisibilityMode({
        recording: recordingConfig,
        ui: uiConfig,
        security: securityConfig
      });

      if (response.success) {
        // Session state will be updated via event listeners
        setError(null);
      } else {
        throw new Error(response.message);
      }
    } catch (err) {
      setError(`Failed to enable invisibility mode: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const disableInvisibilityMode = async () => {
    setLoading(true);
    try {
      const response = await invisibilityService.disableInvisibilityMode();
      
      if (response.success) {
        // Session state will be updated via event listeners
        setError(null);
      } else {
        throw new Error(response.message);
      }
    } catch (err) {
      setError(`Failed to disable invisibility mode: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const toggleRecording = async () => {
    if (!currentSessionId) return;

    setLoading(true);
    try {
      if (recording) {
        await invisibilityService.stopRecording();
      } else {
        await invisibilityService.startRecording({
          screenRecording: recordingConfig.screen_recording,
          voiceRecording: recordingConfig.voice_recording,
          autoNotes: recordingConfig.auto_notes,
          realTimeInsights: recordingConfig.real_time_insights,
          estimatedDuration: 60
        });
      }
      // Recording state will be updated via event listeners
    } catch (err) {
      setError(`Failed to ${recording ? 'stop' : 'start'} recording: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const toggleUIVisibility = async () => {
    if (!currentSessionId) return;

    setLoading(true);
    try {
      if (uiHidden) {
        await invisibilityService.showUIComponents(uiConfig.components_to_hide);
      } else {
        await invisibilityService.hideUIComponents(
          uiConfig.components_to_hide,
          uiConfig.hide_mode
        );
      }
      // UI state will be updated via event listeners
    } catch (err) {
      setError(`Failed to ${uiHidden ? 'show' : 'hide'} UI: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const generateInsights = async () => {
    if (!currentSessionId) return;

    setLoading(true);
    try {
      await invisibilityService.generateInsights([
        'conversation_analysis',
        'sentiment_tracking',
        'key_moments',
        'auto_summary'
      ]);
      // Insights will be updated via event listeners
    } catch (err) {
      setError(`Failed to generate insights: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const checkSecurity = async () => {
    if (!currentSessionId) return;

    try {
      const status = await invisibilityService.getSecurityStatus();
      setSecurityStatus(status);
    } catch (err) {
      setError(`Failed to check security: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    }
  };

  const updateUIConfiguration = async (newConfig) => {
    setLoading(true);
    try {
      const response = await invisibilityService.updateUIConfig(newConfig);
      setUiConfig(newConfig);
      setError(null);
      return response;
    } catch (err) {
      setError(`Failed to update UI config: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const updateSecurityConfiguration = async (newConfig) => {
    setLoading(true);
    try {
      const response = await invisibilityService.updateSecurityConfig(newConfig);
      setSecurityConfig(newConfig);
      setError(null);
      return response;
    } catch (err) {
      setError(`Failed to update security config: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const generateSpecificInsight = async (insightType) => {
    setLoading(true);
    try {
      const response = await invisibilityService.generateSingleInsight(insightType);
      setInsights(prev => ({
        ...prev,
        [insightType]: response
      }));
    } catch (err) {
      setError(`Failed to generate ${insightType}: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const getRecordingInfo = async (recordingType) => {
    try {
      const response = await invisibilityService.getRecording(recordingType);
      return response;
    } catch (err) {
      setError(`Failed to get recording info: ${err.message}`);
      setTimeout(() => setError(null), 5000);
      return null;
    }
  };

  const logError = async (errorData) => {
    try {
      await invisibilityService.createInvisibilityError({
        error_code: errorData.code || 'UNKNOWN',
        message: errorData.message,
        session_id: currentSessionId,
        details: errorData.details || {},
        timestamp: new Date().toISOString()
      });
    } catch (err) {
      console.error('Failed to log error:', err);
    }
  };

  const reportPerformanceMetrics = async () => {
    if (!currentSessionId) return;

    try {
      const metrics = {
        session_id: currentSessionId,
        cpu_usage: performanceMetrics?.cpu_usage || 0,
        memory_usage: performanceMetrics?.memory_usage || 0,
        disk_usage: performanceMetrics?.disk_usage || 0,
        network_activity: {},
        recording_quality_score: 0.85,
        processing_latency: 150,
        timestamp: new Date().toISOString()
      };

      await invisibilityService.createPerformanceMetrics(metrics);
    } catch (err) {
      setError(`Failed to report metrics: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    }
  };

  const cleanupSession = async () => {
    if (!currentSessionId) return;

    setLoading(true);
    try {
      await invisibilityService.cleanupSession();
      // Session state will be updated via event listeners
    } catch (err) {
      setError(`Failed to cleanup session: ${err.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  };

  const Button = ({ onClick, disabled, children, variant = 'primary', className = '' }) => {
    const baseClasses = 'px-6 py-3 rounded-lg font-medium font-roboto transition-all duration-200 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed';
    const variants = {
      primary: 'bg-[#8F74D4] hover:bg-[#7A63C3] text-white',
      secondary: 'bg-gray-600 hover:bg-gray-700 text-white',
      danger: 'bg-red-600 hover:bg-red-700 text-white'
    };

    return (
      <button
        onClick={onClick}
        disabled={disabled || loading}
        className={`${baseClasses} ${variants[variant]} ${className}`}
      >
        {children}
      </button>
    );
  };

  const StatusCard = ({ title, children, icon: Icon }) => (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
      <div className="flex items-center gap-3 mb-4">
        <Icon className="w-5 h-5 text-[#8F74D4]" />
        <h3 className="text-lg font-medium text-[#F8FAFC]">{title}</h3>
      </div>
      <div className="text-gray-300">
        {children}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#1E1E2F] text-[#F8FAFC] font-roboto">
      <div className="container mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Invisibility Mode Control Panel</h1>
          <p className="text-gray-400">Stealth recording and AI insights for interview sessions</p>
        </div>

        {error && (
          <div className="bg-red-900/50 border border-red-600 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {/* Main Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Session Control</h2>
            <div className="space-y-3">
              {!invisibilityEnabled ? (
                <Button onClick={enableInvisibilityMode} disabled={loading}>
                  <Eye className="w-4 h-4" />
                  Enable Invisibility Mode
                </Button>
              ) : (
                <Button onClick={disableInvisibilityMode} variant="danger" disabled={loading}>
                  <EyeOff className="w-4 h-4" />
                  Disable Invisibility Mode
                </Button>
              )}
              
              {invisibilityEnabled && (
                <>
                  <Button onClick={toggleRecording} disabled={loading}>
                    {recording ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    {recording ? 'Stop Recording' : 'Start Recording'}
                  </Button>
                  
                  <Button onClick={toggleUIVisibility} variant="secondary" disabled={loading}>
                    {uiHidden ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                    {uiHidden ? 'Show UI' : 'Hide UI'}
                  </Button>
                </>
              )}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">AI Operations</h2>
            <div className="space-y-3">
              <Button onClick={generateInsights} disabled={!currentSessionId || loading}>
                <Activity className="w-4 h-4" />
                Generate Insights
              </Button>
              
              <Button onClick={checkSecurity} variant="secondary" disabled={!currentSessionId || loading}>
                <Shield className="w-4 h-4" />
                Security Check
              </Button>
              
              {insights && (
                <Button onClick={() => setInsights(null)} variant="secondary">
                  <Download className="w-4 h-4" />
                  Download Results
                </Button>
              )}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Session Management</h2>
            <div className="space-y-3">
              <Button onClick={() => window.location.reload()} variant="secondary">
                <RefreshCw className="w-4 h-4" />
                Refresh Status
              </Button>
              
              <Button onClick={cleanupSession} variant="danger" disabled={!currentSessionId || loading}>
                <Trash2 className="w-4 h-4" />
                Cleanup Session
              </Button>
            </div>
          </div>
        </div>

        {/* Advanced Operations Panel */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Recording Operations</h2>
            <div className="space-y-3">
              <Button onClick={() => getRecordingInfo('screen')} variant="secondary" disabled={loading}>
                Get Screen Recording
              </Button>
              <Button onClick={() => getRecordingInfo('voice')} variant="secondary" disabled={loading}>
                Get Voice Recording
              </Button>
              <Button onClick={() => getRecordingInfo('combined')} variant="secondary" disabled={loading}>
                Get Combined Recording
              </Button>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Insight Types</h2>
            <div className="space-y-3">
              <Button onClick={() => generateSpecificInsight('conversation_analysis')} variant="secondary" disabled={!currentSessionId || loading}>
                Conversation Analysis
              </Button>
              <Button onClick={() => generateSpecificInsight('sentiment_tracking')} variant="secondary" disabled={!currentSessionId || loading}>
                Sentiment Tracking
              </Button>
              <Button onClick={() => generateSpecificInsight('key_moments')} variant="secondary" disabled={!currentSessionId || loading}>
                Key Moments
              </Button>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Configuration Updates</h2>
            <div className="space-y-3">
              <Button onClick={() => updateUIConfiguration(uiConfig)} variant="secondary" disabled={loading}>
                Update UI Config
              </Button>
              <Button onClick={() => updateSecurityConfiguration(securityConfig)} variant="secondary" disabled={loading}>
                Update Security Config
              </Button>
              <Button onClick={reportPerformanceMetrics} variant="secondary" disabled={!currentSessionId || loading}>
                Report Metrics
              </Button>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Error Management</h2>
            <div className="space-y-3">
              <Button onClick={() => logError({code: 'TEST_ERROR', message: 'Test error message', details: {test: true}})} variant="secondary" disabled={!currentSessionId || loading}>
                Log Test Error
              </Button>
              <Button onClick={() => invisibilityService.getInvisibilityError('TEST_ERROR').then(console.log)} variant="secondary" disabled={loading}>
                Get Error Info
              </Button>
            </div>
          </div>
        </div>

        {/* System Information Display */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {systemConfig && (
            <StatusCard title="System Configuration" icon={Settings}>
              <div className="space-y-2">
                <p>Max Sessions: <span className="text-[#8F74D4]">{systemConfig.max_concurrent_sessions}</span></p>
                <p>Max Duration: <span className="text-blue-400">{systemConfig.max_recording_duration}m</span></p>
                <p>Storage Path: <span className="text-green-400">{systemConfig.default_storage_path}</span></p>
                <p>Cleanup Interval: <span className="text-yellow-400">{systemConfig.cleanup_interval}h</span></p>
              </div>
            </StatusCard>
          )}

          {hideModes.length > 0 && (
            <StatusCard title="Available Hide Modes" icon={EyeOff}>
              <div className="space-y-2">
                {hideModes.map((mode, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="capitalize">{mode.replace(/_/g, ' ')}</span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      uiConfig.hide_mode === mode ? 'bg-[#8F74D4] text-white' : 'bg-gray-600 text-gray-300'
                    }`}>
                      {uiConfig.hide_mode === mode ? 'Active' : 'Available'}
                    </span>
                  </div>
                ))}
              </div>
            </StatusCard>
          )}

          {performanceMetrics && (
            <StatusCard title="Performance Metrics" icon={Activity}>
              <div className="space-y-2">
                {Array.isArray(performanceMetrics) && performanceMetrics.length > 0 ? (
                  performanceMetrics.slice(-1).map((metric, index) => (
                    <div key={index}>
                      <p>CPU: <span className="text-blue-400">{metric.cpu_usage?.toFixed(1)}%</span></p>
                      <p>Memory: <span className="text-green-400">{metric.memory_usage?.toFixed(1)}%</span></p>
                      <p>Disk: <span className="text-yellow-400">{metric.disk_usage?.toFixed(1)}%</span></p>
                      <p>Quality Score: <span className="text-[#8F74D4]">{(metric.recording_quality_score * 100)?.toFixed(0)}%</span></p>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-400">No metrics available</p>
                )}
              </div>
            </StatusCard>
          )}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold mb-4">Recording Config</h3>
            <div className="space-y-3">
              {Object.entries(recordingConfig).map(([key, value]) => (
                <label key={key} className="flex items-center gap-2">
                  {typeof value === 'boolean' ? (
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={(e) => setRecordingConfig(prev => ({
                        ...prev,
                        [key]: e.target.checked
                      }))}
                      className="rounded"
                    />
                  ) : (
                    <select
                      value={value}
                      onChange={(e) => setRecordingConfig(prev => ({
                        ...prev,
                        [key]: e.target.value
                      }))}
                      className="bg-gray-700 rounded px-2 py-1"
                    >
                      <option value={value}>{value}</option>
                    </select>
                  )}
                  <span className="text-sm capitalize">{key.replace(/_/g, ' ')}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold mb-4">UI Config</h3>
            <div className="space-y-3">
              <select
                value={uiConfig.hide_mode}
                onChange={(e) => setUiConfig(prev => ({ ...prev, hide_mode: e.target.value }))}
                className="w-full bg-gray-700 rounded px-3 py-2"
              >
                <option value="minimize">Minimize</option>
                <option value="hide_window">Hide Window</option>
                <option value="background_tab">Background Tab</option>
                <option value="separate_display">Separate Display</option>
              </select>
              
              {Object.entries(uiConfig).filter(([key]) => typeof uiConfig[key] === 'boolean').map(([key, value]) => (
                <label key={key} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={value}
                    onChange={(e) => setUiConfig(prev => ({
                      ...prev,
                      [key]: e.target.checked
                    }))}
                    className="rounded"
                  />
                  <span className="text-sm capitalize">{key.replace(/_/g, ' ')}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 className="text-lg font-semibold mb-4">Security Config</h3>
            <div className="space-y-3">
              {Object.entries(securityConfig).map(([key, value]) => (
                <label key={key} className="flex items-center gap-2">
                  {typeof value === 'boolean' ? (
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={(e) => setSecurityConfig(prev => ({
                        ...prev,
                        [key]: e.target.checked
                      }))}
                      className="rounded"
                    />
                  ) : typeof value === 'number' ? (
                    <input
                      type="number"
                      value={value}
                      onChange={(e) => setSecurityConfig(prev => ({
                        ...prev,
                        [key]: parseInt(e.target.value)
                      }))}
                      className="bg-gray-700 rounded px-2 py-1 w-20"
                    />
                  ) : null}
                  <span className="text-sm capitalize">{key.replace(/_/g, ' ')}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Status Information */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {sessionStatus && (
            <StatusCard title="Session Status" icon={Activity}>
              <div className="space-y-2">
                <p>Session ID: <span className="text-[#8F74D4]">{currentSessionId}</span></p>
                <p>Active: <span className={sessionStatus.is_active ? 'text-green-400' : 'text-red-400'}>
                  {sessionStatus.is_active ? 'Yes' : 'No'}
                </span></p>
                <p>Recording: <span className={sessionStatus.recording_status ? 'text-green-400' : 'text-gray-400'}>
                  {sessionStatus.recording_status ? 'Active' : 'Inactive'}
                </span></p>
                <p>UI State: <span className="text-blue-400">{sessionStatus.ui_state}</span></p>
                {sessionStatus.duration && <p>Duration: {Math.floor(sessionStatus.duration / 60)}m {sessionStatus.duration % 60}s</p>}
              </div>
            </StatusCard>
          )}

          {securityStatus && (
            <StatusCard title="Security Status" icon={Shield}>
              <div className="space-y-2">
                <p>Encryption: <span className={securityStatus.data_encrypted ? 'text-green-400' : 'text-red-400'}>
                  {securityStatus.data_encrypted ? 'Enabled' : 'Disabled'}
                </span></p>
                <p>Local Processing: <span className={securityStatus.local_processing ? 'text-green-400' : 'text-red-400'}>
                  {securityStatus.local_processing ? 'Yes' : 'No'}
                </span></p>
                <p>External Leaks: <span className={!securityStatus.no_external_leaks ? 'text-green-400' : 'text-red-400'}>
                  {!securityStatus.no_external_leaks ? 'None' : 'Detected'}
                </span></p>
                <p>Security Score: <span className="text-[#8F74D4] font-bold">{securityStatus.security_score}/100</span></p>
              </div>
            </StatusCard>
          )}

          {insights && (
            <div className="lg:col-span-2">
              <StatusCard title="Generated Insights" icon={Activity}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(insights.insights || {}).map(([type, data]) => (
                    <div key={type} className="bg-gray-700 rounded p-4">
                      <h4 className="font-medium capitalize mb-2">{type.replace(/_/g, ' ')}</h4>
                      {Array.isArray(data) && data.map((insight, idx) => (
                        <div key={idx} className="text-sm">
                          <p>Confidence: {(insight.confidence * 100).toFixed(1)}%</p>
                          <p className="text-gray-400 mt-1">{insight.content?.summary || 'No summary available'}</p>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </StatusCard>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default InvisibilityDashboard;