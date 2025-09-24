import React, { useState, useEffect } from 'react';
import { Play, Square, Eye, EyeOff, Settings, Shield, Activity, Download, Trash2, RefreshCw } from 'lucide-react';
import { InvisibilityService, SessionManager, ConfigurationValidator, UIStateManager } from './invisibilityService.js';

// Legacy API wrapper for backward compatibility
class InvisibilityAPI {
  constructor() {
    this.baseURL = 'http://localhost:8000/api/v1/invisibility';
  }

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
  const [healthStatus, setHealthStatus] = useState(null);

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

  // Session monitoring
  useEffect(() => {
    if (currentSessionId) {
      const interval = setInterval(async () => {
        try {
          const status = await invisibilityService.getSessionStatus();
          setSessionStatus(status);
          
          const healthCheck = await invisibilityService.healthCheck();
          setHealthStatus(healthCheck);
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

        {/* Configuration Panels */}
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