
export class InvisibilityService {
  constructor(baseURL = 'http://localhost:8000/api/v1/invisibility') {
    this.baseURL = baseURL;
    this.sessionId = null;
    this.wsConnection = null;
    this.eventListeners = new Map();
    this.retryConfig = {
      maxRetries: 3,
      baseDelay: 1000,
      maxDelay: 10000
    };
  }

  /**
   * Make authenticated API request with retry logic
   */
  async makeRequest(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    };

    let attempt = 0;
    while (attempt < this.retryConfig.maxRetries) {
      try {
        const response = await fetch(url, defaultOptions);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
      } catch (error) {
        attempt++;
        if (attempt >= this.retryConfig.maxRetries) {
          throw new Error(`API request failed after ${this.retryConfig.maxRetries} attempts: ${error.message}`);
        }
        
        // Exponential backoff
        const delay = Math.min(
          this.retryConfig.baseDelay * Math.pow(2, attempt - 1),
          this.retryConfig.maxDelay
        );
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  /**
   * Session Management Methods
   */
  async enableInvisibilityMode(config) {
    try {
      const response = await this.makeRequest('/mode/enable', {
        method: 'POST',
        body: JSON.stringify({
          recording_config: config.recording,
          ui_config: config.ui,
          security_config: config.security,
          session_name: config.sessionName,
          metadata: config.metadata || {}
        })
      });

      if (response.success) {
        this.sessionId = response.session_id;
        this.emit('session:created', { sessionId: this.sessionId });
        await this.initializeWebSocket();
      }

      return response;
    } catch (error) {
      this.emit('error', { action: 'enable_invisibility_mode', error: error.message });
      throw error;
    }
  }

  async disableInvisibilityMode() {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest('/mode/disable', {
        method: 'POST',
        body: JSON.stringify({ session_id: this.sessionId })
      });

      if (response.success) {
        const sessionId = this.sessionId;
        this.sessionId = null;
        this.closeWebSocket();
        this.emit('session:ended', { sessionId });
      }

      return response;
    } catch (error) {
      this.emit('error', { action: 'disable_invisibility_mode', error: error.message });
      throw error;
    }
  }

  /**
   * Recording Management Methods
   */
  async startRecording(config = {}) {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest('/recording/start', {
        method: 'POST',
        body: JSON.stringify({
          session_id: this.sessionId,
          screen_recording: config.screenRecording ?? true,
          voice_recording: config.voiceRecording ?? true,
          auto_notes: config.autoNotes ?? true,
          real_time_insights: config.realTimeInsights ?? false,
          estimated_duration: config.estimatedDuration
        })
      });

      if (response.success) {
        this.emit('recording:started', { sessionId: this.sessionId });
      }

      return response;
    } catch (error) {
      this.emit('error', { action: 'start_recording', error: error.message });
      throw error;
    }
  }

  async stopRecording() {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest('/recording/stop', {
        method: 'POST',
        body: JSON.stringify({ session_id: this.sessionId })
      });

      if (response.success) {
        this.emit('recording:stopped', { 
          sessionId: this.sessionId,
          duration: response.recording_duration,
          dataSize: response.data_size
        });
      }

      return response;
    } catch (error) {
      this.emit('error', { action: 'stop_recording', error: error.message });
      throw error;
    }
  }

  /**
   * UI Management Methods
   */
  async hideUIComponents(components, hideMode = 'minimize') {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest('/ui/hide', {
        method: 'POST',
        body: JSON.stringify({
          session_id: this.sessionId,
          components_to_hide: components,
          hide_mode: hideMode
        })
      });

      if (response.success) {
        this.emit('ui:hidden', { 
          sessionId: this.sessionId, 
          hiddenComponents: response.hidden_components 
        });
      }

      return response;
    } catch (error) {
      this.emit('error', { action: 'hide_ui_components', error: error.message });
      throw error;
    }
  }

  async showUIComponents(components) {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest('/ui/show', {
        method: 'POST',
        body: JSON.stringify({
          session_id: this.sessionId,
          components_to_show: components
        })
      });

      if (response.success) {
        this.emit('ui:shown', { 
          sessionId: this.sessionId, 
          visibleComponents: response.visible_components 
        });
      }

      return response;
    } catch (error) {
      this.emit('error', { action: 'show_ui_components', error: error.message });
      throw error;
    }
  }

  /**
   * Recording Management Methods - Extended
   */
  async getRecording(recordingType) {
    try {
      const response = await this.makeRequest(`/recording/${recordingType}`);
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_recording', error: error.message });
      throw error;
    }
  }

  async startRecordingWithConfig(config) {
    try {
      const response = await this.makeRequest('/recording/start', {
        method: 'POST',
        body: JSON.stringify(config)
      });
      return response;
    } catch (error) {
      this.emit('error', { action: 'start_recording_with_config', error: error.message });
      throw error;
    }
  }

  /**
   * UI State Management - Extended
   */
  async getUIState() {
    try {
      const response = await this.makeRequest('/ui/state');
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_ui_state', error: error.message });
      throw error;
    }
  }

  async updateUIConfig(config) {
    try {
      const response = await this.makeRequest('/ui/config', {
        method: 'PUT',
        body: JSON.stringify(config)
      });
      return response;
    } catch (error) {
      this.emit('error', { action: 'update_ui_config', error: error.message });
      throw error;
    }
  }

  /**
   * Insights Management Methods - Extended
   */
  async generateInsights(insightTypes, processingOptions = {}) {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest('/insights/generate', {
        method: 'POST',
        body: JSON.stringify({
          session_id: this.sessionId,
          insight_types: insightTypes,
          processing_options: processingOptions,
          priority: processingOptions.priority || 'normal'
        })
      });

      if (response.success) {
        this.emit('insights:generation_started', { 
          sessionId: this.sessionId,
          types: insightTypes,
          estimatedCompletion: response.estimated_completion_time
        });

        // Poll for completion
        this.pollForInsights();
      }

      return response;
    } catch (error) {
      this.emit('error', { action: 'generate_insights', error: error.message });
      throw error;
    }
  }

  /**
   * Security Management - Extended
   */
  async getSecurityStatusGeneral() {
    try {
      const response = await this.makeRequest('/security/status');
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_security_status_general', error: error.message });
      throw error;
    }
  }

  async updateSecurityConfig(config) {
    try {
      const response = await this.makeRequest('/security/config', {
        method: 'POST',
        body: JSON.stringify(config)
      });
      return response;
    } catch (error) {
      this.emit('error', { action: 'update_security_config', error: error.message });
      throw error;
    }
  }

  /**
   * System Configuration Methods
   */
  async getSystemConfig() {
    try {
      const response = await this.makeRequest('/system/config');
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_system_config', error: error.message });
      throw error;
    }
  }

  async getGeneralSessionData() {
    try {
      const response = await this.makeRequest('/session');
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_general_session_data', error: error.message });
      throw error;
    }
  }

  /**
   * Hide Mode Management
   */
  async getHideModes() {
    try {
      const response = await this.makeRequest('/hide-modes');
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_hide_modes', error: error.message });
      throw error;
    }
  }

  async getHideMode(mode) {
    try {
      const response = await this.makeRequest(`/hide-modes/${mode}`);
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_hide_mode', error: error.message });
      throw error;
    }
  }

  /**
   * Error Management
   */
  async createInvisibilityError(error) {
    try {
      const response = await this.makeRequest('/invisibility-error', {
        method: 'POST',
        body: JSON.stringify(error)
      });
      return response;
    } catch (error) {
      this.emit('error', { action: 'create_invisibility_error', error: error.message });
      throw error;
    }
  }

  async getInvisibilityError(errorCode) {
    try {
      const response = await this.makeRequest(`/invisibility-error/${errorCode}`);
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_invisibility_error', error: error.message });
      throw error;
    }
  }

  /**
   * Performance Metrics
   */
  async createPerformanceMetrics(metrics) {
    try {
      const response = await this.makeRequest('/performance-metrics', {
        method: 'POST',
        body: JSON.stringify(metrics)
      });
      return response;
    } catch (error) {
      this.emit('error', { action: 'create_performance_metrics', error: error.message });
      throw error;
    }
  }

  async getPerformanceMetrics(sessionId = null) {
    try {
      const endpoint = sessionId ? `/performance-metrics/${sessionId}` : '/performance-metrics';
      const response = await this.makeRequest(endpoint);
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_performance_metrics', error: error.message });
      throw error;
    }
  }

  async listAllPerformanceMetrics(limit = 50) {
    try {
      const response = await this.makeRequest(`/performance-metrics?limit=${limit}`);
      return response;
    } catch (error) {
      this.emit('error', { action: 'list_all_performance_metrics', error: error.message });
      throw error;
    }
  }

  async getInsights() {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest(`/insights/${this.sessionId}`);
      this.emit('insights:retrieved', { sessionId: this.sessionId, insights: response });
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_insights', error: error.message });
      throw error;
    }
  }

  async pollForInsights() {
    const maxAttempts = 20; // 2 minutes with 6-second intervals
    let attempts = 0;

    const poll = async () => {
      try {
        attempts++;
        const insights = await this.getInsights();
        
        if (insights && insights.insights) {
          this.emit('insights:generated', { 
            sessionId: this.sessionId, 
            insights: insights.insights 
          });
          return;
        }

        if (attempts < maxAttempts) {
          setTimeout(poll, 6000); // Poll every 6 seconds
        } else {
          this.emit('insights:timeout', { sessionId: this.sessionId });
        }
      } catch (error) {
        if (attempts < maxAttempts) {
          setTimeout(poll, 6000);
        } else {
          this.emit('error', { action: 'poll_insights', error: 'Insights polling timeout' });
        }
      }
    };

    setTimeout(poll, 3000); // Start polling after 3 seconds
  }

  /**
   * Security and Status Methods
   */
  async getSessionStatus() {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest(`/session/${this.sessionId}/status`);
      this.emit('session:status_updated', { sessionId: this.sessionId, status: response });
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_session_status', error: error.message });
      throw error;
    }
  }

  async getSecurityStatus() {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest(`/security/status/${this.sessionId}`);
      this.emit('security:status_updated', { sessionId: this.sessionId, status: response });
      return response;
    } catch (error) {
      this.emit('error', { action: 'get_security_status', error: error.message });
      throw error;
    }
  }

  async cleanupSession() {
    if (!this.sessionId) {
      throw new Error('No active session found');
    }

    try {
      const response = await this.makeRequest(`/session/${this.sessionId}`, {
        method: 'DELETE'
      });

      if (response.success) {
        const sessionId = this.sessionId;
        this.sessionId = null;
        this.closeWebSocket();
        this.emit('session:cleaned', { sessionId, dataRemoved: response.data_removed });
      }

      return response;
    } catch (error) {
      this.emit('error', { action: 'cleanup_session', error: error.message });
      throw error;
    }
  }

  async healthCheck() {
    try {
      const response = await this.makeRequest('/health');
      this.emit('health:checked', { status: response });
      return response;
    } catch (error) {
      this.emit('error', { action: 'health_check', error: error.message });
      throw error;
    }
  }

  /**
   * WebSocket Connection Management
   */
  async initializeWebSocket() {
    if (!this.sessionId) return;

    try {
      const wsUrl = `ws://localhost:8000/ws/invisibility/${this.sessionId}`;
      this.wsConnection = new WebSocket(wsUrl);

      this.wsConnection.onopen = () => {
        this.emit('websocket:connected', { sessionId: this.sessionId });
      };

      this.wsConnection.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleWebSocketMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.wsConnection.onclose = () => {
        this.emit('websocket:disconnected', { sessionId: this.sessionId });
      };

      this.wsConnection.onerror = (error) => {
        this.emit('websocket:error', { sessionId: this.sessionId, error });
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  handleWebSocketMessage(data) {
    switch (data.type) {
      case 'recording_status':
        this.emit('recording:status_update', data.payload);
        break;
      case 'ui_status':
        this.emit('ui:status_update', data.payload);
        break;
      case 'security_alert':
        this.emit('security:alert', data.payload);
        break;
      case 'insight_progress':
        this.emit('insights:progress', data.payload);
        break;
      case 'error':
        this.emit('error', { action: 'websocket_message', error: data.payload.message });
        break;
      default:
        console.warn('Unknown WebSocket message type:', data.type);
    }
  }

  closeWebSocket() {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = null;
    }
  }

  /**
   * Event Management
   */
  on(event, callback) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event).push(callback);
  }

  off(event, callback) {
    if (this.eventListeners.has(event)) {
      const listeners = this.eventListeners.get(event);
      const index = listeners.indexOf(callback);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  removeAllListeners() {
    this.eventListeners.clear();
  }

  /**
   * Configuration and Validation Methods
   */
  getDefaultConfiguration() {
    return {
      recording: {
        screen_recording: true,
        voice_recording: true,
        auto_notes: true,
        real_time_insights: false,
        recording_quality: 'medium',
        audio_format: 'mp3',
        video_format: 'mp4',
        max_duration: null
      },
      ui: {
        hide_mode: 'minimize',
        components_to_hide: ['recording_indicator', 'ai_insights_panel'],
        keep_separate_window: false,
        minimize_to_tray: true,
        show_discrete_indicator: false
      },
      security: {
        local_processing_only: true,
        encrypt_data: true,
        auto_delete_after: 24,
        no_cloud_upload: true,
        secure_storage_path: null
      }
    };
  }
}

/**
 * Session Manager Class - Handles session lifecycle
 */
export class SessionManager {
  constructor() {
    this.sessions = new Map();
    this.activeSessionId = null;
  }

  createSession(sessionId, config) {
    const session = {
      id: sessionId,
      config,
      createdAt: new Date(),
      status: 'created',
      recording: false,
      uiHidden: false,
      insights: null,
      securityStatus: null
    };

    this.sessions.set(sessionId, session);
    this.activeSessionId = sessionId;
    return session;
  }

  getSession(sessionId = null) {
    const id = sessionId || this.activeSessionId;
    return this.sessions.get(id);
  }

  updateSession(sessionId, updates) {
    const session = this.sessions.get(sessionId);
    if (session) {
      Object.assign(session, updates, { updatedAt: new Date() });
    }
    return session;
  }

  deleteSession(sessionId) {
    const deleted = this.sessions.delete(sessionId);
    if (this.activeSessionId === sessionId) {
      this.activeSessionId = null;
    }
    return deleted;
  }

  getAllSessions() {
    return Array.from(this.sessions.values());
  }

  getActiveSession() {
    return this.activeSessionId ? this.sessions.get(this.activeSessionId) : null;
  }
}

/**
 * Configuration Validator Class - Validates configuration objects
 */
export class ConfigurationValidator {
  async validateConfiguration(config) {
    const errors = [];

    // Validate recording configuration
    if (config.recording) {
      if (typeof config.recording.screen_recording !== 'boolean') {
        errors.push('screen_recording must be a boolean');
      }
      if (typeof config.recording.voice_recording !== 'boolean') {
        errors.push('voice_recording must be a boolean');
      }
      if (!['low', 'medium', 'high'].includes(config.recording.recording_quality)) {
        errors.push('recording_quality must be low, medium, or high');
      }
    }

    // Validate UI configuration
    if (config.ui) {
      const validHideModes = ['minimize', 'hide_window', 'background_tab', 'separate_display'];
      if (!validHideModes.includes(config.ui.hide_mode)) {
        errors.push('hide_mode must be one of: ' + validHideModes.join(', '));
      }
    }

    // Validate security configuration
    if (config.security) {
      if (config.security.auto_delete_after && typeof config.security.auto_delete_after !== 'number') {
        errors.push('auto_delete_after must be a number (hours)');
      }
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  validateInsightTypes(types) {
    const validTypes = [
      'conversation_analysis',
      'sentiment_tracking', 
      'key_moments',
      'performance_metrics',
      'auto_summary',
      'question_analysis'
    ];

    const errors = types.filter(type => !validTypes.includes(type));
    
    return {
      valid: errors.length === 0,
      errors: errors.map(type => `Invalid insight type: ${type}`)
    };
  }
}

/**
 * UI State Manager Class - Manages UI state changes
 */
export class UIStateManager {
  constructor() {
    this.state = {
      isHidden: false,
      hiddenComponents: [],
      hideMode: null,
      windowPositions: {},
      lastUpdated: null
    };
    this.stateHistory = [];
  }

  updateState(newState) {
    // Save current state to history
    this.stateHistory.push({
      ...this.state,
      timestamp: new Date()
    });

    // Keep only last 10 state changes
    if (this.stateHistory.length > 10) {
      this.stateHistory.shift();
    }

    // Update current state
    this.state = {
      ...this.state,
      ...newState,
      lastUpdated: new Date()
    };

    return this.state;
  }

  hideComponents(components, hideMode) {
    return this.updateState({
      isHidden: true,
      hiddenComponents: components,
      hideMode: hideMode
    });
  }

  showComponents(components = []) {
    const componentsToShow = components.length ? components : this.state.hiddenComponents;
    
    return this.updateState({
      isHidden: false,
      hiddenComponents: this.state.hiddenComponents.filter(c => !componentsToShow.includes(c)),
      hideMode: this.state.hiddenComponents.length === componentsToShow.length ? null : this.state.hideMode
    });
  }

  restorePreviousState() {
    if (this.stateHistory.length > 0) {
      const previousState = this.stateHistory.pop();
      this.state = {
        ...previousState,
        lastUpdated: new Date()
      };
      return this.state;
    }
    return null;
  }

  getState() {
    return { ...this.state };
  }

  getStateHistory() {
    return [...this.stateHistory];
  }

  clearHistory() {
    this.stateHistory = [];
  }
}

/**
 * Utility Functions
 */
export const InvisibilityUtils = {
  formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  },

  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  },

  validateSessionId(sessionId) {
    return typeof sessionId === 'string' && sessionId.length > 0;
  },

  getComponentDisplayName(component) {
    return component.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  },

  sanitizeConfig(config) {
    // Remove any sensitive data from config before logging/displaying
    const sanitized = JSON.parse(JSON.stringify(config));
    
    if (sanitized.security && sanitized.security.secure_storage_path) {
      sanitized.security.secure_storage_path = '[REDACTED]';
    }
    
    return sanitized;
  },

  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  throttle(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }
};

/**
 * Error Handler Class - Centralized error handling
 */
export class ErrorHandler {
  constructor() {
    this.errorLog = [];
    this.maxLogSize = 100;
  }

  logError(error, context = {}) {
    const errorEntry = {
      timestamp: new Date(),
      message: error.message || error,
      stack: error.stack,
      context,
      id: this.generateErrorId()
    };

    this.errorLog.unshift(errorEntry);
    
    // Keep log size manageable
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog = this.errorLog.slice(0, this.maxLogSize);
    }

    console.error('InvisibilityService Error:', errorEntry);
    return errorEntry.id;
  }

  getErrors(limit = 10) {
    return this.errorLog.slice(0, limit);
  }

  clearErrors() {
    this.errorLog = [];
  }

  generateErrorId() {
    return 'err_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
  }
}

/**
 * Performance Monitor Class - Tracks system performance
 */
export class PerformanceMonitor {
  constructor() {
    this.metrics = {
      apiCalls: 0,
      averageResponseTime: 0,
      errorRate: 0,
      sessionDuration: 0,
      dataTransferred: 0
    };
    this.callTimes = [];
  }

  recordAPICall(endpoint, responseTime, success = true) {
    this.metrics.apiCalls++;
    this.callTimes.push(responseTime);
    
    // Keep only last 50 call times for average calculation
    if (this.callTimes.length > 50) {
      this.callTimes.shift();
    }
    
    this.metrics.averageResponseTime = this.callTimes.reduce((a, b) => a + b, 0) / this.callTimes.length;
    
    if (!success) {
      this.metrics.errorRate = (this.metrics.errorRate * (this.metrics.apiCalls - 1) + 1) / this.metrics.apiCalls;
    } else {
      this.metrics.errorRate = (this.metrics.errorRate * (this.metrics.apiCalls - 1)) / this.metrics.apiCalls;
    }

    console.debug(`API Call: ${endpoint} - ${responseTime}ms - ${success ? 'Success' : 'Failed'}`);
  }

  updateSessionDuration(duration) {
    this.metrics.sessionDuration = duration;
  }

  updateDataTransferred(bytes) {
    this.metrics.dataTransferred += bytes;
  }

  getMetrics() {
    return { ...this.metrics };
  }

  reset() {
    this.metrics = {
      apiCalls: 0,
      averageResponseTime: 0,
      errorRate: 0,
      sessionDuration: 0,
      dataTransferred: 0
    };
    this.callTimes = [];
  }
}

/**
 * Storage Manager Class - Handles local data storage
 */
export class StorageManager {
  constructor(prefix = 'invisibility_') {
    this.prefix = prefix;
    this.isSupported = typeof Storage !== 'undefined';
  }

  set(key, value, expiration = null) {
    if (!this.isSupported) return false;

    try {
      const data = {
        value,
        timestamp: Date.now(),
        expiration: expiration ? Date.now() + expiration : null
      };

      localStorage.setItem(this.prefix + key, JSON.stringify(data));
      return true;
    } catch (error) {
      console.error('Failed to store data:', error);
      return false;
    }
  }

  get(key) {
    if (!this.isSupported) return null;

    try {
      const item = localStorage.getItem(this.prefix + key);
      if (!item) return null;

      const data = JSON.parse(item);
      
      // Check expiration
      if (data.expiration && Date.now() > data.expiration) {
        this.remove(key);
        return null;
      }

      return data.value;
    } catch (error) {
      console.error('Failed to retrieve data:', error);
      return null;
    }
  }

  remove(key) {
    if (!this.isSupported) return false;

    try {
      localStorage.removeItem(this.prefix + key);
      return true;
    } catch (error) {
      console.error('Failed to remove data:', error);
      return false;
    }
  }

  clear() {
    if (!this.isSupported) return false;

    try {
      const keys = Object.keys(localStorage).filter(key => key.startsWith(this.prefix));
      keys.forEach(key => localStorage.removeItem(key));
      return true;
    } catch (error) {
      console.error('Failed to clear data:', error);
      return false;
    }
  }

  getAll() {
    if (!this.isSupported) return {};

    const result = {};
    try {
      Object.keys(localStorage)
        .filter(key => key.startsWith(this.prefix))
        .forEach(key => {
          const shortKey = key.replace(this.prefix, '');
          result[shortKey] = this.get(shortKey);
        });
    } catch (error) {
      console.error('Failed to get all data:', error);
    }

    return result;
  }
}

/**
 * Main Export - Factory function to create configured service instance
 */
export function createInvisibilityService(config = {}) {
  const service = new InvisibilityService(config.baseURL);
  const sessionManager = new SessionManager();
  const configValidator = new ConfigurationValidator();
  const uiStateManager = new UIStateManager();
  const errorHandler = new ErrorHandler();
  const performanceMonitor = new PerformanceMonitor();
  const storageManager = new StorageManager();

  // Wire up error handling
  service.on('error', (data) => {
    errorHandler.logError(new Error(data.error), { action: data.action });
  });

  // Wire up performance monitoring
  const originalMakeRequest = service.makeRequest;
  service.makeRequest = async function(endpoint, options = {}) {
    const startTime = performance.now();
    let success = false;
    
    try {
      const result = await originalMakeRequest.call(this, endpoint, options);
      success = true;
      return result;
    } catch (error) {
      throw error;
    } finally {
      const endTime = performance.now();
      performanceMonitor.recordAPICall(endpoint, endTime - startTime, success);
    }
  };

  return {
    service,
    sessionManager,
    configValidator,
    uiStateManager,
    errorHandler,
    performanceMonitor,
    storageManager,
    utils: InvisibilityUtils
  };
}