/**
 * Invisibility Service Handler
 * Manages all backend communication and state for the invisibility feature
 */

class InvisibilityService {
  constructor() {
    this.baseUrl = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1/invisibility';
    this.sessions = new Map();
    this.eventListeners = new Map();
    this.wsConnection = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    
    // Initialize WebSocket connection for real-time updates
    this.initializeWebSocket();
  }

  // WebSocket Management
  initializeWebSocket() {
    try {
      const wsUrl = this.baseUrl.replace('http', 'ws').replace('/api/v1/invisibility', '/ws');
      this.wsConnection = new WebSocket(wsUrl);
      
      this.wsConnection.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
      };
      
      this.wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleWebSocketMessage(data);
      };
      
      this.wsConnection.onclose = () => {
        console.log('WebSocket disconnected');
        this.handleReconnection();
      };
      
      this.wsConnection.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  handleReconnection() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
        this.initializeWebSocket();
      }, 1000 * this.reconnectAttempts);
    }
  }

  handleWebSocketMessage(data) {
    const { type, session_id, payload } = data;
    
    switch (type) {
      case 'session_update':
        this.updateSessionData(session_id, payload);
        break;
      case 'recording_status':
        this.handleRecordingStatusUpdate(session_id, payload);
        break;
      case 'ui_visibility':
        this.handleUIVisibilityUpdate(session_id, payload);
        break;
      case 'insights_ready':
        this.handleInsightsReady(session_id, payload);
        break;
      case 'security_alert':
        this.handleSecurityAlert(session_id, payload);
        break;
      default:
        console.log('Unknown WebSocket message type:', type);
    }
  }

  // Event Management
  addEventListener(eventType, callback) {
    if (!this.eventListeners.has(eventType)) {
      this.eventListeners.set(eventType, []);
    }
    this.eventListeners.get(eventType).push(callback);
  }

  removeEventListener(eventType, callback) {
    if (this.eventListeners.has(eventType)) {
      const listeners = this.eventListeners.get(eventType);
      const index = listeners.indexOf(callback);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
  }

  emit(eventType, data) {
    if (this.eventListeners.has(eventType)) {
      this.eventListeners.get(eventType).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Event listener error:', error);
        }
      });
    }
  }

  // HTTP Request Helper
  async makeRequest(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      credentials: 'include'
    };

    const requestOptions = { ...defaultOptions, ...options };
    
    try {
      const response = await fetch(url, requestOptions);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: 'Request failed' }));
        throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Request failed for ${endpoint}:`, error);
      this.emit('error', { endpoint, error: error.message });
      throw error;
    }
  }

  // Session Management
  async enableInvisibilityMode(config) {
    try {
      const response = await this.makeRequest('/mode/enable', {
        method: 'POST',
        body: JSON.stringify(config)
      });
      
      if (response.success) {
        this.sessions.set(response.session_id, {
          id: response.session_id,
          status: 'active',
          config,
          startTime: new Date(),
          ...response
        });
        
        this.emit('session_created', response);
        return response;
      }
      
      throw new Error(response.message || 'Failed to enable invisibility mode');
    } catch (error) {
      this.emit('session_error', { action: 'enable', error: error.message });
      throw error;
    }
  }

  async disableInvisibilityMode(sessionId) {
    try {
      const response = await this.makeRequest('/mode/disable', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId })
      });
      
      if (response.success) {
        const session = this.sessions.get(sessionId);
        if (session) {
          session.status = 'stopped';
          session.endTime = new Date();
        }
        
        this.emit('session_stopped', response);
        return response;
      }
      
      throw new Error(response.message || 'Failed to disable invisibility mode');
    } catch (error) {
      this.emit('session_error', { action: 'disable', error: error.message });
      throw error;
    }
  }

  async getSessionStatus(sessionId) {
    try {
      const response = await this.makeRequest(`/session/${sessionId}/status`);
      
      // Update local session data
      const session = this.sessions.get(sessionId);
      if (session) {
        Object.assign(session, response);
      }
      
      this.emit('session_status_updated', { sessionId, status: response });
      return response;
    } catch (error) {
      this.emit('session_error', { action: 'status', sessionId, error: error.message });
      throw error;
    }
  }

    // ===== New methods added =====

  // UI Config methods
  async getUIState() {
    return await this.makeRequest('/ui/state');
  }

  async updateUIConfig(config) {
    return await this.makeRequest('/ui/config', {
      method: 'PUT',
      body: JSON.stringify(config)
    });
  }

  // Hide Modes
  async listHideModes() {
    return await this.makeRequest('/hide-modes');
  }

  async getHideMode(mode) {
    return await this.makeRequest(`/hide-modes/${encodeURIComponent(mode)}`);
  }

  // Invisibility errors (store / retrieve)
  async createInvisibilityError(errorObj) {
    return await this.makeRequest('/invisibility-error', {
      method: 'POST',
      body: JSON.stringify(errorObj)
    });
  }

  async getInvisibilityError(errorCode) {
    return await this.makeRequest(`/invisibility-error/${encodeURIComponent(errorCode)}`);
  }

  // Performance Metrics
  async createPerformanceMetrics(metrics) {
    return await this.makeRequest('/performance-metrics', {
      method: 'POST',
      body: JSON.stringify(metrics)
    });
  }

  async getPerformanceMetrics(sessionId) {
    return await this.makeRequest(`/performance-metrics/${encodeURIComponent(sessionId)}`);
  }

  async listAllPerformanceMetrics(limit = 50) {
    return await this.makeRequest(`/performance-metrics?limit=${encodeURIComponent(limit)}`);
  }


  // Recording Management
  async startRecording(config) {
    try {
      const response = await this.makeRequest('/recording/start', {
        method: 'POST',
        body: JSON.stringify(config)
      });
      
      if (response.success) {
        this.emit('recording_started', response);
        return response;
      }
      
      throw new Error(response.message || 'Failed to start recording');
    } catch (error) {
      this.emit('recording_error', { action: 'start', error: error.message });
      throw error;
    }
  }

  async stopRecording(sessionId) {
    try {
      const response = await this.makeRequest('/recording/stop', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId })
      });
      
      if (response.success) {
        this.emit('recording_stopped', response);
        return response;
      }
      
      throw new Error(response.message || 'Failed to stop recording');
    } catch (error) {
      this.emit('recording_error', { action: 'stop', error: error.message });
      throw error;
    }
  }

  // UI Management
  async hideUIComponents(sessionId, components, hideMode) {
    try {
      const response = await this.makeRequest('/ui/hide', {
        method: 'POST',
        body: JSON.stringify({
          session_id: sessionId,
          components_to_hide: components,
          hide_mode: hideMode
        })
      });
      
      if (response.success) {
        this.emit('ui_hidden', response);
        return response;
      }
      
      throw new Error(response.message || 'Failed to hide UI components');
    } catch (error) {
      this.emit('ui_error', { action: 'hide', error: error.message });
      throw error;
    }
  }

  async showUIComponents(sessionId, components) {
    try {
      const response = await this.makeRequest('/ui/show', {
        method: 'POST',
        body: JSON.stringify({
          session_id: sessionId,
          components_to_show: components
        })
      });
      
      if (response.success) {
        this.emit('ui_shown', response);
        return response;
      }
      
      throw new Error(response.message || 'Failed to show UI components');
    } catch (error) {
      this.emit('ui_error', { action: 'show', error: error.message });
      throw error;
    }
  }

  // Insights Management
  async generateInsights(sessionId, insightTypes, options = {}) {
    try {
      const response = await this.makeRequest('/insights/generate', {
        method: 'POST',
        body: JSON.stringify({
          session_id: sessionId,
          insight_types: insightTypes,
          processing_options: options
        })
      });
      
      if (response.success) {
        this.emit('insights_generation_started', response);
        return response;
      }
      
      throw new Error(response.message || 'Failed to generate insights');
    } catch (error) {
      this.emit('insights_error', { action: 'generate', error: error.message });
      throw error;
    }
  }

  async getInsights(sessionId) {
    try {
      const response = await this.makeRequest(`/insights/${sessionId}`);
      
      this.emit('insights_received', { sessionId, insights: response.insights });
      return response;
    } catch (error) {
      if (error.message.includes('404')) {
        return { insights: null, message: 'No insights available' };
      }
      this.emit('insights_error', { action: 'get', error: error.message });
      throw error;
    }
  }

  // Security Management
  async getSecurityStatus(sessionId) {
    try {
      const response = await this.makeRequest(`/security/status/${sessionId}`);
      
      this.emit('security_status_updated', { sessionId, status: response });
      return response;
    } catch (error) {
      this.emit('security_error', { action: 'status', error: error.message });
      throw error;
    }
  }

  // Session Cleanup
  async cleanupSession(sessionId) {
    try {
      const response = await this.makeRequest(`/session/${sessionId}`, {
        method: 'DELETE'
      });
      
      // Remove from local storage
      this.sessions.delete(sessionId);
      
      this.emit('session_cleaned', { sessionId, response });
      return response;
    } catch (error) {
      this.emit('session_error', { action: 'cleanup', error: error.message });
      throw error;
    }
  }

  // Health Check
  async healthCheck() {
    try {
      const response = await this.makeRequest('/health');
      this.emit('health_check_complete', response);
      return response;
    } catch (error) {
      this.emit('health_check_failed', { error: error.message });
      throw error;
    }
  }

  // Utility Methods
  getSession(sessionId) {
    return this.sessions.get(sessionId);
  }

  getAllSessions() {
    return Array.from(this.sessions.values());
  }

  getActiveSessions() {
    return this.getAllSessions().filter(session => session.status === 'active');
  }

  // Local Storage Management
  saveSessionToStorage(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      try {
        const sessionData = {
          ...session,
          savedAt: new Date().toISOString()
        };
        localStorage.setItem(`invisibility_session_${sessionId}`, JSON.stringify(sessionData));
      } catch (error) {
        console.error('Failed to save session to storage:', error);
      }
    }
  }

  loadSessionFromStorage(sessionId) {
    try {
      const stored = localStorage.getItem(`invisibility_session_${sessionId}`);
      if (stored) {
        const sessionData = JSON.parse(stored);
        this.sessions.set(sessionId, sessionData);
        return sessionData;
      }
    } catch (error) {
      console.error('Failed to load session from storage:', error);
    }
    return null;
  }

  clearSessionStorage(sessionId) {
    try {
      localStorage.removeItem(`invisibility_session_${sessionId}`);
    } catch (error) {
      console.error('Failed to clear session storage:', error);
    }
  }

  // Event Handlers for WebSocket Updates
  updateSessionData(sessionId, data) {
    const session = this.sessions.get(sessionId);
    if (session) {
      Object.assign(session, data);
      this.emit('session_updated', { sessionId, data });
    }
  }

  handleRecordingStatusUpdate(sessionId, payload) {
    this.emit('recording_status_changed', { sessionId, ...payload });
  }

  handleUIVisibilityUpdate(sessionId, payload) {
    this.emit('ui_visibility_changed', { sessionId, ...payload });
  }

  handleInsightsReady(sessionId, payload) {
    this.emit('insights_ready', { sessionId, ...payload });
  }

  handleSecurityAlert(sessionId, payload) {
    this.emit('security_alert', { sessionId, ...payload });
  }

  // Configuration Management
  getDefaultConfig() {
    return {
      recording_config: {
        screen_recording: true,
        voice_recording: true,
        auto_notes: true,
        real_time_insights: false,
        recording_quality: 'medium',
        audio_format: 'mp3',
        video_format: 'mp4',
        max_duration: null
      },
      ui_config: {
        hide_mode: 'minimize',
        components_to_hide: ['main_window', 'controls_bar'],
        keep_separate_window: false,
        minimize_to_tray: true,
        show_discrete_indicator: false
      },
      security_config: {
        local_processing_only: true,
        encrypt_data: true,
        auto_delete_after: 24,
        no_cloud_upload: true,
        secure_storage_path: null
      }
    };
  }

  validateConfig(config) {
    const errors = [];
    
    // Validate recording config
    if (config.recording_config) {
      const rc = config.recording_config;
      if (rc.recording_quality && !['low', 'medium', 'high'].includes(rc.recording_quality)) {
        errors.push('Invalid recording quality');
      }
      if (rc.max_duration && (rc.max_duration < 1 || rc.max_duration > 300)) {
        errors.push('Max duration must be between 1 and 300 minutes');
      }
    }
    
    // Validate UI config
    if (config.ui_config) {
      const uc = config.ui_config;
      const validHideModes = ['minimize', 'hide_window', 'background_tab', 'separate_display'];
      if (uc.hide_mode && !validHideModes.includes(uc.hide_mode)) {
        errors.push('Invalid hide mode');
      }
    }
    
    return errors;
  }

  // Performance Monitoring
  startPerformanceMonitoring(sessionId) {
    const interval = setInterval(async () => {
      try {
        const performance = {
          memory: this.getMemoryUsage(),
          timestamp: new Date().toISOString(),
          sessionId
        };
        
        this.emit('performance_update', performance);
      } catch (error) {
        console.error('Performance monitoring error:', error);
      }
    }, 30000); // Every 30 seconds
    
    // Store interval ID for cleanup
    const session = this.sessions.get(sessionId);
    if (session) {
      session.performanceInterval = interval;
    }
    
    return interval;
  }

  stopPerformanceMonitoring(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session && session.performanceInterval) {
      clearInterval(session.performanceInterval);
      delete session.performanceInterval;
    }
  }

  getMemoryUsage() {
    if (performance.memory) {
      return {
        used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
        limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
      };
    }
    return null;
  }

  // Cleanup
  destroy() {
    // Close WebSocket connection
    if (this.wsConnection) {
      this.wsConnection.close();
    }
    
    // Stop all performance monitoring
    this.sessions.forEach((session, sessionId) => {
      this.stopPerformanceMonitoring(sessionId);
    });
    
    // Clear all sessions
    this.sessions.clear();
    
    // Clear event listeners
    this.eventListeners.clear();
  }
}

// Export singleton instance
const invisibilityService = new InvisibilityService();
export default invisibilityService;

// Also export the class for testing
export { InvisibilityService };

// Utility functions for frontend components
export const InvisibilityUtils = {
  formatDuration: (seconds) => {
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

  formatFileSize: (bytes) => {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`;
  },

  generateSessionId: () => {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
  },

  isValidSessionId: (sessionId) => {
    return /^session_\d+_[a-z0-9]{9}$/.test(sessionId);
  },

  getSecurityScoreColor: (score) => {
    if (score >= 90) return '#10B981'; // green
    if (score >= 70) return '#F59E0B'; // yellow
    if (score >= 50) return '#F97316'; // orange
    return '#EF4444'; // red
  },

  getStatusColor: (status) => {
    const colors = {
      active: '#10B981',
      inactive: '#6B7280',
      error: '#EF4444',
      warning: '#F59E0B'
    };
    return colors[status] || '#6B7280';
  }
};