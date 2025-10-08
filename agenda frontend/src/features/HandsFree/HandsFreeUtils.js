// handsFreeService.js - Service layer for Hands-Free Interview System
import { HandsFreeMode } from '../../services/aiService';

HandsFreeMode(context)
  .then(response => {
    console.log("API Response:", response);
  })
  .catch(err => {
    console.error("API Error:", err);
  });

class HandsFreeInterviewService {
  constructor(baseUrl = 'http://127.0.0.1:8000/ping') {
    this.baseUrl = baseUrl;
    this.audioWs = null;
    this.videoWs = null;
    this.sessionId = null;
    this.mediaRecorder = null;
    this.audioStream = null;
    this.videoStream = null;
    this.eventListeners = new Map();
  }

  // Event handling system
  on(event, callback) {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event).push(callback);
  }

  emit(event, data) {
    if (this.eventListeners.has(event)) {
      this.eventListeners.get(event).forEach(callback => callback(data));
    }
  }

  off(event, callback) {
    if (this.eventListeners.has(event)) {
      const callbacks = this.eventListeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  // API Methods
  async startSession(config = {}) {
    const defaultConfig = {
      user_id: `user_${Date.now()}`,
      default_mic_id: '0',
      interview_type: 'general',
      company_info: '',
      job_role: '',
      auto_start: true
    };

    const sessionConfig = { ...defaultConfig, ...config };

    try {
      const response = await fetch(`${this.baseUrl}/session/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sessionConfig)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.sessionId = data.session_id;
      
      this.emit('sessionStarted', data);
      return data;
    } catch (error) {
      console.error('Failed to start session:', error);
      this.emit('error', { type: 'session_start', error });
      throw error;
    }
  }

  async activateHandsFreeMode() {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/activate`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.emit('handsFreeActivated', data);
      return data;
    } catch (error) {
      console.error('Failed to activate hands-free mode:', error);
      this.emit('error', { type: 'hands_free_activation', error });
      throw error;
    }
  }

  async getSessionStatus() {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/status`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.emit('statusUpdate', data);
      return data;
    } catch (error) {
      console.error('Failed to get session status:', error);
      this.emit('error', { type: 'status_check', error });
      throw error;
    }
  }

  async updateSettings(settings) {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.emit('settingsUpdated', data);
      return data;
    } catch (error) {
      console.error('Failed to update settings:', error);
      this.emit('error', { type: 'settings_update', error });
      throw error;
    }
  }

  async generateManualResponse(question, context = null, responseType = 'key_insights') {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/manual-response`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          context,
          response_type: responseType
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.emit('manualResponse', data);
      return data;
    } catch (error) {
      console.error('Failed to generate manual response:', error);
      this.emit('error', { type: 'manual_response', error });
      throw error;
    }
  }

  async getSessionInsights() {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/insights`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.emit('insightsReceived', data);
      return data;
    } catch (error) {
      console.error('Failed to get session insights:', error);
      this.emit('error', { type: 'insights', error });
      throw error;
    }
  }

  async checkSystemHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/system/health`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.emit('healthCheck', data);
      return data;
    } catch (error) {
      console.error('System health check failed:', error);
      this.emit('error', { type: 'health_check', error });
      throw error;
    }
  }

  async emergencyPause() {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/emergency-pause`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.emit('emergencyPaused', data);
      return data;
    } catch (error) {
      console.error('Failed to emergency pause:', error);
      this.emit('error', { type: 'emergency_pause', error });
      throw error;
    }
  }

  async resumeHandsFree() {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/resume`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      this.emit('handsFreeResumed', data);
      return data;
    } catch (error) {
      console.error('Failed to resume hands-free:', error);
      this.emit('error', { type: 'resume', error });
      throw error;
    }
  }

  async stopSession() {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    try {
      const response = await fetch(`${this.baseUrl}/session/${this.sessionId}/stop`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Cleanup resources
      this.cleanup();
      
      this.emit('sessionStopped', data);
      return data;
    } catch (error) {
      console.error('Failed to stop session:', error);
      this.emit('error', { type: 'session_stop', error });
      throw error;
    }
  }

  // WebSocket Methods
  async initializeAudioWebSocket() {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    return new Promise((resolve, reject) => {
      const wsUrl = this.baseUrl.replace('http', 'ws') + `/session/${this.sessionId}/audio-stream`;
      this.audioWs = new WebSocket(wsUrl);

      this.audioWs.onopen = () => {
        console.log('Audio WebSocket connected');
        this.emit('audioWebSocketConnected');
        resolve();
      };

      this.audioWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleAudioWebSocketMessage(data);
        } catch (error) {
          console.error('Failed to parse audio WebSocket message:', error);
        }
      };

      this.audioWs.onerror = (error) => {
        console.error('Audio WebSocket error:', error);
        this.emit('error', { type: 'audio_websocket', error });
        reject(error);
      };

      this.audioWs.onclose = () => {
        console.log('Audio WebSocket disconnected');
        this.emit('audioWebSocketDisconnected');
      };
    });
  }

  async initializeVideoWebSocket() {
    if (!this.sessionId) {
      throw new Error('No active session');
    }

    return new Promise((resolve, reject) => {
      const wsUrl = this.baseUrl.replace('http', 'ws') + `/session/${this.sessionId}/video-analysis`;
      this.videoWs = new WebSocket(wsUrl);

      this.videoWs.onopen = () => {
        console.log('Video WebSocket connected');
        this.emit('videoWebSocketConnected');
        resolve();
      };

      this.videoWs.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleVideoWebSocketMessage(data);
        } catch (error) {
          console.error('Failed to parse video WebSocket message:', error);
        }
      };

      this.videoWs.onerror = (error) => {
        console.error('Video WebSocket error:', error);
        this.emit('error', { type: 'video_websocket', error });
        reject(error);
      };

      this.videoWs.onclose = () => {
        console.log('Video WebSocket disconnected');
        this.emit('videoWebSocketDisconnected');
      };
    });
  }

  handleAudioWebSocketMessage(data) {
    switch (data.type) {
      case 'automated_response':
        this.emit('automatedResponse', {
          question: data.question,
          response: data.response_text || data.response,
          keyInsights: data.key_insights || [],
          confidenceScore: data.confidence_score || 0,
          timestamp: data.timestamp
        });
        break;
      
      case 'status_update':
        this.emit('audioStatusUpdate', {
          listening: data.listening,
          processing: data.processing,
          audioLevel: data.audio_level || 0
        });
        break;
      
      case 'error':
        this.emit('error', { type: 'audio_processing', message: data.message });
        break;
      
      default:
        console.log('Unknown audio WebSocket message type:', data.type);
    }
  }

  handleVideoWebSocketMessage(data) {
    if (data.facial_analysis) {
      this.emit('facialAnalysis', {
        analysis: data.facial_analysis,
        confidenceTips: data.confidence_tips || { tips: [] },
        overallScore: data.overall_score || 0,
        timestamp: data.timestamp,
        recommendations: data.recommendations || []
      });
    } else if (data.type === 'error') {
      this.emit('error', { type: 'video_analysis', message: data.message });
    }
  }

  // Media Methods
  async startAudioCapture(options = {}) {
    const defaultOptions = {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000
      }
    };

    const constraints = { ...defaultOptions, ...options };

    try {
      this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      // Create MediaRecorder for audio streaming
      this.mediaRecorder = new MediaRecorder(this.audioStream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && this.audioWs?.readyState === WebSocket.OPEN) {
          this.audioWs.send(event.data);
        }
      };

      this.mediaRecorder.onerror = (error) => {
        console.error('MediaRecorder error:', error);
        this.emit('error', { type: 'media_recorder', error });
      };

      // Start recording with small chunks for real-time processing
      this.mediaRecorder.start(100);
      
      this.emit('audioStreamStarted');
      return this.audioStream;
    } catch (error) {
      console.error('Failed to start audio capture:', error);
      this.emit('error', { type: 'audio_capture', error });
      throw error;
    }
  }

  async startVideoCapture(options = {}) {
    const defaultOptions = {
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 }
      }
    };

    const constraints = { ...defaultOptions, ...options };

    try {
      this.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
      this.emit('videoStreamStarted');
      return this.videoStream;
    } catch (error) {
      console.error('Failed to start video capture:', error);
      this.emit('error', { type: 'video_capture', error });
      throw error;
    }
  }

  startVideoFrameCapture(videoElement, intervalMs = 1000) {
    if (!videoElement || !this.videoWs) {
      throw new Error('Video element and WebSocket required');
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const captureFrame = () => {
      try {
        if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
          return; // Video not ready yet
        }

        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        ctx.drawImage(videoElement, 0, 0);

        canvas.toBlob((blob) => {
          if (blob && this.videoWs?.readyState === WebSocket.OPEN) {
            this.videoWs.send(blob);
          }
        }, 'image/jpeg', 0.8);
      } catch (error) {
        console.error('Frame capture error:', error);
      }
    };

    const intervalId = setInterval(captureFrame, intervalMs);
    
    // Store interval ID for cleanup
    this.frameCaptufeInterval = intervalId;
    
    return intervalId;
  }

  stopVideoFrameCapture() {
    if (this.frameCaptufeInterval) {
      clearInterval(this.frameCaptufeInterval);
      this.frameCaptufeInterval = null;
    }
  }

  // Utility Methods
  cleanup() {
    // Close WebSockets
    if (this.audioWs) {
      this.audioWs.close();
      this.audioWs = null;
    }

    if (this.videoWs) {
      this.videoWs.close();
      this.videoWs = null;
    }

    // Stop media recording
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.mediaRecorder = null;
    }

    // Stop media streams
    if (this.audioStream) {
      this.audioStream.getTracks().forEach(track => track.stop());
      this.audioStream = null;
    }

    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
      this.videoStream = null;
    }

    // Clear intervals
    this.stopVideoFrameCapture();

    // Reset session
    this.sessionId = null;

    this.emit('cleanup');
  }

  // Helper methods for device management
  async getAvailableDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      
      const audioInputs = devices.filter(device => device.kind === 'audioinput');
      const videoInputs = devices.filter(device => device.kind === 'videoinput');
      
      return {
        audioInputs,
        videoInputs,
        allDevices: devices
      };
    } catch (error) {
      console.error('Failed to enumerate devices:', error);
      this.emit('error', { type: 'device_enumeration', error });
      throw error;
    }
  }

  async requestPermissions() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: true, 
        video: true 
      });
      
      // Stop the stream immediately as we just needed permissions
      stream.getTracks().forEach(track => track.stop());
      
      return true;
    } catch (error) {
      console.error('Permission denied:', error);
      this.emit('error', { type: 'permissions', error });
      return false;
    }
  }

  // Connection status checks
  isAudioWebSocketConnected() {
    return this.audioWs?.readyState === WebSocket.OPEN;
  }

  isVideoWebSocketConnected() {
    return this.videoWs?.readyState === WebSocket.OPEN;
  }

  isSessionActive() {
    return !!this.sessionId;
  }

  getConnectionStatus() {
    return {
      sessionId: this.sessionId,
      audioWebSocket: this.isAudioWebSocketConnected(),
      videoWebSocket: this.isVideoWebSocketConnected(),
      audioStream: !!this.audioStream,
      videoStream: !!this.videoStream,
      mediaRecorder: this.mediaRecorder?.state || 'inactive'
    };
  }
}

// Export for use in modules
export default HandsFreeInterviewService;

// For non-module usage
if (typeof window !== 'undefined') {
  window.HandsFreeInterviewService = HandsFreeInterviewService;
}