import { VoiceProcessing } from '../../services/voiceService';
import { VoiceRecognition } from '../../services/voiceService';

VoiceProcessing();
VoiceRecognition();

// API Configuration
const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  ENDPOINTS: {
    SESSION_START: '/api/voice/session/start',
    SESSION_END: '/api/voice/session',
    MIC_STATUS: '/api/voice/microphone/status',
    MIC_TOGGLE: '/api/voice/microphone/toggle',
    DEVICES_LIST: '/api/voice/devices/list',
    DEVICE_SELECT: '/api/voice/device/select',
    AUDIO_PROCESS: '/api/voice/audio/process',
    TRANSCRIBE: '/api/voice/transcribe',
    AI_RESPOND: '/api/voice/ai/respond',
    VOICE_ANALYZE: '/api/voice/analyze/voice',
    SIMPLIFY: '/api/voice/simplify'
  },
  TIMEOUT: 30000,
  MAX_RETRIES: 3
};

// Response format enums
export const RESPONSE_FORMATS = {
  SUMMARY: 'summary',
  KEY_INSIGHTS: 'key_insights',
  DETAILED: 'detailed',
  BULLET_POINTS: 'bullet_points'
};

export const SIMPLIFICATION_LEVELS = {
  BASIC: 'basic',
  INTERMEDIATE: 'intermediate',
  ADVANCED: 'advanced'
};

export const AUDIO_QUALITY = {
  EXCELLENT: 'excellent',
  GOOD: 'good',
  FAIR: 'fair',
  POOR: 'poor'
};

/**
 * Custom error class for API operations
 */
export class VoiceAPIError extends Error {
  constructor(message, code = null, details = null) {
    super(message);
    this.name = 'VoiceAPIError';
    this.code = code;
    this.details = details;
  }
}

/**
 * HTTP client with retry logic and error handling
 */
class APIClient {
  constructor(baseUrl = API_CONFIG.BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    };

    let lastError;
    
    for (let attempt = 1; attempt <= API_CONFIG.MAX_RETRIES; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), config.timeout);
        
        const response = await fetch(url, {
          ...config,
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
          const errorData = await response.json().catch(() => null);
          throw new VoiceAPIError(
            errorData?.detail || `HTTP ${response.status}: ${response.statusText}`,
            response.status,
            errorData
          );
        }
        
        return await response.json();
        
      } catch (error) {
        lastError = error;
        
        if (error.name === 'AbortError') {
          throw new VoiceAPIError('Request timeout', 'TIMEOUT');
        }
        
        if (attempt === API_CONFIG.MAX_RETRIES) {
          throw error instanceof VoiceAPIError ? error : 
            new VoiceAPIError(`Network error: ${error.message}`, 'NETWORK_ERROR');
        }
        
        // Exponential backoff
        await new Promise(resolve => 
          setTimeout(resolve, Math.pow(2, attempt) * 1000)
        );
      }
    }
    
    throw lastError;
  }

  async get(endpoint, params = {}) {
    const searchParams = new URLSearchParams(params);
    const url = searchParams.toString() ? `${endpoint}?${searchParams}` : endpoint;
    return this.request(url, { method: 'GET' });
  }

  async post(endpoint, data = null, options = {}) {
    const config = { method: 'POST', ...options };
    
    if (data && !(data instanceof FormData)) {
      config.body = JSON.stringify(data);
    } else if (data instanceof FormData) {
      config.body = data;
      // Remove Content-Type header for FormData (browser will set it with boundary)
      delete config.headers?.['Content-Type'];
    }
    
    return this.request(endpoint, config);
  }

  async delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' });
  }
}

/**
 * Main Voice Processing API Service
 */
export class VoiceProcessingAPI {
  constructor(baseUrl) {
    this.client = new APIClient(baseUrl);
    this.sessionId = null;
    this.userId = null;
  }

  /**
   * Session Management
   */
  async startSession(userId, meetingId = null) {
    try {
      const response = await this.client.post(API_CONFIG.ENDPOINTS.SESSION_START, {
        user_id: userId,
        meeting_id: meetingId
      });
      
      if (response.success) {
        this.sessionId = response.session_id;
        this.userId = userId;
      }
      
      return response;
    } catch (error) {
      throw new VoiceAPIError(`Failed to start session: ${error.message}`);
    }
  }

  async endSession(sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      const response = await this.client.delete(`${API_CONFIG.ENDPOINTS.SESSION_END}/${sessionId}`);
      
      if (response.success) {
        this.sessionId = null;
        this.userId = null;
      }
      
      return response;
    } catch (error) {
      throw new VoiceAPIError(`Failed to end session: ${error.message}`);
    }
  }

  async getSessionStatus(sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.get(`${API_CONFIG.ENDPOINTS.SESSION_END}/${sessionId}/status`);
    } catch (error) {
      throw new VoiceAPIError(`Failed to get session status: ${error.message}`);
    }
  }

  /**
   * Microphone Management
   */
  async checkMicrophoneStatus(sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.get(`${API_CONFIG.ENDPOINTS.MIC_STATUS}/${sessionId}`);
    } catch (error) {
      throw new VoiceAPIError(`Failed to check microphone status: ${error.message}`);
    }
  }

  async toggleMicrophone(turnOn, deviceId = null, sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.post(API_CONFIG.ENDPOINTS.MIC_TOGGLE, {
        session_id: sessionId,
        turn_on: turnOn,
        device_id: deviceId
      });
    } catch (error) {
      throw new VoiceAPIError(`Failed to toggle microphone: ${error.message}`);
    }
  }

  /**
   * Audio Device Management
   */  
  async getAudioDevices(sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.get(`${API_CONFIG.ENDPOINTS.DEVICES_LIST}/${sessionId}`);
    } catch (error) {
      throw new VoiceAPIError(`Failed to get audio devices: ${error.message}`);
    }
  }

  async selectAudioDevice(deviceId, deviceName = null, sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.post(API_CONFIG.ENDPOINTS.DEVICE_SELECT, {
        session_id: sessionId,
        device_id: deviceId,
        device_name: deviceName
      });
    } catch (error) {
      throw new VoiceAPIError(`Failed to select audio device: ${error.message}`);
    }
  }

  /**
   * Audio Processing
   */
  async processAudio(audioFile, sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      const formData = new FormData();
      formData.append('audio_file', audioFile);
      
      return await this.client.post(
        `${API_CONFIG.ENDPOINTS.AUDIO_PROCESS}?session_id=${sessionId}`,
        formData
      );
    } catch (error) {
      throw new VoiceAPIError(`Failed to process audio: ${error.message}`);
    }
  }

  async transcribeAudio(audioData, format = 'wav', sampleRate = 16000, sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.post(API_CONFIG.ENDPOINTS.TRANSCRIBE, {
        session_id: sessionId,
        audio_data: audioData,
        format: format,
        sample_rate: sampleRate
      });
    } catch (error) {
      throw new VoiceAPIError(`Failed to transcribe audio: ${error.message}`);
    }
  }

  /**
   * AI Response Generation
   */
  async getAIResponse(question, responseFormat = RESPONSE_FORMATS.SUMMARY, context = null, maxLength = 500, sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.post(API_CONFIG.ENDPOINTS.AI_RESPOND, {
        session_id: sessionId,
        question: question,
        response_format: responseFormat,
        context: context,
        max_length: maxLength
      });
    } catch (error) {
      throw new VoiceAPIError(`Failed to get AI response: ${error.message}`);
    }
  }

  async simplifyResponse(originalResponse, simplificationLevel = SIMPLIFICATION_LEVELS.BASIC, targetAudience = null, sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.post(API_CONFIG.ENDPOINTS.SIMPLIFY, {
        session_id: sessionId,
        original_response: originalResponse,
        simplification_level: simplificationLevel,
        target_audience: targetAudience
      });
    } catch (error) {
      throw new VoiceAPIError(`Failed to simplify response: ${error.message}`);
    }
  }

  /**
   * Voice Analysis
   */
  async analyzeVoice(audioData, sessionId = this.sessionId) {
    if (!sessionId) throw new VoiceAPIError('No active session');
    
    try {
      return await this.client.post(API_CONFIG.ENDPOINTS.VOICE_ANALYZE, {
        session_id: sessionId,
        audio_data: audioData
      });
    } catch (error) {
      throw new VoiceAPIError(`Failed to analyze voice: ${error.message}`);
    }
  }
}

/**
 * Audio Recording Utilities
 */
export class AudioRecorder {
  constructor() {
    this.mediaRecorder = null;
    this.stream = null;
    this.chunks = [];
    this.isRecording = false;
  }

  async startRecording(options = {}) {
    const defaultOptions = {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 44100
      }
    };

    const constraints = { ...defaultOptions, ...options };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      this.chunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.chunks.push(event.data);
        }
      };

      this.mediaRecorder.onstart = () => {
        this.isRecording = true;
      };

      this.mediaRecorder.onstop = () => {
        this.isRecording = false;
      };

      this.mediaRecorder.start(1000); // Collect data every second
      return true;
    } catch (error) {
      throw new VoiceAPIError(`Failed to start recording: ${error.message}`);
    }
  }

  stopRecording() {
    return new Promise((resolve) => {
      if (!this.mediaRecorder || !this.isRecording) {
        resolve(null);
        return;
      }

      this.mediaRecorder.onstop = () => {
        this.isRecording = false;
        
        // Stop all tracks
        if (this.stream) {
          this.stream.getTracks().forEach(track => track.stop());
          this.stream = null;
        }

        // Create blob from chunks
        const audioBlob = new Blob(this.chunks, { type: 'audio/webm' });
        this.chunks = [];
        
        resolve(audioBlob);
      };

      this.mediaRecorder.stop();
    });
  }

  getRecordingDuration() {
    // Approximate duration based on chunks (not precise)
    return this.chunks.length;
  }

  isCurrentlyRecording() {
    return this.isRecording;
  }

  async getAvailableDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter(device => device.kind === 'audioinput');
    } catch (error) {
      throw new VoiceAPIError(`Failed to enumerate devices: ${error.message}`);
    }
  }

  cleanup() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.mediaRecorder = null;
    this.chunks = [];
    this.isRecording = false;
  }
}

/**
 * Audio Format Conversion Utilities
 */
export class AudioConverter {
  static async blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  static async blobToArrayBuffer(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsArrayBuffer(blob);
    });
  }

  static createWaveFile(audioBuffer, sampleRate = 44100) {
    const length = audioBuffer.length;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, audioBuffer[i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
  }
}

/**
 * Session Management Utilities
 */
export class SessionManager {
  constructor() {
    this.sessions = new Map();
    this.activeSessionId = null;
  }

  createSession(userId, sessionData = {}) {
    const sessionId = this.generateSessionId();
    const session = {
      id: sessionId,
      userId: userId,
      createdAt: new Date(),
      lastActivity: new Date(),
      isActive: true,
      ...sessionData
    };

    this.sessions.set(sessionId, session);
    this.activeSessionId = sessionId;
    
    return session;
  }

  getSession(sessionId = this.activeSessionId) {
    return this.sessions.get(sessionId);
  }

  updateSessionActivity(sessionId = this.activeSessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.lastActivity = new Date();
    }
  }

  endSession(sessionId = this.activeSessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      session.isActive = false;
      session.endedAt = new Date();
    }
    
    if (sessionId === this.activeSessionId) {
      this.activeSessionId = null;
    }
  }

  cleanupExpiredSessions(maxAge = 7200000) { // 2 hours
    const now = new Date();
    for (const [sessionId, session] of this.sessions) {
      const age = now - session.lastActivity;
      if (age > maxAge) {
        this.sessions.delete(sessionId);
      }
    }
  }

  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Event System for Real-time Updates
 */
export class VoiceEventEmitter {
  constructor() {
    this.events = {};
  }

  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }

  off(event, callback) {
    if (!this.events[event]) return;
    
    this.events[event] = this.events[event].filter(cb => cb !== callback);
  }

  emit(event, data) {
    if (!this.events[event]) return;
    
    this.events[event].forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Event callback error for ${event}:`, error);
      }
    });
  }

  removeAllListeners(event) {
    if (event) {
      delete this.events[event];
    } else {
      this.events = {};
    }
  }
}

/**
 * Voice Processing State Manager
 */
export class VoiceStateManager {
  constructor() {
    this.state = {
      session: null,
      microphone: {
        enabled: false,
        device: null,
        permissions: 'unknown'
      },
      recording: {
        active: false,
        duration: 0,
        startTime: null
      },
      processing: {
        transcribing: false,
        analyzing: false,
        generating: false
      },
      results: {
        transcription: null,
        aiResponse: null,
        voiceAnalysis: null
      },
      errors: []
    };

    this.listeners = [];
    this.eventEmitter = new VoiceEventEmitter();
  }

  getState() {
    return { ...this.state };
  }

  updateState(updates) {
    const previousState = { ...this.state };
    this.state = { ...this.state, ...updates };
    
    this.eventEmitter.emit('stateChange', {
      previous: previousState,
      current: this.state,
      changes: updates
    });

    this.notifyListeners();
  }

  subscribe(listener) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  notifyListeners() {
    this.listeners.forEach(listener => {
      try {
        listener(this.state);
      } catch (error) {
        console.error('State listener error:', error);
      }
    });
  }

  addError(error) {
    const errorObj = {
      id: Date.now(),
      message: error.message || error,
      timestamp: new Date(),
      code: error.code || 'UNKNOWN'
    };

    this.updateState({
      errors: [...this.state.errors, errorObj]
    });
  }

  removeError(errorId) {
    this.updateState({
      errors: this.state.errors.filter(err => err.id !== errorId)
    });
  }

  clearErrors() {
    this.updateState({ errors: [] });
  }
}

/**
 * Performance Monitoring Utilities
 */
export class PerformanceMonitor {
  constructor() {
    this.metrics = {};
    this.startTimes = {};
  }

  startTimer(operation) {
    this.startTimes[operation] = performance.now();
  }

  endTimer(operation) {
    if (!this.startTimes[operation]) return null;
    
    const duration = performance.now() - this.startTimes[operation];
    delete this.startTimes[operation];
    
    if (!this.metrics[operation]) {
      this.metrics[operation] = [];
    }
    
    this.metrics[operation].push({
      duration,
      timestamp: new Date()
    });

    return duration;
  }

  getMetrics(operation) {
    return this.metrics[operation] || [];
  }

  getAverageTime(operation) {
    const metrics = this.getMetrics(operation);
    if (metrics.length === 0) return 0;
    
    const total = metrics.reduce((sum, metric) => sum + metric.duration, 0);
    return total / metrics.length;
  }

  clearMetrics(operation) {
    if (operation) {
      delete this.metrics[operation];
    } else {
      this.metrics = {};
    }
  }
}

/**
 * Browser Compatibility Utilities
 */
export class BrowserCompatibility {
  static checkWebRTCSupport() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  static checkMediaRecorderSupport() {
    return typeof MediaRecorder !== 'undefined';
  }

  static checkAudioContextSupport() {
    return !!(window.AudioContext || window.webkitAudioContext);
  }

  static getCompatibilityReport() {
    return {
      webrtc: this.checkWebRTCSupport(),
      mediaRecorder: this.checkMediaRecorderSupport(),
      audioContext: this.checkAudioContextSupport(),
      fileApi: typeof FileReader !== 'undefined',
      webSockets: typeof WebSocket !== 'undefined'
    };
  }

  static getSupportedMimeTypes() {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/wav',
      'audio/ogg;codecs=opus'
    ];

    return types.filter(type => {
      try {
        return MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(type);
      } catch (e) {
        return false;
      }
    });
  }
}

/**
 * Export all utilities as a combined service
 */
export class VoiceProcessingService {
  constructor(config = {}) {
    this.api = new VoiceProcessingAPI(config.baseUrl);
    this.recorder = new AudioRecorder();
    this.sessionManager = new SessionManager();
    this.stateManager = new VoiceStateManager();
    this.performanceMonitor = new PerformanceMonitor();
    this.eventEmitter = new VoiceEventEmitter();
    
    this.config = {
      maxRecordingDuration: 300000, // 5 minutes
      autoCleanupInterval: 300000, // 5 minutes
      retryAttempts: 3,
      ...config
    };

    this.startAutoCleanup();
  }

  // Convenience methods that combine multiple operations
  async initializeSession(userId, meetingId = null) {
    try {
      this.performanceMonitor.startTimer('session_init');
      
      const response = await this.api.startSession(userId, meetingId);
      if (response.success) {
        const session = this.sessionManager.createSession(userId, {
          apiSessionId: response.session_id,
          meetingId
        });
        
        this.stateManager.updateState({ session });
        this.eventEmitter.emit('sessionStarted', session);
      }
      
      this.performanceMonitor.endTimer('session_init');
      return response;
    } catch (error) {
      this.stateManager.addError(error);
      throw error;
    }
  }

  async startRecordingWithProcessing(options = {}) {
    try {
      this.performanceMonitor.startTimer('recording_start');
      
      // Start recording
      await this.recorder.startRecording(options);
      
      this.stateManager.updateState({
        recording: {
          active: true,
          startTime: new Date(),
          duration: 0
        }
      });

      // Set up duration tracking
      const durationInterval = setInterval(() => {
        if (!this.recorder.isCurrentlyRecording()) {
          clearInterval(durationInterval);
          return;
        }

        const duration = Date.now() - this.stateManager.state.recording.startTime;
        this.stateManager.updateState({
          recording: {
            ...this.stateManager.state.recording,
            duration
          }
        });

        // Auto-stop if max duration reached
        if (duration >= this.config.maxRecordingDuration) {
          this.stopRecordingWithProcessing();
          clearInterval(durationInterval);
        }
      }, 1000);

      this.performanceMonitor.endTimer('recording_start');
      this.eventEmitter.emit('recordingStarted');
      
    } catch (error) {
      this.stateManager.addError(error);
      throw error;
    }
  }

  async stopRecordingWithProcessing() {
    try {
      this.performanceMonitor.startTimer('recording_stop');
      
      const audioBlob = await this.recorder.stopRecording();
      
      this.stateManager.updateState({
        recording: {
          active: false,
          duration: 0,
          startTime: null
        },
        processing: { transcribing: true }
      });

      if (audioBlob) {
        // Convert to File object
        const audioFile = new File([audioBlob], `recording_${Date.now()}.webm`, {
          type: audioBlob.type
        });

        // Process the audio
        const result = await this.processCompleteAudioWorkflow(audioFile);
        this.performanceMonitor.endTimer('recording_stop');
        
        return result;
      }

      this.performanceMonitor.endTimer('recording_stop');
      return null;
      
    } catch (error) {
      this.stateManager.addError(error);
      this.stateManager.updateState({
        recording: { active: false, duration: 0, startTime: null },
        processing: { transcribing: false, analyzing: false, generating: false }
      });
      throw error;
    }
  }

  async processCompleteAudioWorkflow(audioFile) {
    try {
      // Step 1: Process audio (transcribe + basic analysis)
      this.stateManager.updateState({
        processing: { transcribing: true, analyzing: false, generating: false }
      });

      const processResult = await this.api.processAudio(audioFile);
      
      if (!processResult.success) {
        throw new VoiceAPIError('Audio processing failed');
      }

      // Step 2: Get AI response
      this.stateManager.updateState({
        processing: { transcribing: false, analyzing: false, generating: true }
      });

      const aiResult = await this.api.getAIResponse(
        processResult.transcription,
        this.stateManager.state.responseFormat || RESPONSE_FORMATS.SUMMARY
      );

      // Step 3: Voice analysis
      this.stateManager.updateState({
        processing: { transcribing: false, analyzing: true, generating: false }
      });

      const audioBase64 = await AudioConverter.blobToBase64(audioFile);
      const voiceAnalysis = await this.api.analyzeVoice(audioBase64);

      // Update state with results
      this.stateManager.updateState({
        processing: { transcribing: false, analyzing: false, generating: false },
        results: {
          transcription: processResult.transcription,
          aiResponse: aiResult.success ? aiResult.response : null,
          voiceAnalysis: voiceAnalysis.success ? voiceAnalysis : null
        }
      });

      const results = {
        transcription: processResult,
        aiResponse: aiResult,
        voiceAnalysis: voiceAnalysis
      };

      this.eventEmitter.emit('processingComplete', results);
      return results;

    } catch (error) {
      this.stateManager.updateState({
        processing: { transcribing: false, analyzing: false, generating: false }
      });
      this.stateManager.addError(error);
      throw error;
    }
  }

  startAutoCleanup() {
    setInterval(() => {
      this.sessionManager.cleanupExpiredSessions();
      this.performanceMonitor.clearMetrics();
    }, this.config.autoCleanupInterval);
  }

  async cleanup() {
    try {
      if (this.stateManager.state.session?.apiSessionId) {
        await this.api.endSession(this.stateManager.state.session.apiSessionId);
      }
      
      this.recorder.cleanup();
      this.eventEmitter.removeAllListeners();
      this.sessionManager.sessions.clear();
      
    } catch (error) {
      console.error('Cleanup error:', error);
    }
  }
}

// Default export
export default VoiceProcessingService;