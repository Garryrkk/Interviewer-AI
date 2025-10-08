/**
 * AudioService - Frontend service for handling audio transcription API calls
 * Handles all communication with the FastAPI backend
 */

/////PREOP FETURES////
import { transcribeAudio } from '../../services/voiceService';
import { PreOpVoice } from '../../services/voiceService';

PreOpVoice();

// For demo purposes, we'll include the AudioService inline

transcribeAudio(audioBlob)
  .then(response => {
    console.log("API Response:", response);
  })
  .catch(err => {
    console.error("API Error:", err);
  });

class AudioService {
  constructor() {
    // Base URL for the API - adjust this based on your backend configuration
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    this.apiPrefix = '/api/v1/audio';
    
    // Default timeout for requests (30 seconds for transcription, 10s for others)
    this.defaultTimeout = 10000;
    this.transcriptionTimeout = 30000;
  }

  /**
   * Generic fetch wrapper with error handling
   */
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

    // Add timeout handling
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

  /**
   * Audio Calibration
   */
  async calibrateAudio({ duration = 3, sampleRate = 16000, channels = 1 }) {
    try {
      console.log('Starting audio calibration...', { duration, sampleRate, channels });
      
      const response = await this.makeRequest('/calibrate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          duration,
          sample_rate: sampleRate,
          channels
        }),
        timeout: (duration + 5) * 1000 // Add buffer time
      });

      console.log('Calibration completed:', response);
      return response;
    } catch (error) {
      console.error('Calibration failed:', error);
      throw new Error(`Calibration failed: ${error.message}`);
    }
  }

  /**
   * Test Audio Recording
   */
  async testRecording({ duration = 5, sampleRate = 16000, channels = 1, applyCalibration = true }) {
    try {
      console.log('Starting test recording...', { duration, sampleRate, channels, applyCalibration });
      
      const response = await this.makeRequest('/test-record', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          duration,
          sample_rate: sampleRate,
          channels,
          apply_calibration: applyCalibration
        }),
        timeout: (duration + 10) * 1000 // Add buffer time
      });

      console.log('Test recording completed:', response);
      return response;
    } catch (error) {
      console.error('Test recording failed:', error);
      throw new Error(`Test recording failed: ${error.message}`);
    }
  }

  /**
   * Transcribe Audio File
   */
  async transcribeAudio(audioFile, language = 'auto', modelSize = 'base') {
    try {
      console.log('Starting transcription...', {
        filename: audioFile.name,
        size: audioFile.size,
        type: audioFile.type,
        language,
        modelSize
      });

      // Validate file type
      if (!audioFile.type.startsWith('audio/')) {
        throw new Error('Please select a valid audio file');
      }

      // Validate file size (50MB limit)
      const maxSize = 50 * 1024 * 1024; // 50MB
      if (audioFile.size > maxSize) {
        throw new Error('Audio file is too large. Maximum size is 50MB.');
      }

      const formData = new FormData();
      formData.append('audio_file', audioFile);
      
      // Add query parameters for language and model size
      const queryParams = new URLSearchParams({
        language,
        model_size: modelSize
      });

      const response = await this.makeRequest(`/transcribe?${queryParams}`, {
        method: 'POST',
        body: formData,
        timeout: this.transcriptionTimeout,
        headers: {
          // Don't set Content-Type, let the browser set it for FormData
        }
      });

      console.log('Transcription completed:', response);
      return response;
    } catch (error) {
      console.error('Transcription failed:', error);
      throw new Error(`Transcription failed: ${error.message}`);
    }
  }

  /**
   * Transcribe Latest Test Recording
   */
  async transcribeTestRecording() {
    try {
      console.log('Transcribing latest test recording...');
      
      const response = await this.makeRequest('/transcribe-test', {
        method: 'POST',
        timeout: this.transcriptionTimeout
      });

      console.log('Test recording transcription completed:', response);
      return response;
    } catch (error) {
      console.error('Test recording transcription failed:', error);
      throw new Error(`Test recording transcription failed: ${error.message}`);
    }
  }

  /**
   * Get Calibration Status
   */
  async getCalibrationStatus() {
    try {
      const response = await this.makeRequest('/calibration-status', {
        method: 'GET'
      });

      console.log('Calibration status retrieved:', response);
      return response;
    } catch (error) {
      console.error('Failed to get calibration status:', error);
      throw new Error(`Failed to get calibration status: ${error.message}`);
    }
  }

  /**
   * Reset Audio Calibration
   */
  async resetCalibration() {
    try {
      console.log('Resetting calibration...');
      
      const response = await this.makeRequest('/reset-calibration', {
        method: 'DELETE'
      });

      console.log('Calibration reset:', response);
      return response;
    } catch (error) {
      console.error('Failed to reset calibration:', error);
      throw new Error(`Failed to reset calibration: ${error.message}`);
    }
  }

  /**
   * Get Supported Audio Formats
   */
  async getSupportedFormats() {
    try {
      const response = await this.makeRequest('/supported-formats', {
        method: 'GET'
      });

      console.log('Supported formats retrieved:', response);
      return response;
    } catch (error) {
      console.error('Failed to get supported formats:', error);
      throw new Error(`Failed to get supported formats: ${error.message}`);
    }
  }

  /**
   * Health Check
   */
  async healthCheck() {
    try {
      const response = await this.makeRequest('/health', {
        method: 'GET',
        timeout: 5000 // Short timeout for health check
      });

      console.log('Health check result:', response);
      return response;
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error(`Health check failed: ${error.message}`);
    }
  }

  /**
   * Get Available Audio Devices (if implemented in backend)
   */
  async getAudioDevices() {
    try {
      const response = await this.makeRequest('/devices', {
        method: 'GET'
      });

      console.log('Audio devices retrieved:', response);
      return response;
    } catch (error) {
      console.error('Failed to get audio devices:', error);
      // Don't throw error as this might not be implemented
      return { devices: [], default_device: -1 };
    }
  }

  /**
   * Set Audio Device (if implemented in backend)
   */
  async setAudioDevice(deviceId) {
    try {
      console.log('Setting audio device:', deviceId);
      
      const response = await this.makeRequest('/set-device', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ device_id: deviceId })
      });

      console.log('Audio device set:', response);
      return response;
    } catch (error) {
      console.error('Failed to set audio device:', error);
      throw new Error(`Failed to set audio device: ${error.message}`);
    }
  }

  /**
   * Utility Methods
   */

  /**
   * Validate audio file before upload
   */
  validateAudioFile(file) {
    const errors = [];
    
    // Check if file exists
    if (!file) {
      errors.push('No file selected');
      return errors;
    }

    // Check file type
    const supportedTypes = [
      'audio/wav',
      'audio/mp3',
      'audio/mpeg',
      'audio/flac',
      'audio/ogg',
      'audio/m4a',
      'audio/aac',
      'audio/webm'
    ];

    if (!supportedTypes.some(type => file.type === type || file.name.toLowerCase().endsWith(type.split('/')[1]))) {
      errors.push('Unsupported file format. Please use WAV, MP3, FLAC, OGG, M4A, or AAC files.');
    }

    // Check file size (50MB limit)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
      errors.push('File is too large. Maximum size is 50MB.');
    }

    // Check minimum file size
    const minSize = 1024; // 1KB
    if (file.size < minSize) {
      errors.push('File is too small. Please select a valid audio file.');
    }

    return errors;
  }

  /**
   * Format file size for display
   */
  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  /**
   * Format duration for display
   */
  formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }

  /**
   * Get confidence level description
   */
  getConfidenceDescription(confidence) {
    if (confidence >= 0.9) return 'Excellent';
    if (confidence >= 0.8) return 'Very Good';
    if (confidence >= 0.7) return 'Good';
    if (confidence >= 0.6) return 'Fair';
    if (confidence >= 0.5) return 'Poor';
    return 'Very Poor';
  }

  /**
   * Get quality score description
   */
  getQualityDescription(score) {
    if (score >= 0.9) return 'Excellent Audio Quality';
    if (score >= 0.8) return 'Very Good Audio Quality';
    if (score >= 0.7) return 'Good Audio Quality';
    if (score >= 0.6) return 'Fair Audio Quality';
    if (score >= 0.5) return 'Poor Audio Quality';
    return 'Very Poor Audio Quality';
  }

  /**
   * Download transcription result as text file
   */
  downloadTranscription(transcriptionResult, filename = 'transcription.txt') {
    let content = '';
    
    // Add header information
    content += `Audio Transcription Result\n`;
    content += `Generated: ${new Date().toLocaleString()}\n`;
    content += `Language: ${transcriptionResult.language}\n`;
    content += `Confidence: ${(transcriptionResult.confidence * 100).toFixed(1)}%\n`;
    content += `Duration: ${this.formatDuration(transcriptionResult.duration)}\n`;
    content += `Word Count: ${transcriptionResult.word_count}\n`;
    content += `\n${'='.repeat(50)}\n\n`;
    
    // Add main transcription text
    content += `TRANSCRIPTION:\n${transcriptionResult.text}\n\n`;
    
    // Add segments if available
    if (transcriptionResult.segments && transcriptionResult.segments.length > 0) {
      content += `${'='.repeat(50)}\n`;
      content += `DETAILED SEGMENTS:\n\n`;
      
      transcriptionResult.segments.forEach((segment, index) => {
        const start = this.formatDuration(segment.start);
        const end = this.formatDuration(segment.end);
        const confidence = (segment.confidence * 100).toFixed(1);
        
        content += `[${index + 1}] ${start} - ${end} (${confidence}% confidence)\n`;
        content += `${segment.text}\n\n`;
      });
    }
    
    // Create and download file
    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }

  /**
   * Export transcription as JSON
   */
  exportTranscriptionAsJSON(transcriptionResult, filename = 'transcription.json') {
    const exportData = {
      ...transcriptionResult,
      exported_at: new Date().toISOString(),
      export_version: '1.0'
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }

  /**
   * Copy transcription text to clipboard
   */
  async copyToClipboard(text) {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        const success = document.execCommand('copy');
        document.body.removeChild(textArea);
        return success;
      }
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      return false;
    }
  }
}

// Export for use in React components
export { AudioService };