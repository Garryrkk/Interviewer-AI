/**
 * Meeting Summarization Service
 * Handles all API calls, data processing, and business logic
 * for the Meeting Summarization Application
 */
import { analyzeImage } from "../../services/imageService";
import { CameraCapture } from "../../services/imageService";

CameraCapture()
  .then((imageBlob) => {
    console.log("Captured image Blob:", imageBlob);

    // Analyze the captured image
    return analyzeImage(imageBlob);
  })
  .then((analysis) => {
    console.log("Image Analysis:", analysis);
  })
  .catch((err) => {
    console.error("Camera or Analysis Error:", err);
  });
  
class MeetingSummarizationService {
  constructor() {
    this.API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '/api/v1/summarization';
    this.authToken = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.recordingStartTime = null;
    
    // Initialize authentication
    this.initAuth();
  }

  // ===============================
  // Authentication & Setup
  // ===============================

  initAuth() {
    this.authToken = localStorage.getItem('authToken');
    if (!this.authToken) {
      console.warn('No auth token found. Some features may not work.');
    }
  }

  setAuthToken(token) {
    this.authToken = token;
    localStorage.setItem('authToken', token);
  }

  getAuthHeaders() {
    return {
      'Authorization': `Bearer ${this.authToken}`,
      'Content-Type': 'application/json',
    };
  }

  // ===============================
  // Generic API Handler
  // ===============================

  async apiCall(endpoint, options = {}) {
    const url = `${this.API_BASE_URL}${endpoint}`;
    
    const config = {
      headers: this.getAuthHeaders(),
      ...options,
    };

    // Remove Content-Type for FormData
    if (options.body instanceof FormData) {
      delete config.headers['Content-Type'];
    }

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ 
          detail: `HTTP ${response.status}: ${response.statusText}` 
        }));
        throw new Error(errorData.detail || `Request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API call failed for ${endpoint}:`, error);
      throw new Error(`API Error: ${error.message}`);
    }
  }

  // ===============================
  // File Upload & Management
  // ===============================

  validateAudioFile(file) {
    const validTypes = [
      'audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/ogg', 
      'audio/m4a', 'audio/aac', 'audio/flac'
    ];
    
    const maxSize = 100 * 1024 * 1024; // 100MB
    
    if (!validTypes.includes(file.type)) {
      throw new Error('Invalid file type. Please upload an audio file.');
    }
    
    if (file.size > maxSize) {
      throw new Error('File too large. Maximum size is 100MB.');
    }
    
    return true;
  }

  async uploadAudioFile(file, meetingId = null, onProgress = null) {
    this.validateAudioFile(file);
    
    const formData = new FormData();
    formData.append('audio_file', file);
    
    if (meetingId) {
      formData.append('meeting_id', meetingId);
    } else {
      formData.append('meeting_id', `meeting_${Date.now()}`);
    }

    // Create XMLHttpRequest for progress tracking
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const percentComplete = (event.loaded / event.total) * 100;
          onProgress(percentComplete);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          reject(new Error(`Upload failed: ${xhr.statusText}`));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'));
      });

      xhr.open('POST', `${this.API_BASE_URL}/upload-audio`);
      xhr.setRequestHeader('Authorization', `Bearer ${this.authToken}`);
      xhr.send(formData);
    });
  }

  // ===============================
  // Audio Recording
  // ===============================

  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        } 
      });
      
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      this.audioChunks = [];
      this.recordingStartTime = Date.now();

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
      };

      this.mediaRecorder.start(1000); // Collect data every second
      return true;
    } catch (error) {
      throw new Error(`Recording failed: ${error.message}`);
    }
  }

  async stopRecording() {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
        reject(new Error('No active recording to stop'));
        return;
      }

      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const duration = (Date.now() - this.recordingStartTime) / 1000;
        
        const audioFile = new File([audioBlob], `recording_${Date.now()}.webm`, { 
          type: 'audio/webm' 
        });
        
        resolve({
          file: audioFile,
          duration: duration,
          size: audioBlob.size
        });
      };

      this.mediaRecorder.stop();
    });
  }

  isRecording() {
    return this.mediaRecorder && this.mediaRecorder.state === 'recording';
  }

  // ===============================
  // Meeting Analysis
  // ===============================

  async analyzeMeetingAudio(params) {
    const {
      audioFilePath,
      meetingContext = '',
      analysisType = 'post_meeting',
      includeSentiment = true,
      includeSpeakers = true
    } = params;

    const requestData = {
      audio_file_path: audioFilePath,
      meeting_context: meetingContext,
      analysis_type: analysisType,
      include_sentiment: includeSentiment,
      include_speakers: includeSpeakers,
    };

    return await this.apiCall('/analyze-meeting', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  }

  async performRealTimeAnalysis(params) {
    const {
      audioFilePath,
      meetingContext = ''
    } = params;

    const requestData = {
      audio_file_path: audioFilePath,
      meeting_context: meetingContext,
    };

    return await this.apiCall('/real-time-analysis', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  }

  // ===============================
  // Summary Generation
  // ===============================

  async generateSummary(params) {
    const {
      content,
      summaryType = 'brief',
      meetingId = null,
      includeActionItems = true,
      maxLength = null,
      focusAreas = null
    } = params;

    const requestData = {
      content,
      summary_type: summaryType,
      meeting_id: meetingId,
      include_action_items: includeActionItems,
    };

    if (maxLength) requestData.max_length = maxLength;
    if (focusAreas) requestData.focus_areas = focusAreas;

    return await this.apiCall('/summarize', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  }

  // ===============================
  // Summary Management
  // ===============================

  async getUserSummaries(limit = 10, offset = 0) {
    return await this.apiCall(`/user/summaries?limit=${limit}&offset=${offset}`, {
      method: 'GET',
    });
  }

  async getMeetingSummary(meetingId) {
    return await this.apiCall(`/meeting/${meetingId}/summary`, {
      method: 'GET',
    });
  }

  async deleteMeetingSummary(meetingId) {
    return await this.apiCall(`/meeting/${meetingId}/summary`, {
      method: 'DELETE',
    });
  }

  async updateSummary(summaryId, updateData) {
    return await this.apiCall(`/summary/${summaryId}`, {
      method: 'PUT',
      body: JSON.stringify(updateData),
    });
  }

  // ===============================
  // Batch Operations
  // ===============================

  async batchSummarize(meetingIds, summaryType = 'brief', includeComparative = false) {
    const requestData = {
      meeting_ids: meetingIds,
      summary_type: summaryType,
      include_comparative_analysis: includeComparative,
    };

    return await this.apiCall('/batch-summarize', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  }

  // ===============================
  // Data Processing Utilities
  // ===============================

  formatDuration(seconds) {
    if (!seconds || seconds < 0) return '0:00';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  }

  formatFileSize(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`;
  }

  calculateCompressionRatio(originalSize, compressedSize) {
    if (originalSize === 0) return 0;
    return Math.round((1 - compressedSize / originalSize) * 100);
  }

  // ===============================
  // Analytics & Insights
  // ===============================

  analyzeActionItemTrends(summaries) {
    const trends = {
      totalActionItems: 0,
      priorityDistribution: { high: 0, medium: 0, low: 0 },
      averagePerMeeting: 0,
      completionRate: 0,
    };

    let totalActionItems = 0;
    let completedItems = 0;

    summaries.forEach(summary => {
      const actionItems = summary.action_items || [];
      totalActionItems += actionItems.length;

      actionItems.forEach(item => {
        trends.priorityDistribution[item.priority] = 
          (trends.priorityDistribution[item.priority] || 0) + 1;
        
        if (item.status === 'completed') {
          completedItems++;
        }
      });
    });

    trends.totalActionItems = totalActionItems;
    trends.averagePerMeeting = summaries.length > 0 ? 
      Math.round((totalActionItems / summaries.length) * 100) / 100 : 0;
    trends.completionRate = totalActionItems > 0 ? 
      Math.round((completedItems / totalActionItems) * 100) : 0;

    return trends;
  }

  analyzeSentimentTrends(summaries) {
    const sentiments = { positive: 0, neutral: 0, negative: 0 };
    let totalMeetings = 0;

    summaries.forEach(summary => {
      if (summary.sentiment_analysis) {
        const sentiment = summary.sentiment_analysis.overall_sentiment;
        if (sentiments.hasOwnProperty(sentiment)) {
          sentiments[sentiment]++;
          totalMeetings++;
        }
      }
    });

    return {
      distribution: sentiments,
      totalAnalyzed: totalMeetings,
      positivePercentage: totalMeetings > 0 ? 
        Math.round((sentiments.positive / totalMeetings) * 100) : 0,
    };
  }

  calculateMeetingEffectiveness(summaries) {
    let totalScore = 0;
    let scoredMeetings = 0;

    summaries.forEach(summary => {
      if (summary.meeting_effectiveness_score) {
        totalScore += summary.meeting_effectiveness_score;
        scoredMeetings++;
      }
    });

    return {
      averageScore: scoredMeetings > 0 ? 
        Math.round((totalScore / scoredMeetings) * 100) / 100 : 0,
      totalMeetings: scoredMeetings,
      recommendation: this.getEffectivenessRecommendation(totalScore / scoredMeetings)
    };
  }

  getEffectivenessRecommendation(averageScore) {
    if (averageScore >= 8) {
      return "Excellent meeting effectiveness! Keep up the great work.";
    } else if (averageScore >= 6) {
      return "Good meeting effectiveness. Consider focusing on clearer action items.";
    } else if (averageScore >= 4) {
      return "Room for improvement. Try setting clearer agendas and objectives.";
    } else {
      return "Meetings need significant improvement. Consider restructuring your meeting format.";
    }
  }

  // ===============================
  // Export & Import Functions
  // ===============================

  exportSummaryToJson(summary) {
    const exportData = {
      ...summary,
      exported_at: new Date().toISOString(),
      export_version: '1.0'
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `meeting_summary_${summary.meeting_id || 'export'}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  }

  exportSummaryToPdf(summary) {
    // This would require a PDF generation library like jsPDF
    console.log('PDF export not implemented. Use exportSummaryToJson instead.');
    throw new Error('PDF export requires additional PDF library integration');
  }

  // ===============================
  // Local Storage Management
  // ===============================

  saveDraftSummary(summaryData) {
    const drafts = this.getDraftSummaries();
    const draftId = `draft_${Date.now()}`;
    
    drafts[draftId] = {
      ...summaryData,
      draft_id: draftId,
      saved_at: new Date().toISOString(),
    };
    
    localStorage.setItem('meeting_summary_drafts', JSON.stringify(drafts));
    return draftId;
  }

  getDraftSummaries() {
    const stored = localStorage.getItem('meeting_summary_drafts');
    return stored ? JSON.parse(stored) : {};
  }

  deleteDraftSummary(draftId) {
    const drafts = this.getDraftSummaries();
    delete drafts[draftId];
    localStorage.setItem('meeting_summary_drafts', JSON.stringify(drafts));
  }

  // ===============================
  // Error Handling & Retry Logic
  // ===============================

  async retryApiCall(apiFunction, maxRetries = 3, delay = 1000) {
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await apiFunction();
      } catch (error) {
        if (attempt === maxRetries) {
          throw error;
        }
        
        console.warn(`API call failed (attempt ${attempt}/${maxRetries}):`, error.message);
        await this.delay(delay * attempt); // Exponential backoff
      }
    }
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // ===============================
  // Configuration & Settings
  // ===============================

  getDefaultConfig() {
    return {
      summaryType: 'brief',
      analysisType: 'post_meeting',
      includeActionItems: true,
      includeSentiment: true,
      includeSpeakers: true,
      autoSave: true,
      maxFileSize: 100 * 1024 * 1024, // 100MB
      supportedFormats: ['mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac'],
    };
  }

  updateConfig(newConfig) {
    const currentConfig = this.getConfig();
    const updatedConfig = { ...currentConfig, ...newConfig };
    localStorage.setItem('meeting_app_config', JSON.stringify(updatedConfig));
    return updatedConfig;
  }

  getConfig() {
    const stored = localStorage.getItem('meeting_app_config');
    return stored ? { ...this.getDefaultConfig(), ...JSON.parse(stored) } : this.getDefaultConfig();
  }

  // ===============================
  // Health Check & Status
  // ===============================

  async checkApiHealth() {
    try {
      const response = await fetch(`${this.API_BASE_URL}/health`, {
        method: 'GET',
        headers: { 'Authorization': `Bearer ${this.authToken}` },
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  getConnectionStatus() {
    return {
      online: navigator.onLine,
      apiBaseUrl: this.API_BASE_URL,
      hasAuthToken: !!this.authToken,
      browserSupportsRecording: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
    };
  }
}

// ===============================
// Utility Functions (Standalone)
// ===============================

export const audioUtils = {
  // Convert audio file to different format if needed
  async convertAudioFormat(file, targetFormat = 'wav') {
    // This is a placeholder - would need actual audio conversion library
    console.warn('Audio conversion not implemented - returning original file');
    return file;
  },

  // Extract audio metadata
  async getAudioMetadata(file) {
    return new Promise((resolve) => {
      const audio = new Audio(URL.createObjectURL(file));
      
      audio.addEventListener('loadedmetadata', () => {
        resolve({
          duration: audio.duration,
          hasAudio: !isNaN(audio.duration),
        });
        URL.revokeObjectURL(audio.src);
      });
      
      audio.addEventListener('error', () => {
        resolve({ duration: 0, hasAudio: false });
        URL.revokeObjectURL(audio.src);
      });
    });
  },

  // Check browser audio support
  checkAudioSupport() {
    const audio = document.createElement('audio');
    return {
      mp3: !!(audio.canPlayType && audio.canPlayType('audio/mpeg;').replace(/no/, '')),
      wav: !!(audio.canPlayType && audio.canPlayType('audio/wav;').replace(/no/, '')),
      ogg: !!(audio.canPlayType && audio.canPlayType('audio/ogg;').replace(/no/, '')),
      webm: !!(audio.canPlayType && audio.canPlayType('audio/webm;').replace(/no/, '')),
    };
  }
};

// ===============================
// Event Emitter for Real-time Updates
// ===============================

class EventEmitter {
  constructor() {
    this.events = {};
  }

  on(event, callback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }

  emit(event, data) {
    if (this.events[event]) {
      this.events[event].forEach(callback => callback(data));
    }
  }

  off(event, callback) {
    if (this.events[event]) {
      this.events[event] = this.events[event].filter(cb => cb !== callback);
    }
  }
}

// ===============================
// Export Service Instance
// ===============================

const meetingService = new MeetingSummarizationService();
meetingService.eventEmitter = new EventEmitter();

// Add global event listeners
window.addEventListener('online', () => {
  meetingService.eventEmitter.emit('connectionChange', { online: true });
});

window.addEventListener('offline', () => {
  meetingService.eventEmitter.emit('connectionChange', { online: false });
});

export default meetingService;
export { MeetingSummarizationService, audioUtils, EventEmitter };