// meetingAnalysisService.js
// Complete API service layer for Meeting Analysis Dashboard
import { QuickRespond } from '../../services/aiService';

QuickRespond("This is my user prompt")
  .then(response => {
    console.log("API Response:", response);
  })
  .catch(err => {
    console.error("API Error:", err);
  });
  
class MeetingAnalysisService {
  constructor(baseUrl = 'http://127.0.0.1:8000/api/quick-respond') {
    this.baseUrl = baseUrl;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
    };
  }

  // Generic request method with error handling
  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      headers: {
        ...this.defaultHeaders,
        ...options.headers
      },
      ...options
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
      }

      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      } else {
        return await response.text();
      }
    } catch (error) {
      console.error(`API request failed for ${url}:`, error);
      throw error;
    }
  }

  // File upload method
  async uploadFile(endpoint, file, additionalData = {}) {
    const formData = new FormData();
    formData.append('screenshot', file);
    
    Object.keys(additionalData).forEach(key => {
      if (additionalData[key] !== null && additionalData[key] !== undefined) {
        formData.append(key, additionalData[key]);
      }
    });

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
    }
    
    return await response.json();
  }

  // Streaming response handler
  async streamAnalysis(file, additionalData = {}) {
    const formData = new FormData();
    formData.append('screenshot', file);
    
    Object.keys(additionalData).forEach(key => {
      if (additionalData[key] !== null && additionalData[key] !== undefined) {
        formData.append(key, additionalData[key]);
      }
    });

    const response = await fetch(`${this.baseUrl}/analyze-screenshot/stream`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
    }

    return response;
  }

  // ==================== ANALYSIS ENDPOINTS ====================

  // Single screenshot analysis
  async analyzeScreenshot(file, meetingContext = null, audioTranscript = null) {
    return await this.uploadFile('/analyze-screenshot', file, {
      meeting_context: meetingContext,
      audio_transcript: audioTranscript
    });
  }

  // Streaming screenshot analysis
  async streamScreenshotAnalysis(file, meetingContext = null, audioTranscript = null) {
    return await this.streamAnalysis(file, {
      meeting_context: meetingContext,
      audio_transcript: audioTranscript
    });
  }

  // Batch analysis
  async batchAnalyzeScreenshots(files, meetingContext = null) {
    const formData = new FormData();
    
    Array.from(files).forEach(file => {
      formData.append('screenshots', file);
    });
    
    if (meetingContext) {
      formData.append('meeting_context', meetingContext);
    }

    const response = await fetch(`${this.baseUrl}/batch-analyze`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
    }

    return await response.json();
  }

  // Quick respond analysis
  async quickRespond(requestData) {
    return await this.request('/quick-respond', {
      method: 'POST',
      body: JSON.stringify(requestData)
    });
  }

  // Simplify analysis
  async simplifyAnalysis(originalAnalysis, sessionId = null, simplificationLevel = 1, focusAreas = null) {
    return await this.request('/simplify', {
      method: 'POST',
      body: JSON.stringify({
        original_analysis: originalAnalysis,
        session_id: sessionId,
        simplification_level: simplificationLevel,
        focus_areas: focusAreas
      })
    });
  }

  // Advanced analysis
  async advancedAnalysis(requestData) {
    return await this.request('/advanced', {
      method: 'POST',
      body: JSON.stringify(requestData)
    });
  }

  // Batch analysis with detailed options
  async batchAnalysis(requestData) {
    return await this.request('/batch', {
      method: 'POST',
      body: JSON.stringify(requestData)
    });
  }

  // ==================== CONTEXT MANAGEMENT ====================

  // Update meeting context
  async updateMeetingContext(context) {
    return await this.request('/context/update', {
      method: 'POST',
      body: JSON.stringify(context)
    });
  }

  // Clear meeting context
  async clearMeetingContext() {
    return await this.request('/context/clear', {
      method: 'DELETE'
    });
  }

  // ==================== CONFIGURATION ENDPOINTS ====================

  // Ollama Configuration
  async createOllamaConfig(config) {
    return await this.request('/ollama-config', {
      method: 'POST',
      body: JSON.stringify(config)
    });
  }

  async getAllOllamaConfigs() {
    return await this.request('/ollama-config');
  }

  // Quick Respond Configuration
  async createQuickRespondConfig(config) {
    return await this.request('/quick-respond-config', {
      method: 'POST',
      body: JSON.stringify(config)
    });
  }

  async getAllQuickRespondConfigs() {
    return await this.request('/quick-respond-config');
  }

  // Model Prompts
  async createModelPrompt(prompt) {
    return await this.request('/model-prompts', {
      method: 'POST',
      body: JSON.stringify(prompt)
    });
  }

  async getAllModelPrompts() {
    return await this.request('/model-prompts');
  }

  // ==================== MEETING STATUS CRUD ====================

  async createMeetingStatus(meetingStatus) {
    return await this.request('/meeting_status/', {
      method: 'POST',
      body: JSON.stringify(meetingStatus)
    });
  }

  async getMeetingStatus(id) {
    return await this.request(`/meeting_status/${id}`);
  }

  async listMeetingStatuses() {
    return await this.request('/meeting_status/');
  }

  async updateMeetingStatus(id, meetingStatus) {
    return await this.request(`/meeting_status/${id}`, {
      method: 'PUT',
      body: JSON.stringify(meetingStatus)
    });
  }

  async deleteMeetingStatus(id) {
    return await this.request(`/meeting_status/${id}`, {
      method: 'DELETE'
    });
  }

  // ==================== PARTICIPANT INFO CRUD ====================

  async createParticipantInfo(participant) {
    return await this.request('/participant_info/', {
      method: 'POST',
      body: JSON.stringify(participant)
    });
  }

  async getParticipantInfo(id) {
    return await this.request(`/participant_info/${id}`);
  }

  async listParticipants() {
    return await this.request('/participant_info/');
  }

  async updateParticipantInfo(id, participant) {
    return await this.request(`/participant_info/${id}`, {
      method: 'PUT',
      body: JSON.stringify(participant)
    });
  }

  async deleteParticipantInfo(id) {
    return await this.request(`/participant_info/${id}`, {
      method: 'DELETE'
    });
  }

  // ==================== SCREEN CONTENT CRUD ====================

  async createScreenContent(screen) {
    return await this.request('/screen_content/', {
      method: 'POST',
      body: JSON.stringify(screen)
    });
  }

  async getScreenContent(id) {
    return await this.request(`/screen_content/${id}`);
  }

  async listScreenContent() {
    return await this.request('/screen_content/');
  }

  async updateScreenContent(id, screen) {
    return await this.request(`/screen_content/${id}`, {
      method: 'PUT',
      body: JSON.stringify(screen)
    });
  }

  async deleteScreenContent(id) {
    return await this.request(`/screen_content/${id}`, {
      method: 'DELETE'
    });
  }

  // ==================== MEETING METRICS CRUD ====================

  async createMeetingMetrics(metrics) {
    return await this.request('/meeting_metrics/', {
      method: 'POST',
      body: JSON.stringify(metrics)
    });
  }

  async getMeetingMetrics(id) {
    return await this.request(`/meeting_metrics/${id}`);
  }

  async listMeetingMetrics() {
    return await this.request('/meeting_metrics/');
  }

  async updateMeetingMetrics(id, metrics) {
    return await this.request(`/meeting_metrics/${id}`, {
      method: 'PUT',
      body: JSON.stringify(metrics)
    });
  }

  async deleteMeetingMetrics(id) {
    return await this.request(`/meeting_metrics/${id}`, {
      method: 'DELETE'
    });
  }

  // ==================== UTILITY ENDPOINTS ====================

  // Health check
  async checkHealth() {
    return await this.request('/health');
  }

  // Get urgency levels
  async getUrgencyLevels() {
    return await this.request('/urgency-levels');
  }

  // Get specific urgency level
  async getUrgencyLevel(level) {
    return await this.request(`/urgency-levels/${level}`);
  }

  // Get paginated items
  async getPaginatedItems(page = 1, pageSize = 10) {
    return await this.request(`/items?page=${page}&page_size=${pageSize}`);
  }

  // ==================== WEBHOOK ENDPOINTS ====================

  async createWebhookEvent(event) {
    return await this.request('/webhook-event', {
      method: 'POST',
      body: JSON.stringify(event)
    });
  }

  // ==================== HELPER METHODS ====================

  // Process streaming response
  async processStreamingResponse(response, onChunk, onComplete, onError) {
    try {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep the incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (onChunk) onChunk(data);
            } catch (e) {
              console.warn('Failed to parse streaming data:', line);
            }
          }
        }
      }

      if (onComplete) onComplete();
    } catch (error) {
      console.error('Streaming error:', error);
      if (onError) onError(error);
      throw error;
    }
  }

  // Validate file before upload
  validateFile(file, maxSizeMB = 10, allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']) {
    const errors = [];

    if (!file) {
      errors.push('No file selected');
      return { valid: false, errors };
    }

    // Check file size
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      errors.push(`File size must be less than ${maxSizeMB}MB`);
    }

    // Check file type
    if (!allowedTypes.includes(file.type)) {
      errors.push(`File type must be one of: ${allowedTypes.join(', ')}`);
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  // Convert file to base64
  async fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = error => reject(error);
    });
  }

  // Format analysis response for display
  formatAnalysisResponse(response) {
    if (!response) return null;

    return {
      ...response,
      timestamp: new Date(response.timestamp).toLocaleString(),
      confidence_percentage: Math.round(response.confidence_score * 100),
      insights_count: response.key_insights?.length || 0,
      urgency_counts: this.countInsightsByUrgency(response.key_insights || [])
    };
  }

  // Count insights by urgency level
  countInsightsByUrgency(insights) {
    return insights.reduce((counts, insight) => {
      counts[insight.urgency] = (counts[insight.urgency] || 0) + 1;
      return counts;
    }, {});
  }

  // Generate session ID
  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Format error for display
  formatError(error) {
    if (typeof error === 'string') {
      return { message: error, timestamp: new Date().toISOString() };
    }
    
    return {
      message: error.message || 'An unknown error occurred',
      timestamp: new Date().toISOString(),
      stack: error.stack
    };
  }

  // Retry mechanism for failed requests
  async retryRequest(requestFn, maxRetries = 3, delay = 1000) {
    let lastError;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await requestFn();
      } catch (error) {
        lastError = error;
        if (i < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
        }
      }
    }
    
    throw lastError;
  }

  // Cache management
  setupCache(ttlMinutes = 30) {
    this.cache = new Map();
    this.cacheTTL = ttlMinutes * 60 * 1000;
  }

  getCached(key) {
    if (!this.cache) return null;
    
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
      return cached.data;
    }
    
    this.cache.delete(key);
    return null;
  }

  setCache(key, data) {
    if (!this.cache) return;
    
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  clearCache() {
    if (this.cache) {
      this.cache.clear();
    }
  }
}

// Export singleton instance and class
const meetingAnalysisService = new MeetingAnalysisService();

// Export both the instance and the class for flexibility
export default meetingAnalysisService;
export { MeetingAnalysisService };

// Usage examples:
/*
// Import the service
import meetingAnalysisService from './meetingAnalysisService.js';

// Basic screenshot analysis
const file = document.getElementById('fileInput').files[0];
try {
  const result = await meetingAnalysisService.analyzeScreenshot(file, 'Board meeting', 'Recent discussion about Q4 targets');
  console.log('Analysis:', result);
} catch (error) {
  console.error('Analysis failed:', error);
}

// Streaming analysis
try {
  const stream = await meetingAnalysisService.streamScreenshotAnalysis(file);
  await meetingAnalysisService.processStreamingResponse(
    stream,
    (data) => console.log('Chunk:', data),
    () => console.log('Stream complete'),
    (error) => console.error('Stream error:', error)
  );
} catch (error) {
  console.error('Streaming failed:', error);
}

// Batch analysis
const files = document.getElementById('batchInput').files;
try {
  const results = await meetingAnalysisService.batchAnalyzeScreenshots(files, 'Weekly standup meeting');
  console.log('Batch results:', results);
} catch (error) {
  console.error('Batch analysis failed:', error);
}

// Health check
try {
  const health = await meetingAnalysisService.checkHealth();
  console.log('Service health:', health);
} catch (error) {
  console.error('Health check failed:', error);
}

// Configuration management
try {
  const configs = await meetingAnalysisService.getAllOllamaConfigs();
  console.log('Ollama configs:', configs);
} catch (error) {
  console.error('Config fetch failed:', error);
}
*/