// summarizationService.js
// Service for handling all meeting summarization API calls

class SummarizationService {
  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    this.apiPrefix = '/api/v1/summarization';
  }

  // Get authentication token from localStorage or session
  getAuthToken() {
    return localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
  }

  // Create headers with authentication
  getHeaders(isFormData = false) {
    const headers = {};
    
    const token = this.getAuthToken();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    if (!isFormData) {
      headers['Content-Type'] = 'application/json';
    }

    return headers;
  }

  // Handle API response errors
  async handleResponse(response) {
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const errorMessage = errorData.detail || errorData.message || `HTTP ${response.status}: ${response.statusText}`;
      throw new Error(errorMessage);
    }
    return response.json();
  }

  // Upload audio file
  async uploadAudio(audioFile, meetingId = null) {
    try {
      const formData = new FormData();
      formData.append('audio_file', audioFile);
      
      if (meetingId) {
        formData.append('meeting_id', meetingId);
      }

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/upload-audio`, {
        method: 'POST',
        headers: this.getHeaders(true), // isFormData = true
        body: formData
      });

      const result = await this.handleResponse(response);
      console.log('Audio upload successful:', result);
      return result;
    } catch (error) {
      console.error('Error uploading audio:', error);
      throw new Error(`Failed to upload audio: ${error.message}`);
    }
  }

  // Analyze meeting audio
  async analyzeMeeting(analysisRequest) {
    try {
      const requestBody = {
        audio_file_path: analysisRequest.audio_file_path,
        meeting_context: analysisRequest.meeting_context || null,
        analysis_type: analysisRequest.analysis_type || 'post_meeting',
        include_sentiment: analysisRequest.include_sentiment ?? true,
        include_speakers: analysisRequest.include_speakers ?? true
      };

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/analyze-meeting`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(requestBody)
      });

      const result = await this.handleResponse(response);
      console.log('Meeting analysis successful:', result);
      return result;
    } catch (error) {
      console.error('Error analyzing meeting:', error);
      throw new Error(`Failed to analyze meeting: ${error.message}`);
    }
  }

  // Generate summary from content
  async generateSummary(summaryRequest) {
    try {
      const requestBody = {
        content: summaryRequest.content,
        summary_type: summaryRequest.summary_type || 'brief',
        meeting_id: summaryRequest.meeting_id || null,
        include_action_items: summaryRequest.include_action_items ?? true,
        max_length: summaryRequest.max_length || null,
        focus_areas: summaryRequest.focus_areas || null
      };

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/summarize`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(requestBody)
      });

      const result = await this.handleResponse(response);
      console.log('Summary generation successful:', result);
      return result;
    } catch (error) {
      console.error('Error generating summary:', error);
      throw new Error(`Failed to generate summary: ${error.message}`);
    }
  }

  // Get meeting summary by ID
  async getMeetingSummary(meetingId) {
    try {
      const response = await fetch(`${this.baseURL}${this.apiPrefix}/meeting/${meetingId}/summary`, {
        method: 'GET',
        headers: this.getHeaders()
      });

      const result = await this.handleResponse(response);
      console.log('Retrieved meeting summary:', result);
      return result;
    } catch (error) {
      console.error('Error getting meeting summary:', error);
      throw new Error(`Failed to get meeting summary: ${error.message}`);
    }
  }

  // Get all summaries for current user
  async getUserSummaries(limit = 10, offset = 0) {
    try {
      const queryParams = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString()
      });

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/user/summaries?${queryParams}`, {
        method: 'GET',
        headers: this.getHeaders()
      });

      const result = await this.handleResponse(response);
      console.log('Retrieved user summaries:', result);
      return result;
    } catch (error) {
      console.error('Error getting user summaries:', error);
      throw new Error(`Failed to get user summaries: ${error.message}`);
    }
  }

  // Delete meeting summary
  async deleteSummary(meetingId) {
    try {
      const response = await fetch(`${this.baseURL}${this.apiPrefix}/meeting/${meetingId}/summary`, {
        method: 'DELETE',
        headers: this.getHeaders()
      });

      const result = await this.handleResponse(response);
      console.log('Summary deleted successfully:', result);
      return result;
    } catch (error) {
      console.error('Error deleting summary:', error);
      throw new Error(`Failed to delete summary: ${error.message}`);
    }
  }

  // Real-time meeting analysis
  async realTimeAnalysis(analysisRequest) {
    try {
      const requestBody = {
        audio_file_path: analysisRequest.audio_chunk_path,
        meeting_context: analysisRequest.meeting_context || null,
        analysis_type: 'real_time',
        include_sentiment: true,
        include_speakers: true
      };

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/real-time-analysis`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(requestBody)
      });

      const result = await this.handleResponse(response);
      console.log('Real-time analysis successful:', result);
      return result;
    } catch (error) {
      console.error('Error in real-time analysis:', error);
      throw new Error(`Failed to perform real-time analysis: ${error.message}`);
    }
  }

  // Update existing summary
  async updateSummary(summaryId, updateData) {
    try {
      const requestBody = {
        summary_text: updateData.summary_text || null,
        key_points: updateData.key_points || null,
        action_items: updateData.action_items || null,
        next_steps: updateData.next_steps || null
      };

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/summary/${summaryId}`, {
        method: 'PATCH',
        headers: this.getHeaders(),
        body: JSON.stringify(requestBody)
      });

      const result = await this.handleResponse(response);
      console.log('Summary updated successfully:', result);
      return result;
    } catch (error) {
      console.error('Error updating summary:', error);
      throw new Error(`Failed to update summary: ${error.message}`);
    }
  }

  // Batch process multiple meetings
  async batchProcessMeetings(meetingIds, summaryType = 'brief', includeComparative = false) {
    try {
      const requestBody = {
        meeting_ids: meetingIds,
        summary_type: summaryType,
        include_comparative_analysis: includeComparative
      };

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/batch-process`, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(requestBody)
      });

      const result = await this.handleResponse(response);
      console.log('Batch processing successful:', result);
      return result;
    } catch (error) {
      console.error('Error in batch processing:', error);
      throw new Error(`Failed to batch process meetings: ${error.message}`);
    }
  }

  // Get analysis status for long-running operations
  async getAnalysisStatus(analysisId) {
    try {
      const response = await fetch(`${this.baseURL}${this.apiPrefix}/analysis/${analysisId}/status`, {
        method: 'GET',
        headers: this.getHeaders()
      });

      const result = await this.handleResponse(response);
      return result;
    } catch (error) {
      console.error('Error getting analysis status:', error);
      throw new Error(`Failed to get analysis status: ${error.message}`);
    }
  }

  // Export summary to different formats
  async exportSummary(summaryId, format = 'pdf') {
    try {
      const response = await fetch(`${this.baseURL}${this.apiPrefix}/summary/${summaryId}/export?format=${format}`, {
        method: 'GET',
        headers: this.getHeaders()
      });

      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }

      // Handle file download
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = `meeting-summary-${summaryId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);

      console.log('Summary exported successfully');
      return true;
    } catch (error) {
      console.error('Error exporting summary:', error);
      throw new Error(`Failed to export summary: ${error.message}`);
    }
  }

  // Search summaries
  async searchSummaries(query, filters = {}) {
    try {
      const queryParams = new URLSearchParams({
        q: query,
        ...filters
      });

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/search?${queryParams}`, {
        method: 'GET',
        headers: this.getHeaders()
      });

      const result = await this.handleResponse(response);
      console.log('Search results:', result);
      return result;
    } catch (error) {
      console.error('Error searching summaries:', error);
      throw new Error(`Failed to search summaries: ${error.message}`);
    }
  }

  // Get meeting analytics/insights
  async getMeetingAnalytics(dateRange = null, filters = {}) {
    try {
      const queryParams = new URLSearchParams(filters);
      if (dateRange) {
        queryParams.append('start_date', dateRange.start);
        queryParams.append('end_date', dateRange.end);
      }

      const response = await fetch(`${this.baseURL}${this.apiPrefix}/analytics?${queryParams}`, {
        method: 'GET',
        headers: this.getHeaders()
      });

      const result = await this.handleResponse(response);
      console.log('Analytics retrieved:', result);
      return result;
    } catch (error) {
      console.error('Error getting analytics:', error);
      throw new Error(`Failed to get meeting analytics: ${error.message}`);
    }
  }

  // WebSocket connection for real-time updates
  connectWebSocket(meetingId, onMessage, onError = null, onClose = null) {
    try {
      const token = this.getAuthToken();
      const wsUrl = `${this.baseURL.replace('http', 'ws')}/ws/meeting/${meetingId}?token=${token}`;
      
      const ws = new WebSocket(wsUrl);
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (onError) onError(error);
      };

      ws.onclose = (event) => {
        console.log('WebSocket connection closed:', event);
        if (onClose) onClose(event);
      };

      ws.onopen = () => {
        console.log('WebSocket connection established');
      };

      return ws;
    } catch (error) {
      console.error('Error establishing WebSocket connection:', error);
      throw new Error(`Failed to connect WebSocket: ${error.message}`);
    }
  }

  // Utility method to validate audio file
  validateAudioFile(file) {
    const validTypes = [
      'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave',
      'audio/m4a', 'audio/aac', 'audio/ogg', 'audio/webm'
    ];
    
    const maxSize = 100 * 1024 * 1024; // 100MB
    const maxDuration = 3 * 60 * 60; // 3 hours in seconds

    if (!validTypes.includes(file.type)) {
      throw new Error('Invalid file type. Please upload an audio file (MP3, WAV, M4A, etc.)');
    }

    if (file.size > maxSize) {
      throw new Error('File too large. Maximum size is 100MB');
    }

    return true;
  }

  // Utility method to format duration
  formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }

  // Utility method to format file size
  formatFileSize(bytes) {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${sizes[i]}`;
  }

  // Method to check API health
  async checkHealth() {
    try {
      const response = await fetch(`${this.baseURL}/health`, {
        method: 'GET'
      });

      return response.ok;
    } catch (error) {
      console.error('API health check failed:', error);
      return false;
    }
  }

  // Method to get API version and status
  async getApiInfo() {
    try {
      const response = await fetch(`${this.baseURL}/info`, {
        method: 'GET'
      });

      const result = await this.handleResponse(response);
      return result;
    } catch (error) {
      console.error('Error getting API info:', error);
      throw new Error(`Failed to get API info: ${error.message}`);
    }
  }
}

// Create and export singleton instance
export const summarizationService = new SummarizationService();

// Export the class as well for custom instances
export { SummarizationService };

// Export utility functions
export const audioUtils = {
  validateFile: (file) => summarizationService.validateAudioFile(file),
  formatDuration: (seconds) => summarizationService.formatDuration(seconds),
  formatFileSize: (bytes) => summarizationService.formatFileSize(bytes)
};

// Constants for the frontend
export const SUMMARY_TYPES = {
  BRIEF: 'brief',
  DETAILED: 'detailed',
  ACTION_ITEMS: 'action_items',
  KEY_POINTS: 'key_points',
  FULL_TRANSCRIPT: 'full_transcript'
};

export const ANALYSIS_TYPES = {
  REAL_TIME: 'real_time',
  POST_MEETING: 'post_meeting',
  CONTINUOUS: 'continuous'
};

export const PRIORITY_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high'
};

export const STATUS_TYPES = {
  PENDING: 'pending',
  IN_PROGRESS: 'in_progress',
  COMPLETED: 'completed',
  CANCELLED: 'cancelled'
};