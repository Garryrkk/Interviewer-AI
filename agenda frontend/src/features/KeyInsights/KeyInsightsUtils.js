/**
 * Key Insights Service - JavaScript utility file
 * This file contains all the business logic, API calls, and data processing
 * for the Key Insights Dashboard React component.
 */

class KeyInsightsService {
  constructor(baseUrl = 'http://localhost:8000/api/v1/key-insights') {
    this.baseUrl = baseUrl;
    this.pollingIntervals = new Map();
    this.requestTimeouts = new Map();
  }

  // API Configuration and Headers
  getDefaultHeaders() {
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    };
  }

  // Error handling utility
  async handleResponse(response) {
    if (!response.ok) {
      let errorMessage;
      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.message || `HTTP error! status: ${response.status}`;
      } catch {
        errorMessage = `HTTP error! status: ${response.status}`;
      }
      throw new Error(errorMessage);
    }
    return await response.json();
  }

  // Generate unique meeting ID
  generateMeetingId() {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    return `meeting_${timestamp}_${random}`;
  }

  // Validate file upload
  validateImageFile(file) {
    if (!file) return { valid: false, error: 'No file selected' };
    
    if (!file.type.startsWith('image/')) {
      return { valid: false, error: 'Please select a valid image file' };
    }
    
    // Check file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      return { valid: false, error: 'File size must be less than 10MB' };
    }
    
    return { valid: true };
  }

  // Parse participants string
  parseParticipants(participantsString) {
    if (!participantsString) return [];
    return participantsString
      .split(',')
      .map(p => p.trim())
      .filter(p => p.length > 0);
  }

  // Validate meeting context
  validateMeetingContext(context) {
    if (!context || context.trim().length < 20) {
      return { 
        valid: false, 
        error: 'Meeting context must be at least 20 characters long' 
      };
    }
    return { valid: true };
  }

  // Generate insights from meeting context and optional image
  async generateInsights({
    meetingContext,
    meetingId,
    participants,
    analysisFocus,
    selectedFile,
    onProgress = () => {},
    onStatusUpdate = () => {}
  }) {
    // Validation
    if (!meetingContext && !selectedFile) {
      throw new Error('Please provide either meeting context or upload an image file');
    }

    if (meetingContext) {
      const contextValidation = this.validateMeetingContext(meetingContext);
      if (!contextValidation.valid) {
        throw new Error(contextValidation.error);
      }
    }

    if (selectedFile) {
      const fileValidation = this.validateImageFile(selectedFile);
      if (!fileValidation.valid) {
        throw new Error(fileValidation.error);
      }
    }

    const parsedParticipants = this.parseParticipants(participants);
    
    try {
      let response;
      
      if (selectedFile) {
        // Use FormData for file upload
        const formData = new FormData();
        formData.append('meeting_context', meetingContext || '');
        formData.append('meeting_id', meetingId || '');
        formData.append('participants', JSON.stringify(parsedParticipants));
        formData.append('analysis_focus', analysisFocus || '');
        formData.append('include_visual_analysis', 'true');
        formData.append('image_file', selectedFile);

        response = await fetch(`${this.baseUrl}/analyze`, {
          method: 'POST',
          body: formData,
        });
      } else {
        // Use JSON for text-only requests
        const requestData = {
          meeting_context: meetingContext,
          meeting_id: meetingId || null,
          participants: parsedParticipants,
          analysis_focus: analysisFocus || null,
          include_visual_analysis: false
        };

        response = await fetch(`${this.baseUrl}/analyze`, {
          method: 'POST',
          headers: this.getDefaultHeaders(),
          body: JSON.stringify(requestData),
        });
      }

      const data = await this.handleResponse(response);
      
      // Start status polling if insight ID is available
      if (data.insight_id) {
        this.startStatusPolling(data.insight_id, onStatusUpdate);
      }

      return data;
    } catch (error) {
      throw new Error(`Failed to generate insights: ${error.message}`);
    }
  }

  // Simplify existing insights
  async simplifyInsights({
    originalInsightId,
    originalInsights,
    originalTips,
    simplificationLevel = 'moderate',
    targetAudience = null
  }) {
    if (!originalInsights || originalInsights.length === 0) {
      throw new Error('No insights to simplify');
    }

    const requestData = {
      original_insight_id: originalInsightId,
      original_insights: originalInsights,
      original_tips: originalTips,
      simplification_level: simplificationLevel,
      target_audience: targetAudience
    };

    try {
      const response = await fetch(`${this.baseUrl}/simplify`, {
        method: 'POST',
        headers: this.getDefaultHeaders(),
        body: JSON.stringify(requestData),
      });

      return await this.handleResponse(response);
    } catch (error) {
      throw new Error(`Failed to simplify insights: ${error.message}`);
    }
  }

  // Get analysis status
  async getAnalysisStatus(insightId) {
    try {
      const response = await fetch(`${this.baseUrl}/status/${insightId}`);
      return await this.handleResponse(response);
    } catch (error) {
      throw new Error(`Failed to get analysis status: ${error.message}`);
    }
  }

  // Start polling for analysis status
  startStatusPolling(insightId, onStatusUpdate, pollInterval = 2000, maxDuration = 300000) {
    // Clear existing polling for this insight
    this.stopStatusPolling(insightId);
    
    const intervalId = setInterval(async () => {
      try {
        const status = await this.getAnalysisStatus(insightId);
        onStatusUpdate(status);

        if (status.status === 'completed' || status.status === 'failed') {
          this.stopStatusPolling(insightId);
        }
      } catch (error) {
        console.error('Status polling error:', error);
        this.stopStatusPolling(insightId);
        onStatusUpdate({ 
          insight_id: insightId, 
          status: 'failed', 
          error: error.message 
        });
      }
    }, pollInterval);

    this.pollingIntervals.set(insightId, intervalId);

    // Stop polling after max duration
    const timeoutId = setTimeout(() => {
      this.stopStatusPolling(insightId);
      onStatusUpdate({ 
        insight_id: insightId, 
        status: 'timeout', 
        error: 'Status polling timed out' 
      });
    }, maxDuration);

    this.requestTimeouts.set(insightId, timeoutId);
  }

  // Stop status polling
  stopStatusPolling(insightId) {
    const intervalId = this.pollingIntervals.get(insightId);
    const timeoutId = this.requestTimeouts.get(insightId);
    
    if (intervalId) {
      clearInterval(intervalId);
      this.pollingIntervals.delete(insightId);
    }
    
    if (timeoutId) {
      clearTimeout(timeoutId);
      this.requestTimeouts.delete(insightId);
    }
  }

  // Get insights history for a meeting
  async getInsightsHistory(meetingId) {
    if (!meetingId) {
      throw new Error('Meeting ID is required');
    }

    try {
      const response = await fetch(`${this.baseUrl}/history/${meetingId}`);
      const data = await this.handleResponse(response);
      return data.insights_history || [];
    } catch (error) {
      throw new Error(`Failed to get insights history: ${error.message}`);
    }
  }

  // Delete specific insights
  async deleteInsights(insightId) {
    if (!insightId) {
      throw new Error('Insight ID is required');
    }

    try {
      const response = await fetch(`${this.baseUrl}/insights/${insightId}`, {
        method: 'DELETE',
      });

      return await this.handleResponse(response);
    } catch (error) {
      throw new Error(`Failed to delete insights: ${error.message}`);
    }
  }

  // Batch analysis for multiple meetings
  async batchAnalyzeInsights({
    meetingContexts,
    meetingIds,
    participantsList = null,
    analysisFocusList = null,
    onBatchProgress = () => {}
  }) {
    // Validation
    const validContexts = meetingContexts.filter(c => c && c.trim());
    const validIds = meetingIds.filter(id => id && id.trim());
    
    if (validContexts.length !== validIds.length || validContexts.length === 0) {
      throw new Error('Please provide valid contexts and IDs for batch analysis. Each meeting must have both a context and an ID.');
    }

    // Validate each context
    for (let i = 0; i < validContexts.length; i++) {
      const contextValidation = this.validateMeetingContext(validContexts[i]);
      if (!contextValidation.valid) {
        throw new Error(`Meeting ${i + 1}: ${contextValidation.error}`);
      }
    }

    const requestData = {
      meeting_contexts: validContexts,
      meeting_ids: validIds,
      participants_list: participantsList,
      analysis_focus_list: analysisFocusList,
      batch_id: this.generateMeetingId()
    };

    try {
      const response = await fetch(`${this.baseUrl}/batch-analyze`, {
        method: 'POST',
        headers: this.getDefaultHeaders(),
        body: JSON.stringify(requestData),
      });

      return await this.handleResponse(response);
    } catch (error) {
      throw new Error(`Failed to perform batch analysis: ${error.message}`);
    }
  }

  // Utility methods for data processing
  calculateInsightsMetrics(insights) {
    if (!insights) return null;

    const keyInsights = insights.key_insights || [];
    const situationTips = insights.situation_tips || [];
    
    return {
      totalInsights: keyInsights.length,
      totalTips: situationTips.length,
      averageConfidence: this.calculateAverageConfidence(keyInsights),
      highPriorityCount: keyInsights.filter(i => i.priority === 'high' || i.priority === 'critical').length,
      highActionabilityTips: situationTips.filter(t => t.actionability === 'high').length,
      visualAnalysisIncluded: insights.visual_analysis_included || false,
      participantsCount: insights.participants_analyzed?.length || 0
    };
  }

  calculateAverageConfidence(insights) {
    if (!insights || insights.length === 0) return 0;
    
    const totalConfidence = insights.reduce((sum, insight) => sum + (insight.confidence || 0), 0);
    return (totalConfidence / insights.length * 100).toFixed(1);
  }

  // Format insights for display
  formatInsightsForDisplay(insights) {
    if (!insights) return null;

    return {
      ...insights,
      key_insights: insights.key_insights?.map(insight => ({
        ...insight,
        confidence_percentage: Math.round(insight.confidence * 100),
        category_display: this.formatCategoryDisplay(insight.category),
        priority_display: this.formatPriorityDisplay(insight.priority)
      })) || [],
      situation_tips: insights.situation_tips?.map(tip => ({
        ...tip,
        category_display: this.formatCategoryDisplay(tip.category),
        actionability_display: this.formatActionabilityDisplay(tip.actionability)
      })) || []
    };
  }

  formatCategoryDisplay(category) {
    return category?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'General';
  }

  formatPriorityDisplay(priority) {
    const priorityMap = {
      low: { text: 'Low', color: '#6B7280' },
      medium: { text: 'Medium', color: '#F59E0B' },
      high: { text: 'High', color: '#EF4444' },
      critical: { text: 'Critical', color: '#DC2626' }
    };
    return priorityMap[priority] || priorityMap.medium;
  }

  formatActionabilityDisplay(actionability) {
    const actionabilityMap = {
      low: { text: 'Low', color: '#6B7280' },
      medium: { text: 'Medium', color: '#F59E0B' },
      high: { text: 'High', color: '#10B981' }
    };
    return actionabilityMap[actionability] || actionabilityMap.medium;
  }

  // Export insights data
  exportInsightsToJSON(insights) {
    if (!insights) return null;
    
    const exportData = {
      export_timestamp: new Date().toISOString(),
      insight_id: insights.insight_id,
      meeting_id: insights.meeting_id,
      generated_at: insights.generated_at,
      confidence_score: insights.confidence_score,
      visual_analysis_included: insights.visual_analysis_included,
      participants_analyzed: insights.participants_analyzed,
      key_insights: insights.key_insights,
      situation_tips: insights.situation_tips,
      metrics: this.calculateInsightsMetrics(insights)
    };

    return JSON.stringify(exportData, null, 2);
  }

  // Import insights data
  importInsightsFromJSON(jsonString) {
    try {
      const data = JSON.parse(jsonString);
      
      // Validate required fields
      if (!data.insight_id || !data.key_insights || !data.situation_tips) {
        throw new Error('Invalid insights data format');
      }
      
      return data;
    } catch (error) {
      throw new Error(`Failed to import insights: ${error.message}`);
    }
  }

  // Clean up resources
  cleanup() {
    // Clear all polling intervals
    for (const [insightId] of this.pollingIntervals) {
      this.stopStatusPolling(insightId);
    }
    
    this.pollingIntervals.clear();
    this.requestTimeouts.clear();
  }

  // Health check for the API
  async healthCheck() {
    try {
      const response = await fetch(`${this.baseUrl.replace('/key-insights', '')}/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      
      if (response.ok) {
        return { status: 'healthy', timestamp: new Date().toISOString() };
      } else {
        return { status: 'unhealthy', error: `HTTP ${response.status}` };
      }
    } catch (error) {
      return { status: 'unreachable', error: error.message };
    }
  }

  // Get API configuration info
  getApiInfo() {
    return {
      baseUrl: this.baseUrl,
      endpoints: {
        analyze: `${this.baseUrl}/analyze`,
        simplify: `${this.baseUrl}/simplify`,
        status: `${this.baseUrl}/status/:id`,
        history: `${this.baseUrl}/history/:meeting_id`,
        delete: `${this.baseUrl}/insights/:id`,
        batchAnalyze: `${this.baseUrl}/batch-analyze`
      },
      supportedMethods: ['POST', 'GET', 'DELETE'],
      supportedFileTypes: ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
      maxFileSize: '10MB',
      maxBatchSize: 10
    };
  }
}

// Factory function to create service instance
function createKeyInsightsService(baseUrl) {
  return new KeyInsightsService(baseUrl);
}

// Singleton instance for default usage
const defaultService = new KeyInsightsService();

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
  // CommonJS
  module.exports = {
    KeyInsightsService,
    createKeyInsightsService,
    defaultService
  };
} else if (typeof window !== 'undefined') {
  // Browser globals
  window.KeyInsightsService = KeyInsightsService;
  window.createKeyInsightsService = createKeyInsightsService;
  window.keyInsightsService = defaultService;
}

// ES6 modules export (if supported)
export { KeyInsightsService, createKeyInsightsService, defaultService };