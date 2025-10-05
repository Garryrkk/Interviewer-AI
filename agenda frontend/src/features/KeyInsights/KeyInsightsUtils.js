// keyInsightsAPI.js - Complete API Service for Key Insights Feature

/**
 * Configuration object for API endpoints
 */
import { KeyInsights } from '../../services/aiService';

KeyInsights("This is the text I want insights for")
  .then(response => {
    console.log("API Response:", response);
  })
  .catch(err => {
    console.error("API Error:", err);
  });
  
const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  ENDPOINTS: {
    TYPES: '/api/v1/key-insights/types',
    SAMPLE: '/api/v1/key-insights/sample',
    ANALYZE: '/api/v1/key-insights/analyze',
    SIMPLIFY: '/api/v1/key-insights/simplify',
    STATUS: '/api/v1/key-insights/status',
    HISTORY: '/api/v1/key-insights/history',
    DELETE: '/api/v1/key-insights/insights',
    BATCH_ANALYZE: '/api/v1/key-insights/batch-analyze'
  },
  TIMEOUT: 60000, // 60 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000 // 1 second
};

/**
 * Utility functions for API handling
 */
class APIUtils {
  /**
   * Sleep function for retry delays
   * @param {number} ms - Milliseconds to sleep
   */
  static sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Retry a function with exponential backoff
   * @param {Function} fn - Function to retry
   * @param {number} maxAttempts - Maximum retry attempts
   * @param {number} delay - Initial delay in ms
   */
  static async retry(fn, maxAttempts = API_CONFIG.RETRY_ATTEMPTS, delay = API_CONFIG.RETRY_DELAY) {
    let attempt = 1;
    
    while (attempt <= maxAttempts) {
      try {
        return await fn();
      } catch (error) {
        if (attempt === maxAttempts) {
          throw error;
        }
        
        console.warn(`Attempt ${attempt} failed, retrying in ${delay}ms...`, error.message);
        await APIUtils.sleep(delay);
        delay *= 2; // Exponential backoff
        attempt++;
      }
    }
  }

  /**
   * Validate image file
   * @param {File} file - Image file to validate
   * @returns {boolean} - Whether file is valid
   */
  static validateImageFile(file) {
    if (!file) return false;
    
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    const maxSize = 10 * 1024 * 1024; // 10MB
    
    if (!validTypes.includes(file.type)) {
      throw new Error('Invalid file type. Please upload JPEG, PNG, GIF, or WebP images.');
    }
    
    if (file.size > maxSize) {
      throw new Error('File size too large. Please upload images smaller than 10MB.');
    }
    
    return true;
  }

  /**
   * Format insight types for display
   * @param {string} type - Insight type
   * @returns {string} - Formatted type
   */
  static formatInsightType(type) {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }

  /**
   * Calculate confidence color based on score
   * @param {number} score - Confidence score (0-1)
   * @returns {string} - CSS color class
   */
  static getConfidenceColor(score) {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  }

  /**
   * Format timestamp for display
   * @param {string} timestamp - ISO timestamp
   * @returns {string} - Formatted timestamp
   */
  static formatTimestamp(timestamp) {
    try {
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      }).format(new Date(timestamp));
    } catch (error) {
      return 'Invalid date';
    }
  }
}

/**
 * Main API service class for Key Insights
 */
class KeyInsightsAPIService {
  constructor(baseURL = API_CONFIG.BASE_URL) {
    this.baseURL = baseURL;
    this.requestId = 0;
  }

  /**
   * Generate unique request ID for tracking
   */
  generateRequestId() {
    return `req_${Date.now()}_${++this.requestId}`;
  }

  /**
   * Generic HTTP request method with error handling and retries
   * @param {string} endpoint - API endpoint
   * @param {Object} options - Request options
   * @returns {Promise} - Response data
   */
  async request(endpoint, options = {}) {
    const requestId = this.generateRequestId();
    const url = `${this.baseURL}${endpoint}`;
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId,
      },
      timeout: API_CONFIG.TIMEOUT,
      ...options,
    };

    // Remove Content-Type for FormData
    if (options.body && options.body instanceof FormData) {
      delete config.headers['Content-Type'];
    }

    const makeRequest = async () => {
      console.log(`[${requestId}] Making request to: ${url}`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), config.timeout);

      try {
        const response = await fetch(url, {
          ...config,
          signal: controller.signal
        });

        clearTimeout(timeoutId);

        const data = await response.json();
        
        if (!response.ok) {
          throw new APIError(
            data.detail || `HTTP error! status: ${response.status}`,
            response.status,
            data.error_code || 'UNKNOWN_ERROR',
            requestId
          );
        }
        
        console.log(`[${requestId}] Request successful`);
        return data;
      } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
          throw new APIError('Request timeout', 408, 'TIMEOUT', requestId);
        }
        
        throw error;
      }
    };

    return APIUtils.retry(makeRequest);
  }

  /**
   * Get all available insight types
   * @returns {Promise<Array>} - Array of insight type strings
   */
  async getInsightTypes() {
    return this.request(API_CONFIG.ENDPOINTS.TYPES);
  }

  /**
   * Get a sample insight for testing/demo purposes
   * @returns {Promise<Object>} - Sample insight object
   */
  async getSampleInsight() {
    return this.request(API_CONFIG.ENDPOINTS.SAMPLE);
  }

  /**
   * Generate insights from meeting transcript with optional image analysis
   * @param {Object} params - Request parameters
   * @param {string} params.transcript - Meeting transcript text
   * @param {string} [params.meetingId] - Optional meeting identifier
   * @param {File} [params.imageFile] - Optional image file for visual analysis
   * @param {Array} [params.extractTypes] - Specific insight types to extract
   * @param {number} [params.maxInsights=10] - Maximum number of insights to return
   * @returns {Promise<Object>} - Generated insights response
   */
  async generateInsights({
    transcript,
    meetingId = null,
    imageFile = null,
    extractTypes = null,
    maxInsights = 10
  }) {
    if (!transcript || transcript.trim().length < 10) {
      throw new APIError('Transcript must be at least 10 characters long', 400, 'INVALID_TRANSCRIPT');
    }

    if (imageFile) {
      APIUtils.validateImageFile(imageFile);
    }

    const formData = new FormData();
    
    // Create the request object that matches the backend schema
    const requestData = {
      transcript: transcript.trim(),
      meeting_id: meetingId,
      extract_types: extractTypes,
      max_insights: Math.max(1, Math.min(50, maxInsights))
    };

    formData.append('request', JSON.stringify(requestData));
    
    if (imageFile) {
      formData.append('image_file', imageFile);
    }

    return this.request(API_CONFIG.ENDPOINTS.ANALYZE, {
      method: 'POST',
      body: formData
    });
  }

  /**
   * Simplify existing insights for easier understanding
   * @param {Object} params - Simplification parameters
   * @param {Array} params.originalInsights - Original insight objects
   * @param {Array} params.originalTips - Original tip objects
   * @param {string} [params.simplificationLevel='moderate'] - Level of simplification
   * @param {string} [params.originalInsightId] - ID of original insight
   * @returns {Promise<Object>} - Simplified insights response
   */
  async simplifyInsights({
    originalInsights,
    originalTips,
    simplificationLevel = 'moderate',
    originalInsightId = null
  }) {
    if (!originalInsights || !Array.isArray(originalInsights) || originalInsights.length === 0) {
      throw new APIError('Original insights are required for simplification', 400, 'MISSING_INSIGHTS');
    }

    const validLevels = ['light', 'moderate', 'heavy'];
    if (!validLevels.includes(simplificationLevel)) {
      throw new APIError('Invalid simplification level. Must be: light, moderate, or heavy', 400, 'INVALID_LEVEL');
    }

    return this.request(API_CONFIG.ENDPOINTS.SIMPLIFY, {
      method: 'POST',
      body: JSON.stringify({
        original_insights: originalInsights,
        original_tips: originalTips || [],
        simplification_level: simplificationLevel,
        original_insight_id: originalInsightId
      })
    });
  }

  /**
   * Get the status of an ongoing insight analysis
   * @param {string} insightId - Insight analysis ID
   * @returns {Promise<Object>} - Analysis status response
   */
  async getAnalysisStatus(insightId) {
    if (!insightId || typeof insightId !== 'string') {
      throw new APIError('Valid insight ID is required', 400, 'INVALID_INSIGHT_ID');
    }

    return this.request(`${API_CONFIG.ENDPOINTS.STATUS}/${encodeURIComponent(insightId)}`);
  }

  /**
   * Get insights history for a specific meeting
   * @param {string} meetingId - Meeting identifier
   * @returns {Promise<Object>} - Insights history response
   */
  async getInsightsHistory(meetingId) {
    if (!meetingId || typeof meetingId !== 'string') {
      throw new APIError('Valid meeting ID is required', 400, 'INVALID_MEETING_ID');
    }

    return this.request(`${API_CONFIG.ENDPOINTS.HISTORY}/${encodeURIComponent(meetingId)}`);
  }

  /**
   * Delete specific insights by ID
   * @param {string} insightId - Insight ID to delete
   * @returns {Promise<Object>} - Deletion confirmation
   */
  async deleteInsights(insightId) {
    if (!insightId || typeof insightId !== 'string') {
      throw new APIError('Valid insight ID is required', 400, 'INVALID_INSIGHT_ID');
    }

    return this.request(`${API_CONFIG.ENDPOINTS.DELETE}/${encodeURIComponent(insightId)}`, {
      method: 'DELETE'
    });
  }

  /**
   * Batch analyze multiple meetings at once
   * @param {Object} params - Batch analysis parameters
   * @param {Array<string>} params.meetingContexts - Array of meeting transcripts
   * @param {Array<string>} params.meetingIds - Array of meeting IDs
   * @param {Array<File>} [params.imageFiles] - Optional array of image files
   * @returns {Promise<Object>} - Batch analysis results
   */
  async batchAnalyze({
    meetingContexts,
    meetingIds,
    imageFiles = null
  }) {
    if (!Array.isArray(meetingContexts) || !Array.isArray(meetingIds)) {
      throw new APIError('Meeting contexts and IDs must be arrays', 400, 'INVALID_BATCH_DATA');
    }

    if (meetingContexts.length !== meetingIds.length) {
      throw new APIError('Number of meeting contexts and IDs must match', 400, 'MISMATCHED_ARRAYS');
    }

    if (meetingContexts.length === 0) {
      throw new APIError('At least one meeting context is required', 400, 'EMPTY_BATCH');
    }

    // Validate each context
    meetingContexts.forEach((context, index) => {
      if (!context || context.trim().length < 10) {
        throw new APIError(`Meeting context at index ${index} is too short`, 400, 'INVALID_CONTEXT');
      }
    });

    // Validate image files if provided
    if (imageFiles) {
      if (!Array.isArray(imageFiles)) {
        throw new APIError('Image files must be an array', 400, 'INVALID_IMAGE_FILES');
      }
      imageFiles.forEach((file, index) => {
        if (file) {
          try {
            APIUtils.validateImageFile(file);
          } catch (error) {
            throw new APIError(`Image file at index ${index}: ${error.message}`, 400, 'INVALID_IMAGE_FILE');
          }
        }
      });
    }

    const formData = new FormData();
    
    meetingContexts.forEach(context => formData.append('meeting_contexts', context.trim()));
    meetingIds.forEach(id => formData.append('meeting_ids', id));
    
    if (imageFiles) {
      imageFiles.forEach(file => {
        if (file) {
          formData.append('image_files', file);
        }
      });
    }

    return this.request(API_CONFIG.ENDPOINTS.BATCH_ANALYZE, {
      method: 'POST',
      body: formData
    });
  }

  /**
   * Poll analysis status until completion or timeout
   * @param {string} insightId - Insight ID to poll
   * @param {Object} options - Polling options
   * @param {number} [options.interval=2000] - Polling interval in ms
   * @param {number} [options.timeout=300000] - Timeout in ms (5 minutes)
   * @param {Function} [options.onUpdate] - Callback for status updates
   * @returns {Promise<Object>} - Final status when completed
   */
  async pollAnalysisStatus(insightId, options = {}) {
    const {
      interval = 2000,
      timeout = 300000,
      onUpdate = null
    } = options;

    const startTime = Date.now();

    while (true) {
      try {
        const status = await this.getAnalysisStatus(insightId);
        
        if (onUpdate) {
          onUpdate(status);
        }

        if (status.status === 'completed' || status.status === 'failed') {
          return status;
        }

        if (Date.now() - startTime > timeout) {
          throw new APIError('Analysis polling timeout', 408, 'POLLING_TIMEOUT');
        }

        await APIUtils.sleep(interval);
      } catch (error) {
        if (error.status === 404) {
          throw new APIError('Analysis not found', 404, 'ANALYSIS_NOT_FOUND');
        }
        throw error;
      }
    }
  }
}

/**
 * Custom API Error class with enhanced error information
 */
class APIError extends Error {
  constructor(message, status = 500, code = 'UNKNOWN_ERROR', requestId = null) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.timestamp = new Date().toISOString();
  }

  /**
   * Convert error to user-friendly message
   * @returns {string} - User-friendly error message
   */
  toUserMessage() {
    const userMessages = {
      'INVALID_TRANSCRIPT': 'Please enter a valid meeting transcript (at least 10 characters).',
      'MISSING_INSIGHTS': 'No insights available to simplify. Please generate insights first.',
      'INVALID_LEVEL': 'Please select a valid simplification level.',
      'INVALID_INSIGHT_ID': 'Invalid insight ID provided.',
      'INVALID_MEETING_ID': 'Invalid meeting ID provided.',
      'TIMEOUT': 'Request timed out. Please try again.',
      'NETWORK_ERROR': 'Network error. Please check your connection.',
      'SERVER_ERROR': 'Server error. Please try again later.',
      'ANALYSIS_NOT_FOUND': 'Analysis not found. It may have expired.',
      'POLLING_TIMEOUT': 'Analysis is taking longer than expected. Please check status manually.'
    };

    return userMessages[this.code] || this.message || 'An unexpected error occurred.';
  }

  /**
   * Check if error is retryable
   * @returns {boolean} - Whether the error can be retried
   */
  isRetryable() {
    const retryableCodes = ['TIMEOUT', 'NETWORK_ERROR', 'SERVER_ERROR'];
    const retryableStatuses = [408, 429, 500, 502, 503, 504];
    
    return retryableCodes.includes(this.code) || retryableStatuses.includes(this.status);
  }
}

/**
 * React Hook for managing Key Insights API state
 * @param {string} [baseURL] - API base URL
 * @returns {Object} - Hook state and methods
 */
function useKeyInsightsAPI(baseURL) {
  const [api] = useState(() => new KeyInsightsAPIService(baseURL));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRequest = async (requestFn, successMessage = null) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await requestFn();
      if (successMessage) {
        // You could trigger a success notification here
        console.log(successMessage);
      }
      return result;
    } catch (err) {
      const apiError = err instanceof APIError ? err : new APIError(err.message);
      setError(apiError);
      console.error('API request failed:', apiError);
      throw apiError;
    } finally {
      setLoading(false);
    }
  };

  return {
    api,
    loading,
    error,
    setError,
    handleRequest
  };
}

/**
 * Local storage utilities for caching insights
 */
class InsightsCache {
  static CACHE_PREFIX = 'key_insights_';
  static CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours

  static set(key, data) {
    try {
      const cacheData = {
        data,
        timestamp: Date.now(),
        expires: Date.now() + this.CACHE_EXPIRY
      };
      localStorage.setItem(this.CACHE_PREFIX + key, JSON.stringify(cacheData));
    } catch (error) {
      console.warn('Failed to cache data:', error);
    }
  }

  static get(key) {
    try {
      const cached = localStorage.getItem(this.CACHE_PREFIX + key);
      if (!cached) return null;

      const cacheData = JSON.parse(cached);
      if (Date.now() > cacheData.expires) {
        this.remove(key);
        return null;
      }

      return cacheData.data;
    } catch (error) {
      console.warn('Failed to retrieve cached data:', error);
      return null;
    }
  }

  static remove(key) {
    try {
      localStorage.removeItem(this.CACHE_PREFIX + key);
    } catch (error) {
      console.warn('Failed to remove cached data:', error);
    }
  }

  static clear() {
    try {
      const keys = Object.keys(localStorage);
      keys.forEach(key => {
        if (key.startsWith(this.CACHE_PREFIX)) {
          localStorage.removeItem(key);
        }
      });
    } catch (error) {
      console.warn('Failed to clear cache:', error);
    }
  }
}

// Export everything for use in React components
export {
  KeyInsightsAPIService,
  APIUtils,
  APIError,
  useKeyInsightsAPI,
  InsightsCache,
  API_CONFIG
};

// Default export
export default KeyInsightsAPIService;