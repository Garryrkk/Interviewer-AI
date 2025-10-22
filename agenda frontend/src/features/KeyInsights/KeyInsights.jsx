import React, { useState, useEffect } from 'react';
import { Upload, FileText, Brain, Clock, History, Trash2, Eye, BarChart3, RefreshCw, CheckCircle, AlertCircle } from 'lucide-react';
import { keyInsightsAPI } from '../../services/aiService';

const keyInsightsAPI = new KeyInsightsAPIService();

// Get all insight types
keyInsightsAPI.getInsightTypes()
  .then(types => {
    console.log("Insight Types:", types);
  })
  .catch(err => {
    console.error("API Error:", err);
  });

// Get a sample insight
keyInsightsAPI.getSampleInsight()
  .then(sample => {
    console.log("Sample Insight:", sample);
  })
  .catch(err => {
    console.error("API Error:", err);
  });


KeyInsights("This is the text I want insights for")
  .then(response => {
    console.log("API Response:", response);
  })
  .catch(err => {
    console.error("API Error:", err);
  });

// API Service Class
class KeyInsightsAPI {
  constructor(baseURL = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    };

    if (options.body && options.body instanceof FormData) {
      delete config.headers['Content-Type'];
    }

    try {
      const response = await fetch(url, config);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || `HTTP error! status: ${response.status}`);
      }
      
      return data;
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      throw error;
    }
  }

  // Get all insight types
  async getInsightTypes() {
    return this.request('/api/v1/key-insights/types');
  }

  // Get sample insight
  async getSampleInsight() {
    return this.request('/api/v1/key-insights/sample');
  }

  // Generate insights from transcript
  async generateInsights(transcript, meetingId = null, imageFile = null, extractTypes = null, maxInsights = 10) {
    const formData = new FormData();
    
    const requestData = {
      transcript,
      meeting_id: meetingId,
      extract_types: extractTypes,
      max_insights: maxInsights
    };

    formData.append('request', JSON.stringify(requestData));
    
    if (imageFile) {
      formData.append('image_file', imageFile);
    }

    return this.request('/api/v1/key-insights/analyze', {
      method: 'POST',
      body: formData
    });
  }

  // Simplify insights
  async simplifyInsights(originalInsights, originalTips, simplificationLevel = 'moderate', originalInsightId = null) {
    return this.request('/api/v1/key-insights/simplify', {
      method: 'POST',
      body: JSON.stringify({
        original_insights: originalInsights,
        original_tips: originalTips,
        simplification_level: simplificationLevel,
        original_insight_id: originalInsightId
      })
    });
  }

  // Get analysis status
  async getAnalysisStatus(insightId) {
    return this.request(`/api/v1/key-insights/status/${insightId}`);
  }

  // Get insights history
  async getInsightsHistory(meetingId) {
    return this.request(`/api/v1/key-insights/history/${meetingId}`);
  }

  // Delete insights
  async deleteInsights(insightId) {
    return this.request(`/api/v1/key-insights/insights/${insightId}`, {
      method: 'DELETE'
    });
  }

  // Batch analyze
  async batchAnalyze(meetingContexts, meetingIds, imageFiles = null) {
    const formData = new FormData();
    
    meetingContexts.forEach(context => formData.append('meeting_contexts', context));
    meetingIds.forEach(id => formData.append('meeting_ids', id));
    
    if (imageFiles) {
      imageFiles.forEach(file => formData.append('image_files', file));
    }

    return this.request('/api/v1/key-insights/batch-analyze', {
      method: 'POST',
      body: formData
    });
  }
}

const KeyInsightsDashboard = () => {
  const [api] = useState(new KeyInsightsAPI());
  const [activeTab, setActiveTab] = useState('analyze');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // State for different features
  const [transcript, setTranscript] = useState('');
  const [meetingId, setMeetingId] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [insights, setInsights] = useState(null);
  const [insightTypes, setInsightTypes] = useState([]);
  const [selectedTypes, setSelectedTypes] = useState([]);
  const [maxInsights, setMaxInsights] = useState(10);
  const [analysisStatus, setAnalysisStatus] = useState({});
  const [history, setHistory] = useState([]);
  const [simplifiedInsights, setSimplifiedInsights] = useState(null);
  const [simplificationLevel, setSimplificationLevel] = useState('moderate');
  const [insightsHistory, setInsightsHistory] = useState([]);
  const [batchMode, setBatchMode] = useState(false);
  const [batchContexts, setBatchContexts] = useState(['']);
  const [batchIds, setBatchIds] = useState(['']);

  // Clear error handler
  const clearError = () => {
    setError('');
  };

  // Generate random meeting ID
  const generateMeetingId = () => {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    setMeetingId(`meeting_${timestamp}_${random}`);
  };

  // Generate Insights
  const generateInsights = async () => {
    if (!meetingContext && !selectedFile) {
      setError('Please provide either meeting context or upload an image file');
      return;
    }

    setLoading(true);
    setError('');
    setInsights(null);

    try {
      let response;
      
      if (selectedFile) {
        // Use FormData for file upload
        const formData = new FormData();
        
        const requestData = {
          meeting_context: meetingContext || null,
          meeting_id: meetingId || null,
          participants: participants.split(',').map(p => p.trim()).filter(p => p),
          analysis_focus: analysisFocus || null,
          include_visual_analysis: true
        };

        // Append each field individually for FormData
        formData.append('meeting_context', requestData.meeting_context || '');
        formData.append('meeting_id', requestData.meeting_id || '');
        formData.append('participants', JSON.stringify(requestData.participants));
        formData.append('analysis_focus', requestData.analysis_focus || '');
        formData.append('include_visual_analysis', 'true');
        formData.append('image_file', selectedFile);

        response = await fetch(`${BASE_URL}/analyze`, {
          method: 'POST',
          body: formData, // No Content-Type header for FormData
        });
      } else {
        // Use JSON for text-only requests
        const requestData = {
          meeting_context: meetingContext,
          meeting_id: meetingId || null,
          participants: participants.split(',').map(p => p.trim()).filter(p => p),
          analysis_focus: analysisFocus || null,
          include_visual_analysis: false
        };

        response = await fetch(`${BASE_URL}/analyze`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestData),
        });
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setInsights(data);
      setSuccessMessage('Insights generated successfully!');
      
      // Start status polling
      if (data.insight_id) {
        pollAnalysisStatus(data.insight_id);
      }
    } catch (err) {
      setError(`Failed to generate insights: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSimplifyInsights = async () => {
    if (!insights?.key_insights) {
      setError('No insights to simplify. Generate insights first.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await api.simplifyInsights(
        insights.key_insights,
        insights.situation_tips || [],
        simplificationLevel,
        insights.insight_id
      );
      setSimplifiedInsights(result);
      setSuccess('Insights simplified successfully!');
    } catch (err) {
      setError(`Failed to simplify insights: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Poll Analysis Status
  const pollAnalysisStatus = async (insightId) => {
    setStatusPolling(true);
    
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${BASE_URL}/status/${insightId}`);
        const status = await response.json();
        setAnalysisStatus(status);

        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(pollInterval);
          setStatusPolling(false);
        }
      } catch (err) {
        console.error('Status polling error:', err);
        clearInterval(pollInterval);
        setStatusPolling(false);
      }
    }, 2000);

    // Stop polling after 5 minutes
    setTimeout(() => {
      clearInterval(pollInterval);
      setStatusPolling(false);
    }, 300000);
  };

  const handleLoadHistory = async () => {
    if (!meetingId.trim()) {
      setError('Please enter a meeting ID');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await api.getInsightsHistory(meetingId);
      setHistory(result.insights_history || []);
      setSuccess('History loaded successfully!');
    } catch (err) {
      setError(`Failed to load history: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Delete Insights
  const deleteInsights = async (insightId) => {
    if (!insightId) return;

    try {
      const response = await fetch(`${BASE_URL}/insights/${insightId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setInsights(null);
      setInsightsHistory(prev => prev.filter(item => item.insight_id !== insightId));
    } catch (err) {
      setError(`Failed to delete insights: ${err.message}`);
    }
  };

  // Batch Analysis
  const batchAnalyzeInsights = async () => {
    const validContexts = batchContexts.filter(c => c.trim());
    const validIds = batchIds.filter(id => id.trim());
    
    if (validContexts.length !== validIds.length || validContexts.length === 0) {
      setError('Please provide valid contexts and IDs for batch analysis. Each meeting must have both a context and an ID.');
      return;
    }

    setLoading(true);
    try {
      await api.deleteInsights(insightId);
      setHistory(history.filter(h => h.insight_id !== insightId));
      setSuccess('Insight deleted successfully!');
    } catch (err) {
      setError(`Failed to delete insight: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Add batch input
  const addBatchInput = () => {
    setBatchContexts([...batchContexts, '']);
    setBatchIds([...batchIds, '']);
  };

  // Remove batch input
  const removeBatchInput = (index) => {
    setBatchContexts(batchContexts.filter((_, i) => i !== index));
    setBatchIds(batchIds.filter((_, i) => i !== index));
  };

  // File upload handler
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setImageFile(file);
    } else {
      setError('Please select a valid image file');
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#1E1E2F', 
      color: '#F8FAFC', 
      fontFamily: 'Roboto, sans-serif',
      padding: '20px'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '2rem', textAlign: 'center' }}>
          <Brain style={{ display: 'inline', marginRight: '10px' }} />
          Key Insights Dashboard
        </h1>

        {/* Success Message */}
        {successMessage && (
          <div style={{ 
            backgroundColor: '#d1fae5', 
            color: '#065f46', 
            padding: '12px', 
            borderRadius: '8px', 
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <CheckCircle style={{ marginRight: '8px', minWidth: '20px' }} size={20} />
              {successMessage}
            </div>
            <button
              onClick={() => setSuccessMessage('')}
              style={{
                backgroundColor: 'transparent',
                border: 'none',
                color: '#065f46',
                cursor: 'pointer',
                fontSize: '16px',
                fontWeight: 'bold'
              }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div style={{ 
            backgroundColor: '#fee2e2', 
            color: '#dc2626', 
            padding: '12px', 
            borderRadius: '8px', 
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <AlertCircle style={{ marginRight: '8px', minWidth: '20px' }} size={20} />
              {error}
            </div>
            <button
              onClick={clearError}
              style={{
                backgroundColor: 'transparent',
                border: 'none',
                color: '#dc2626',
                cursor: 'pointer',
                fontSize: '16px',
                fontWeight: 'bold'
              }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Analysis Status */}
        {analysisStatus && (
          <div style={{ 
            backgroundColor: '#374151', 
            padding: '16px', 
            borderRadius: '12px', 
            marginBottom: '20px',
            border: '1px solid #4B5563'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
              <Clock style={{ marginRight: '8px' }} size={20} />
              <span style={{ fontWeight: '600' }}>Analysis Status: {analysisStatus.status}</span>
              {statusPolling && <RefreshCw style={{ marginLeft: '10px', animation: 'spin 1s linear infinite' }} size={16} />}
            </div>
            <div style={{ 
              backgroundColor: '#1F2937', 
              borderRadius: '8px', 
              height: '8px', 
              overflow: 'hidden' 
            }}>
              <div style={{ 
                backgroundColor: '#8F74D4', 
                height: '100%', 
                width: `${analysisStatus.progress}%`,
                transition: 'width 0.3s ease'
              }} />
            </div>
            <div style={{ fontSize: '0.875rem', color: '#9CA3AF', marginTop: '4px' }}>
              Progress: {analysisStatus.progress}%
            </div>
          </div>
        )}

        {/* Mode Toggle */}
        <div style={{ marginBottom: '20px', textAlign: 'center' }}>
          <button
            onClick={() => setBatchMode(!batchMode)}
            style={{
              backgroundColor: batchMode ? '#8F74D4' : '#374151',
              color: '#F8FAFC',
              border: 'none',
              padding: '12px 24px',
              borderRadius: '12px',
              cursor: 'pointer',
              fontWeight: '600',
              transition: 'all 0.3s ease'
            }}
          >
            {batchMode ? 'Switch to Single Analysis' : 'Switch to Batch Analysis'}
          </button>
        </div>

        {!batchMode ? (
          // Single Analysis Mode
          <div style={{ 
            backgroundColor: '#374151', 
            padding: '24px', 
            borderRadius: '16px', 
            marginBottom: '24px',
            border: '1px solid #4B5563'
          }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '20px', display: 'flex', alignItems: 'center' }}>
              <FileText style={{ marginRight: '8px' }} />
              Meeting Analysis
            </h2>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
              <div>
                <h2 className="text-2xl font-semibold mb-4 flex items-center">
                  <FileText className="mr-2" size={24} />
                  Generate Insights
                </h2>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Meeting Transcript *
                    </label>
                    <textarea
                      value={transcript}
                      onChange={(e) => setTranscript(e.target.value)}
                      placeholder="Enter your meeting transcript here..."
                      className="w-full h-32 p-3 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Meeting ID (Optional)
                    </label>
                    <input
                      type="text"
                      value={meetingId}
                      onChange={(e) => setMeetingId(e.target.value)}
                      placeholder="Enter meeting ID"
                      className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Upload Image (Optional)
                    </label>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-purple-600 file:text-white hover:file:bg-purple-700"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Insight Types (Optional)
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                      {insightTypes.map((type) => (
                        <label key={type} className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={selectedTypes.includes(type)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedTypes([...selectedTypes, type]);
                              } else {
                                setSelectedTypes(selectedTypes.filter(t => t !== type));
                              }
                            }}
                            className="rounded border-gray-600 text-purple-600 focus:ring-purple-500"
                          />
                          <span className="text-sm text-gray-300 capitalize">
                            {type.replace('_', ' ')}
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Max Insights: {maxInsights}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="50"
                      value={maxInsights}
                      onChange={(e) => setMaxInsights(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                    />
                  </div>

            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
              <button
                onClick={generateInsights}
                disabled={loading}
                style={{
                  backgroundColor: '#8F74D4',
                  color: '#F8FAFC',
                  border: 'none',
                  padding: '14px 28px',
                  borderRadius: '12px',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  fontWeight: '600',
                  fontSize: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  opacity: loading ? 0.7 : 1,
                  transition: 'all 0.3s ease'
                }}
              >
                {loading ? (
                  <RefreshCw style={{ marginRight: '8px', animation: 'spin 1s linear infinite' }} size={20} />
                ) : (
                  <Brain style={{ marginRight: '8px' }} size={20} />
                )}
                {loading ? 'Analyzing...' : 'Generate Insights'}
              </button>

              <button
                onClick={getInsightsHistory}
                style={{
                  backgroundColor: '#374151',
                  color: '#F8FAFC',
                  border: '1px solid #8F74D4',
                  padding: '14px 28px',
                  borderRadius: '12px',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  transition: 'all 0.3s ease'
                }}
              >
                <Clock style={{ marginRight: '8px' }} size={20} />
                Get History
              </button>
            </div>
          </div>
        ) : (
          // Batch Analysis Mode
          <div style={{ 
            backgroundColor: '#374151', 
            padding: '24px', 
            borderRadius: '16px', 
            marginBottom: '24px',
            border: '1px solid #4B5563'
          }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '20px', display: 'flex', alignItems: 'center' }}>
              <TrendingUp style={{ marginRight: '8px' }} />
              Batch Analysis
            </h2>

            {batchContexts.map((context, index) => (
              <div key={index} style={{ marginBottom: '16px', border: '1px solid #4B5563', borderRadius: '8px', padding: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'between', alignItems: 'center', marginBottom: '12px' }}>
                  <span style={{ fontWeight: '600' }}>Meeting {index + 1}</span>
                  {batchContexts.length > 1 && (
                    <button
                      onClick={() => removeBatchInput(index)}
                      style={{
                        backgroundColor: '#dc2626',
                        color: '#F8FAFC',
                        border: 'none',
                        padding: '4px 8px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      Remove
                    </button>
                  )}
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '12px' }}>
                  <input
                    type="text"
                    value={batchIds[index]}
                    onChange={(e) => {
                      const newIds = [...batchIds];
                      newIds[index] = e.target.value;
                      setBatchIds(newIds);
                    }}
                    placeholder={`Meeting ID ${index + 1}`}
                    style={{
                      width: '100%',
                      padding: '8px',
                      backgroundColor: '#1F2937',
                      border: '1px solid #4B5563',
                      borderRadius: '6px',
                      color: '#F8FAFC',
                      fontSize: '14px'
                    }}
                  />
                  
                  <textarea
                    value={context}
                    onChange={(e) => {
                      const newContexts = [...batchContexts];
                      newContexts[index] = e.target.value;
                      setBatchContexts(newContexts);
                    }}
                    placeholder={`Meeting context ${index + 1}...`}
                    rows={3}
                    style={{
                      width: '100%',
                      padding: '8px',
                      backgroundColor: '#1F2937',
                      border: '1px solid #4B5563',
                      borderRadius: '6px',
                      color: '#F8FAFC',
                      fontSize: '14px'
                    }}
                  />
                </div>
              </div>
            )}

        {/* Insights Display */}
        {insights && (
          <div style={{ 
            backgroundColor: '#374151', 
            padding: '24px', 
            borderRadius: '16px', 
            marginBottom: '24px',
            border: '1px solid #4B5563'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: '600', display: 'flex', alignItems: 'center' }}>
                <CheckCircle style={{ marginRight: '8px', color: '#10B981' }} />
                Generated Insights
              </h2>
              
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <select
                  value={simplificationLevel}
                  onChange={(e) => setSimplificationLevel(e.target.value)}
                  style={{
                    padding: '8px 12px',
                    backgroundColor: '#1F2937',
                    border: '1px solid #4B5563',
                    borderRadius: '8px',
                    color: '#F8FAFC',
                    fontSize: '14px'
                  }}
                >
                  <option value="light">Light Simplification</option>
                  <option value="moderate">Moderate Simplification</option>
                  <option value="heavy">Heavy Simplification</option>
                </select>
                
                <button
                  onClick={simplifyInsights}
                  style={{
                    backgroundColor: '#10B981',
                    color: '#F8FAFC',
                    border: 'none',
                    padding: '8px 16px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    fontWeight: '500',
                    fontSize: '14px'
                  }}
                >
                  Simplify
                </button>
                
                <button
                  onClick={() => deleteInsights(insights.insight_id)}
                  style={{
                    backgroundColor: '#dc2626',
                    color: '#F8FAFC',
                    border: 'none',
                    padding: '8px 12px',
                    borderRadius: '8px',
                    cursor: 'pointer'
                  }}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>

            {/* Insights Meta Info */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
              gap: '16px', 
              marginBottom: '20px' 
            }}>
              <div style={{ backgroundColor: '#1F2937', padding: '12px', borderRadius: '8px' }}>
                <div style={{ fontSize: '12px', color: '#9CA3AF', marginBottom: '4px' }}>Insight ID</div>
                <div style={{ fontSize: '14px', fontWeight: '500' }}>{insights.insight_id}</div>
              </div>
              <div style={{ backgroundColor: '#1F2937', padding: '12px', borderRadius: '8px' }}>
                <div style={{ fontSize: '12px', color: '#9CA3AF', marginBottom: '4px' }}>Confidence Score</div>
                <div style={{ fontSize: '14px', fontWeight: '500' }}>{(insights.confidence_score * 100).toFixed(1)}%</div>
              </div>
              <div style={{ backgroundColor: '#1F2937', padding: '12px', borderRadius: '8px' }}>
                <div style={{ fontSize: '12px', color: '#9CA3AF', marginBottom: '4px' }}>Visual Analysis</div>
                <div style={{ fontSize: '14px', fontWeight: '500' }}>
                  {insights.visual_analysis_included ? 'Included' : 'Not Included'}
                </div>
              </div>
              <div style={{ backgroundColor: '#1F2937', padding: '12px', borderRadius: '8px' }}>
                <div style={{ fontSize: '12px', color: '#9CA3AF', marginBottom: '4px' }}>Participants</div>
                <div style={{ fontSize: '14px', fontWeight: '500' }}>
                  {insights.participants_analyzed?.length || 0}
                </div>
              </div>
            </div>

            {/* Key Insights */}
            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center' }}>
                <Brain style={{ marginRight: '8px' }} size={20} />
                Key Insights
              </h3>
              <div style={{ display: 'grid', gap: '12px' }}>
                {(insights.simplified_insights || insights.key_insights)?.map((insight, index) => (
                  <div key={index} style={{ 
                    backgroundColor: '#1F2937', 
                    padding: '16px', 
                    borderRadius: '8px',
                    borderLeft: '4px solid #8F74D4'
                  }}>
                    <div style={{ fontSize: '16px', marginBottom: '8px' }}>{insight.point}</div>
                    <div style={{ display: 'flex', gap: '12px', fontSize: '12px', color: '#9CA3AF' }}>
                      <span>Category: {insight.category}</span>
                      <span>Priority: {insight.priority}</span>
                      <span>Confidence: {(insight.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Situation Tips */}
            <div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center' }}>
                <TrendingUp style={{ marginRight: '8px' }} size={20} />
                Situation Tips
              </h3>
              <div style={{ display: 'grid', gap: '12px' }}>
                {(insights.simplified_tips || insights.situation_tips)?.map((tip, index) => (
                  <div key={index} style={{ 
                    backgroundColor: '#1F2937', 
                    padding: '16px', 
                    borderRadius: '8px',
                    borderLeft: '4px solid #10B981'
                  }}>
                    <div style={{ fontSize: '16px', marginBottom: '8px' }}>{tip.tip}</div>
                    <div style={{ display: 'flex', gap: '12px', fontSize: '12px', color: '#9CA3AF' }}>
                      <span>Category: {tip.category}</span>
                      <span>Actionability: {tip.actionability}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* History Display */}
        {insightsHistory.length > 0 && (
          <div style={{ 
            backgroundColor: '#374151', 
            padding: '24px', 
            borderRadius: '16px',
            border: '1px solid #4B5563'
          }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '20px', display: 'flex', alignItems: 'center' }}>
              <Clock style={{ marginRight: '8px' }} />
              Insights History
            </h2>
            
            <div style={{ display: 'grid', gap: '16px' }}>
              {insightsHistory.map((item, index) => (
                <div key={index} style={{ 
                  backgroundColor: '#1F2937', 
                  padding: '16px', 
                  borderRadius: '8px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div>
                    <div style={{ fontWeight: '600', marginBottom: '4px' }}>
                      Insight ID: {item.insight_id}
                    </div>
                    <div style={{ fontSize: '14px', color: '#9CA3AF' }}>
                      Insights: {item.insights_count} | Tips: {item.tips_count}
                    </div>
                  </div>
                  
                  <button
                    onClick={() => deleteInsights(item.insight_id)}
                    style={{
                      backgroundColor: '#dc2626',
                      color: '#F8FAFC',
                      border: 'none',
                      padding: '8px 12px',
                      borderRadius: '8px',
                      cursor: 'pointer'
                    }}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Batch Results Display */}
        {insights?.batch_results && (
          <div style={{ 
            backgroundColor: '#374151', 
            padding: '24px', 
            borderRadius: '16px',
            marginTop: '24px',
            border: '1px solid #4B5563'
          }}>
            <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '20px', display: 'flex', alignItems: 'center' }}>
              <TrendingUp style={{ marginRight: '8px' }} />
              Batch Analysis Results
            </h2>
            
            <div style={{ display: 'grid', gap: '16px' }}>
              {insights.batch_results.map((result, index) => (
                <div key={index} style={{ 
                  backgroundColor: result.status === 'success' ? '#1F2937' : '#7F1D1D', 
                  padding: '16px', 
                  borderRadius: '8px',
                  borderLeft: `4px solid ${result.status === 'success' ? '#10B981' : '#EF4444'}`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontWeight: '600' }}>Meeting ID: {result.meeting_id}</span>
                    <span style={{ 
                      fontSize: '12px', 
                      padding: '4px 8px', 
                      borderRadius: '12px',
                      backgroundColor: result.status === 'success' ? '#10B981' : '#EF4444',
                      color: '#F8FAFC'
                    }}>
                      {result.status.toUpperCase()}
                    </span>
                  </div>
                  
                  {result.status === 'success' && result.insights && (
                    <div>
                      <div style={{ fontSize: '14px', color: '#9CA3AF', marginBottom: '8px' }}>
                        Insights: {result.insights.key_insights?.length || 0} | 
                        Tips: {result.insights.situation_tips?.length || 0} |
                        Confidence: {((result.insights.confidence_score || 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                  )}
                  
                  {result.status === 'error' && (
                    <div style={{ fontSize: '14px', color: '#FCA5A5' }}>
                      Error: {result.error}
                    </div>
                  )}
                  Get Sample Insight
                </button>
              </div>
            )}
          </div>
        )}

        {/* API Configuration */}
        <div style={{ 
          backgroundColor: '#374151', 
          padding: '20px', 
          borderRadius: '12px',
          marginTop: '24px',
          border: '1px solid #4B5563'
        }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '16px' }}>
            API Configuration
          </h3>
          <div style={{ fontSize: '14px', color: '#9CA3AF' }}>
            <p>Base URL: {BASE_URL}</p>
            <p>Available Endpoints:</p>
            <ul style={{ marginLeft: '20px', marginTop: '8px' }}>
              <li>POST /analyze - Generate insights from meeting context</li>
              <li>POST /simplify - Simplify existing insights</li>
              <li>GET /status/:id - Get analysis status</li>
              <li>GET /history/:meeting_id - Get insights history</li>
              <li>DELETE /insights/:id - Delete insights</li>
              <li>POST /batch-analyze - Batch analysis</li>
            </ul>
          </div>
          {/* Insight Types */}
          {insightTypes.length > 0 && (
            <div style={{ marginTop: '16px' }}>
              <h4 style={{ fontWeight: '600', marginBottom: '8px' }}>Available Insight Types</h4>
              <ul style={{ marginLeft: '20px' }}>
                {insightTypes.map((t, idx) => (
                  <li key={idx}>{t}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Sample Insight */}
          {sampleInsight && (
            <div style={{ marginTop: '16px' }}>
              <h4 style={{ fontWeight: '600', marginBottom: '8px' }}>Sample Insight</h4>
              <pre style={{
                backgroundColor: '#1F2937',
                padding: '12px',
                borderRadius: '8px',
                overflowX: 'auto'
              }}>
                {JSON.stringify(sampleInsight, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #8F74D4;
          cursor: pointer;
        }
        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #8F74D4;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  );
};

export default KeyInsightsDashboard;Generated: {new Date(item.generated_at).toLocaleString()}
                    