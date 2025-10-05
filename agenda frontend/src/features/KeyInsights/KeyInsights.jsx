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

  useEffect(() => {
    loadInsightTypes();
  }, []);

  const loadInsightTypes = async () => {
    try {
      const types = await api.getInsightTypes();
      setInsightTypes(types);
    } catch (err) {
      console.error('Failed to load insight types:', err);
    }
  };

  const handleGenerateInsights = async () => {
    if (!transcript.trim()) {
      setError('Please enter a transcript');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await api.generateInsights(
        transcript,
        meetingId || null,
        imageFile,
        selectedTypes.length > 0 ? selectedTypes : null,
        maxInsights
      );
      setInsights(result);
      setSuccess('Insights generated successfully!');
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

  const handleGetSample = async () => {
    setLoading(true);
    try {
      const sample = await api.getSampleInsight();
      setInsights({ key_insights: [sample], situation_tips: [] });
      setSuccess('Sample insight loaded!');
    } catch (err) {
      setError(`Failed to get sample: ${err.message}`);
    } finally {
      setLoading(false);
    }
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

  const handleDeleteInsight = async (insightId) => {
    if (!window.confirm('Are you sure you want to delete this insight?')) return;

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

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setImageFile(file);
    } else {
      setError('Please select a valid image file');
    }
  };

  const renderInsights = (insightsData) => {
    if (!insightsData) return null;

    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 className="text-xl font-semibold text-gray-100 mb-4 flex items-center">
          <Brain className="mr-2" size={20} />
          Key Insights
        </h3>
        
        {insightsData.key_insights?.map((insight, index) => (
          <div key={index} className="mb-3 p-3 bg-gray-700 rounded-lg border border-gray-600">
            <div className="flex justify-between items-start">
              <p className="text-gray-200 flex-1">{insight.point || insight.content}</p>
              {insight.confidence && (
                <span className="ml-2 px-2 py-1 bg-purple-600 text-white text-xs rounded">
                  {Math.round(insight.confidence * 100)}%
                </span>
              )}
            </div>
            {insight.category && (
              <span className="text-xs text-purple-400 mt-1 block">
                {insight.category}
              </span>
            )}
          </div>
        ))}

        {insightsData.situation_tips?.length > 0 && (
          <div className="mt-6">
            <h4 className="text-lg font-medium text-gray-200 mb-3">Tips</h4>
            {insightsData.situation_tips.map((tip, index) => (
              <div key={index} className="mb-2 p-3 bg-gray-700 rounded-lg border border-gray-600">
                <p className="text-gray-200">{tip.tip}</p>
                {tip.category && (
                  <span className="text-xs text-green-400 mt-1 block">
                    {tip.category}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-roboto">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-100 mb-4">
            Key Insights Dashboard
          </h1>
          <p className="text-gray-400 text-lg">
            Generate and manage meeting insights with AI analysis
          </p>
        </div>

        {/* Alert Messages */}
        {error && (
          <div className="mb-4 p-4 bg-red-900 border border-red-700 rounded-lg flex items-center">
            <AlertCircle className="mr-2" size={20} />
            <span className="text-red-200">{error}</span>
            <button 
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-200"
            >
              ×
            </button>
          </div>
        )}

        {success && (
          <div className="mb-4 p-4 bg-green-900 border border-green-700 rounded-lg flex items-center">
            <CheckCircle className="mr-2" size={20} />
            <span className="text-green-200">{success}</span>
            <button 
              onClick={() => setSuccess(null)}
              className="ml-auto text-green-400 hover:text-green-200"
            >
              ×
            </button>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="flex flex-wrap justify-center mb-8 space-x-2">
          {[
            { id: 'analyze', label: 'Analyze', icon: Brain },
            { id: 'simplify', label: 'Simplify', icon: RefreshCw },
            { id: 'history', label: 'History', icon: History },
            { id: 'sample', label: 'Sample', icon: Eye }
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center ${
                activeTab === id
                  ? 'bg-purple-600 text-white shadow-lg'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Icon className="mr-2" size={18} />
              {label}
            </button>
          ))}
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            {activeTab === 'analyze' && (
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

                  <button
                    onClick={handleGenerateInsights}
                    disabled={loading || !transcript.trim()}
                    className="w-full py-3 px-6 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center"
                  >
                    {loading ? (
                      <RefreshCw className="animate-spin mr-2" size={20} />
                    ) : (
                      <Brain className="mr-2" size={20} />
                    )}
                    Generate Insights
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'simplify' && (
              <div>
                <h2 className="text-2xl font-semibold mb-4 flex items-center">
                  <RefreshCw className="mr-2" size={24} />
                  Simplify Insights
                </h2>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Simplification Level
                    </label>
                    <select
                      value={simplificationLevel}
                      onChange={(e) => setSimplificationLevel(e.target.value)}
                      className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
                    >
                      <option value="light">Light</option>
                      <option value="moderate">Moderate</option>
                      <option value="heavy">Heavy</option>
                    </select>
                  </div>

                  <button
                    onClick={handleSimplifyInsights}
                    disabled={loading || !insights?.key_insights}
                    className="w-full py-3 px-6 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center"
                  >
                    {loading ? (
                      <RefreshCw className="animate-spin mr-2" size={20} />
                    ) : (
                      <RefreshCw className="mr-2" size={20} />
                    )}
                    Simplify Insights
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'history' && (
              <div>
                <h2 className="text-2xl font-semibold mb-4 flex items-center">
                  <History className="mr-2" size={24} />
                  Insights History
                </h2>
                
                <div className="space-y-4">
                  <button
                    onClick={handleLoadHistory}
                    disabled={loading || !meetingId.trim()}
                    className="w-full py-3 px-6 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center"
                  >
                    {loading ? (
                      <RefreshCw className="animate-spin mr-2" size={20} />
                    ) : (
                      <History className="mr-2" size={20} />
                    )}
                    Load History
                  </button>

                  {history.length > 0 && (
                    <div className="space-y-2">
                      {history.map((item, index) => (
                        <div key={index} className="p-3 bg-gray-700 rounded-lg border border-gray-600 flex justify-between items-center">
                          <div>
                            <p className="text-sm text-gray-200">
                              {new Date(item.generated_at).toLocaleString()}
                            </p>
                            <p className="text-xs text-gray-400">
                              {item.insights_count} insights, {item.tips_count} tips
                            </p>
                          </div>
                          <button
                            onClick={() => handleDeleteInsight(item.insight_id)}
                            className="p-2 text-red-400 hover:text-red-300 hover:bg-red-900 rounded transition-colors"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'sample' && (
              <div>
                <h2 className="text-2xl font-semibold mb-4 flex items-center">
                  <Eye className="mr-2" size={24} />
                  Sample Insight
                </h2>
                
                <button
                  onClick={handleGetSample}
                  disabled={loading}
                  className="w-full py-3 px-6 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center"
                >
                  {loading ? (
                    <RefreshCw className="animate-spin mr-2" size={20} />
                  ) : (
                    <Eye className="mr-2" size={20} />
                  )}
                  Get Sample Insight
                </button>
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div>
            {activeTab === 'simplify' && simplifiedInsights ? (
              <div>
                <h3 className="text-xl font-semibold text-gray-100 mb-4">
                  Simplified Results
                </h3>
                {renderInsights(simplifiedInsights)}
              </div>
            ) : (
              insights && renderInsights(insights)
            )}
          </div>
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

export default KeyInsightsDashboard;