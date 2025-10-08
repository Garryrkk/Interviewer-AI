import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, Square, Settings, Users, BarChart3, Clock, AlertTriangle, CheckCircle, XCircle, Camera, Mic, FileText, Download, RefreshCw } from 'lucide-react';

// Main Dashboard Component
const MeetingAnalysisDashboard = () => {
  const [activeTab, setActiveTab] = useState('analyze');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [healthStatus, setHealthStatus] = useState(null);
  const [meetingContext, setMeetingContext] = useState('');
  const [audioTranscript, setAudioTranscript] = useState('');
  const [batchResults, setBatchResults] = useState([]);
  const [configs, setConfigs] = useState({
    ollama: [],
    quickRespond: [],
    modelPrompts: []
  });
  const [urgencyLevels, setUrgencyLevels] = useState([]);
  const [paginatedData, setPaginatedData] = useState(null);
  const [streamingData, setStreamingData] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  const fileInputRef = useRef(null);
  const batchFileInputRef = useRef(null);

  // Initialize component
  useEffect(() => {
    checkHealth();
    fetchUrgencyLevels();
    fetchConfigs();
  }, []);

  // API Service Functions
  const apiService = {
    baseUrl: 'http://localhost:8000/api/quick-respond',
    
    async request(endpoint, options = {}) {
      const url = `${this.baseUrl}${endpoint}`;
      const config = {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      };
      
      try {
        const response = await fetch(url, config);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
      } catch (error) {
        console.error('API request failed:', error);
        throw error;
      }
    },

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
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    },

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
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return response;
    }
  };

  // Handler Functions
  const checkHealth = async () => {
    try {
      const health = await apiService.request('/health');
      setHealthStatus(health);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealthStatus({ status: 'error', error: error.message });
    }
  };

  const fetchUrgencyLevels = async () => {
    try {
      const levels = await apiService.request('/urgency-levels');
      setUrgencyLevels(levels);
    } catch (error) {
      console.error('Failed to fetch urgency levels:', error);
    }
  };

  const fetchConfigs = async () => {
    try {
      const [ollama, quickRespond, modelPrompts] = await Promise.all([
        apiService.request('/ollama-config'),
        apiService.request('/quick-respond-config'),
        apiService.request('/model-prompts')
      ]);
      
      setConfigs({
        ollama,
        quickRespond,
        modelPrompts
      });
    } catch (error) {
      console.error('Failed to fetch configs:', error);
    }
  };

  const handleFileAnalysis = async (file) => {
    if (!file) return;

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const result = await apiService.uploadFile('/analyze-screenshot', file, {
        meeting_context: meetingContext,
        audio_transcript: audioTranscript
      });
      
      setAnalysisResult(result);
    } catch (error) {
      console.error('Analysis failed:', error);
      setAnalysisResult({ error: error.message });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleStreamingAnalysis = async (file) => {
    if (!file) return;

    setIsStreaming(true);
    setStreamingData('');

    try {
      const response = await apiService.streamAnalysis(file, {
        meeting_context: meetingContext,
        audio_transcript: audioTranscript
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              setStreamingData(prev => prev + JSON.stringify(data, null, 2) + '\n');
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming failed:', error);
      setStreamingData(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  }

  const handleBatchAnalysis = async (files) => {
    if (!files || files.length === 0) return;

    setIsAnalyzing(true);
    setBatchResults([]);

    try {
      const formData = new FormData();
      Array.from(files).forEach(file => {
        formData.append('screenshots', file);
      });
      
      if (meetingContext) {
        formData.append('meeting_context', meetingContext);
      }

      const response = await fetch(`${apiService.baseUrl}/batch-analyze`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setBatchResults(result.batch_results || []);
    } catch (error) {
      console.error('Batch analysis failed:', error);
      setBatchResults([{ error: error.message }]);
    } finally {
      setIsAnalyzing(false);
    }
  }



  const createMeetingStatus = async (statusData) => {
    try {
      const response = await fetch(`${BASE_URL}/api/meeting_status/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(statusData)
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchMeetingStatuses();
      alert('Meeting status created successfully!');
    } catch (error) {
      console.error('Failed to create meeting status:', error);
      alert('Failed to create meeting status');
    }
  };

  const deleteMeetingStatus = async (id) => {
    try {
      const response = await fetch(`${BASE_URL}/api/meeting_status/${id}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchMeetingStatuses();
      alert('Meeting status deleted successfully!');
    } catch (error) {
      console.error('Failed to delete meeting status:', error);
      alert('Failed to delete meeting status');
    }
  };

  // ============================================================================
  // CRUD OPERATIONS - PARTICIPANTS
  // ============================================================================

  const fetchParticipants = async () => {
    try {
      const response = await fetch(`${BASE_URL}/api/participant_info/`);
      const data = await response.json();
      setParticipants(data);
    } catch (error) {
      console.error('Failed to fetch participants:', error);
    }
  };

  const createParticipant = async (participantData) => {
    try {
      const response = await fetch(`${BASE_URL}/api/participant_info/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(participantData)
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchParticipants();
      alert('Participant created successfully!');
    } catch (error) {
      console.error('Failed to create participant:', error);
      alert('Failed to create participant');
    }
  };

  const deleteParticipant = async (id) => {
    try {
      const response = await fetch(`${BASE_URL}/api/participant_info/${id}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchParticipants();
      alert('Participant deleted successfully!');
    } catch (error) {
      console.error('Failed to delete participant:', error);
      alert('Failed to delete participant');
    }
  };

  // ============================================================================
  // CRUD OPERATIONS - SCREEN CONTENT
  // ============================================================================

  const fetchScreenContents = async () => {
    try {
      const response = await fetch(`${BASE_URL}/api/screen_content/`);
      const data = await response.json();
      setScreenContents(data);
    } catch (error) {
      console.error('Failed to fetch screen contents:', error);
    }
  };

  const createScreenContent = async (contentData) => {
    try {
      const response = await fetch(`${BASE_URL}/api/screen_content/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(contentData)
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchScreenContents();
      alert('Screen content created successfully!');
    } catch (error) {
      console.error('Failed to create screen content:', error);
      alert('Failed to create screen content');
    }
  };

  const deleteScreenContent = async (id) => {
    try {
      const response = await fetch(`${BASE_URL}/api/screen_content/${id}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchScreenContents();
      alert('Screen content deleted successfully!');
    } catch (error) {
      console.error('Failed to delete screen content:', error);
      alert('Failed to delete screen content');
    }
  };

  // ============================================================================
  // CRUD OPERATIONS - MEETING METRICS
  // ============================================================================

  const fetchMeetingMetrics = async () => {
    try {
      const response = await fetch(`${BASE_URL}/api/meeting_metrics/`);
      const data = await response.json();
      setMeetingMetrics(data);
    } catch (error) {
      console.error('Failed to fetch meeting metrics:', error);
    }
  };

  const createMeetingMetrics = async (metricsData) => {
    try {
      const response = await fetch(`${BASE_URL}/api/meeting_metrics/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metricsData)
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchMeetingMetrics();
      alert('Meeting metrics created successfully!');
    } catch (error) {
      console.error('Failed to create meeting metrics:', error);
      alert('Failed to create meeting metrics');
    }
  };

  const deleteMeetingMetrics = async (id) => {
    try {
      const response = await fetch(`${BASE_URL}/api/meeting_metrics/${id}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      await fetchMeetingMetrics();
      alert('Meeting metrics deleted successfully!');
    } catch (error) {
      console.error('Failed to delete meeting metrics:', error);
      alert('Failed to delete meeting metrics');
    }
  };

  // ============================================================================
  // PAGINATION
  // ============================================================================

  const fetchPaginatedItems = async (page = 1, pageSize = 10) => {
    try {
      const response = await fetch(`${BASE_URL}/api/items?page=${page}&page_size=${pageSize}`);
      const data = await response.json();
      setPaginatedData(data);
    } catch (error) {
      console.error('Failed to fetch paginated data:', error);
    }
  };

  // ============================================================================
  // FETCH ALL DATA
  // ============================================================================

  const fetchAllData = async () => {
    await Promise.all([
      fetchMeetingStatuses(),
      fetchParticipants(),
      fetchScreenContents(),
      fetchMeetingMetrics()
    ]);
  };

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  const getUrgencyColor = (urgency) => {
    switch (urgency?.toUpperCase()) {
      case 'HIGH': return 'bg-red-500 text-white';
      case 'MEDIUM': return 'bg-yellow-500 text-black';
      case 'LOW': return 'bg-green-500 text-white';
      default: return 'bg-gray-500 text-white';
    }
  };

  // ============================================================================
  // RENDER FUNCTIONS
  // ============================================================================

  const renderHealthStatus = () => (
    <div className="bg-gray-800 rounded-lg p-4 mb-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-200 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2" />
          Service Health
        </h3>
        <button
          onClick={checkHealth}
          className="p-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      
      {healthStatus && (
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="flex items-center">
            {healthStatus.status === 'healthy' ? 
              <CheckCircle className="w-5 h-5 text-green-500 mr-2" /> :
              <XCircle className="w-5 h-5 text-red-500 mr-2" />
            }
            <span className="text-sm text-gray-300">Overall: {healthStatus.status}</span>
          </div>
          
          {healthStatus.services && Object.entries(healthStatus.services).map(([service, status]) => (
            <div key={service} className="flex items-center">
              {status === 'available' ? 
                <CheckCircle className="w-5 h-5 text-green-500 mr-2" /> :
                <XCircle className="w-5 h-5 text-red-500 mr-2" />
              }
              <span className="text-sm text-gray-300">{service}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  // Render Analysis Tab
  const renderAnalysisTab = () => (
    <div className="space-y-6">
      {/* Context Inputs */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-200 mb-4">Meeting Context</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Meeting Context
            </label>
            <textarea
              value={meetingContext}
              onChange={(e) => setMeetingContext(e.target.value)}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              rows="3"
              placeholder="Enter meeting context, agenda, or relevant information..."
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Audio Transcript
            </label>
            <textarea
              value={audioTranscript}
              onChange={(e) => setAudioTranscript(e.target.value)}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              rows="3"
              placeholder="Enter recent audio transcript..."
            />
          </div>
        </div>
        
        <div className="flex gap-2 mt-4">
          <button
            onClick={() => updateMeetingContext({
              meeting_title: "Current Meeting",
              participants: [],
              agenda: meetingContext,
              meeting_type: "general"
            })}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
          >
            Update Context
          </button>
          
          <button
            onClick={clearMeetingContext}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-lg text-white transition-colors"
          >
            Clear Context
          </button>
        </div>
      </div>

      {/* Single Screenshot Analysis */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-200 mb-4 flex items-center">
          <Camera className="w-5 h-5 mr-2" />
          Screenshot Analysis
        </h3>
        
        <div className="flex gap-4 mb-4">
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*"
            onChange={(e) => e.target.files[0] && handleFileAnalysis(e.target.files[0])}
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isAnalyzing}
            className="flex items-center px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
          >
            <Upload className="w-4 h-4 mr-2" />
            {isAnalyzing ? 'Analyzing...' : 'Upload & Analyze'}
          </button>
          
          <button
            onClick={() => {
              const file = fileInputRef.current?.files[0];
              if (file) handleStreamingAnalysis(file);
            }}
            disabled={isStreaming || !fileInputRef.current?.files[0]}
            className="flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
          >
            <Play className="w-4 h-4 mr-2" />
            {isStreaming ? 'Streaming...' : 'Stream Analysis'}
          </button>
        </div>
        
        {analysisResult && (
          <div className="mt-4 bg-gray-700 rounded-lg p-4">
            <h4 className="text-md font-semibold text-gray-200 mb-3">Analysis Results</h4>
            
            {analysisResult.error ? (
              <div className="text-red-400">
                Error: {analysisResult.error}
              </div>
            ) : (
              <div className="space-y-4">
                {/* Key Insights */}
                {analysisResult.key_insights && analysisResult.key_insights.length > 0 && (
                  <div>
                    <h5 className="font-medium text-gray-300 mb-2">Key Insights:</h5>
                    <div className="space-y-2">
                      {analysisResult.key_insights.map((insight, index) => (
                        <div key={index} className="bg-gray-600 rounded p-3">
                          <div className="flex items-start justify-between">
                            <p className="text-gray-200 flex-1">{insight.insight}</p>
                            <span className={`ml-2 px-2 py-1 rounded text-xs font-medium ${getUrgencyColor(insight.urgency)}`}>
                              {insight.urgency}
                            </span>
                          </div>
                          {insight.context && (
                            <p className="text-gray-400 text-sm mt-1">{insight.context}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {analysisResult.full_analysis && (
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <h5 className="font-medium text-gray-300">Full Analysis:</h5>
                      <button
                        onClick={() => handleSimplify(analysisResult.full_analysis)}
                        className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-white text-sm transition-colors"
                      >
                        Simplify
                      </button>
                    </div>
                    <div className="bg-gray-600 rounded p-3">
                      <p className="text-gray-200 whitespace-pre-wrap">{analysisResult.full_analysis}</p>
                    </div>
                  </div>
                )}
                
                {simplifiedResult && (
                  <div>
                    <h5 className="font-medium text-gray-300 mb-2">Simplified Analysis:</h5>
                    <div className="bg-gray-600 rounded p-3">
                      <p className="text-gray-200 whitespace-pre-wrap">{simplifiedResult.simplified_text}</p>
                    </div>
                  </div>
                )}
                
                <div className="text-sm text-gray-400 flex justify-between">
                  <span>Confidence: {((analysisResult.confidence_score || 0) * 100).toFixed(1)}%</span>
                  <span>Session: {analysisResult.session_id}</span>
                </div>
              </div>
            )}
          </div>
        )}
        
        {/* Streaming Results */}
        {streamingData && (
          <div className="mt-4 bg-gray-700 rounded-lg p-4">
            <h4 className="text-md font-semibold text-gray-200 mb-3">Streaming Results</h4>
            <pre className="text-gray-300 text-sm whitespace-pre-wrap max-h-64 overflow-y-auto">
              {streamingData}
            </pre>
          </div>
        )}
      </div>

      {/* Batch Analysis */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-200 mb-4 flex items-center">
          <FileText className="w-5 h-5 mr-2" />
          Batch Analysis
        </h3>
        
        <input
          type="file"
          ref={batchFileInputRef}
          className="hidden"
          accept="image/*"
          multiple
          onChange={(e) => e.target.files.length > 0 && handleBatchAnalysis(e.target.files)}
        />
        
        <button
          onClick={() => batchFileInputRef.current?.click()}
          disabled={isAnalyzing}
          className="flex items-center px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
        >
          <Upload className="w-4 h-4 mr-2" />
          {isAnalyzing ? 'Processing...' : 'Upload Multiple Screenshots'}
        </button>
        
        {batchResults.length > 0 && (
          <div className="mt-4 space-y-3">
            <h4 className="text-md font-semibold text-gray-200">Batch Results</h4>
            {batchResults.map((result, index) => (
              <div key={index} className="bg-gray-700 rounded-lg p-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-300 font-medium">
                    {result.filename || `Screenshot ${index + 1}`}
                  </span>
                </div>
                
                {result.error ? (
                  <p className="text-red-400">Error: {result.error}</p>
                ) : result.analysis && (
                  <div className="text-gray-200 text-sm">
                    <p className="mb-2">{result.analysis.full_analysis}</p>
                    {result.analysis.key_insights && result.analysis.key_insights.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {result.analysis.key_insights.map((insight, i) => (
                          <span key={i} className="px-2 py-1 bg-gray-600 rounded text-xs">
                            {insight.insight}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const renderDataTab = () => (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-200">Meeting Statuses</h3>
          <button
            onClick={fetchMeetingStatuses}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
          >
            Refresh
          </button>
        </div>
        <div className="space-y-2">
          {meetingStatuses.length === 0 ? (
            <p className="text-gray-400">No meeting statuses found</p>
          ) : (
            meetingStatuses.map((status, index) => (
              <div key={index} className="bg-gray-700 rounded p-3 flex justify-between items-center">
                <span className="text-gray-200">{status.status || `Status ${status.id}`}</span>
                <button
                  onClick={() => deleteMeetingStatus(status.id)}
                  className="p-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-200">Participants</h3>
          <button
            onClick={fetchParticipants}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
          >
            Refresh
          </button>
        </div>
        <div className="space-y-2">
          {participants.length === 0 ? (
            <p className="text-gray-400">No participants found</p>
          ) : (
            participants.map((participant, index) => (
              <div key={index} className="bg-gray-700 rounded p-3 flex justify-between items-center">
                <span className="text-gray-200">{participant.name || `Participant ${participant.id}`}</span>
                <button
                  onClick={() => deleteParticipant(participant.id)}
                  className="p-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-200">Screen Contents</h3>
          <button
            onClick={fetchScreenContents}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
          >
            Refresh
          </button>
        </div>
        <div className="space-y-2">
          {screenContents.length === 0 ? (
            <p className="text-gray-400">No screen contents found</p>
          ) : (
            screenContents.map((content, index) => (
              <div key={index} className="bg-gray-700 rounded p-3 flex justify-between items-center">
                <span className="text-gray-200">{content.content || `Content ${content.id}`}</span>
                <button
                  onClick={() => deleteScreenContent(content.id)}
                  className="p-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-200">Meeting Metrics</h3>
          <button
            onClick={fetchMeetingMetrics}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
          >
            Refresh
          </button>
        </div>
        <div className="space-y-2">
          {meetingMetrics.length === 0 ? (
            <p className="text-gray-400">No meeting metrics found</p>
          ) : (
            meetingMetrics.map((metric, index) => (
              <div key={index} className="bg-gray-700 rounded p-3 flex justify-between items-center">
                <span className="text-gray-200">Metrics {metric.id}</span>
                <button
                  onClick={() => deleteMeetingMetrics(metric.id)}
                  className="p-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-200 mb-4">Urgency Levels</h3>
        <div className="flex flex-wrap gap-2">
          {urgencyLevels.map((level, index) => (
            <span
              key={index}
              className={`px-3 py-2 rounded-full text-sm font-medium ${getUrgencyColor(level)}`}
            >
              {level}
            </span>
          ))}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-200">Paginated Items</h3>
          <button
            onClick={() => fetchPaginatedItems()}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
          >
            Load Items
          </button>
        </div>
        
        {paginatedData && (
          <div>
            <div className="mb-4">
              <p className="text-gray-300">
                Page {paginatedData.page} of {Math.ceil(paginatedData.total / paginatedData.page_size)}
                ({paginatedData.total} total items)
              </p>
            </div>
            
            <div className="bg-gray-700 rounded p-3 max-h-60 overflow-y-auto mb-4">
              {paginatedData.items.map((item, index) => (
                <div key={index} className="text-gray-300 text-sm py-1 border-b border-gray-600 last:border-b-0">
                  {typeof item === 'string' ? item : JSON.stringify(item)}
                </div>
              ))}
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={() => fetchPaginatedItems(paginatedData.page - 1)}
                disabled={paginatedData.page <= 1}
                className="px-3 py-1 bg-gray-600 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded text-white transition-colors"
              >
                Previous
              </button>
              
              <button
                onClick={() => fetchPaginatedItems(paginatedData.page + 1)}
                disabled={paginatedData.page >= Math.ceil(paginatedData.total / paginatedData.page_size)}
                className="px-3 py-1 bg-gray-600 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed rounded text-white transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif', backgroundColor: '#1E1E2F', minHeight: '100vh', color: '#F8FAFC' }}>
      <div className="container mx-auto px-4 py-8" style={{ maxWidth: '1400px' }}>
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Meeting Analysis Dashboard</h1>
          <p className="text-gray-400">AI-powered meeting screenshot analysis and insights</p>
        </div>

        {renderHealthStatus()}

        <div className="flex justify-center mb-8">
          <div className="bg-gray-800 rounded-lg p-1 flex space-x-1">
            <button
              onClick={() => setActiveTab('analyze')}
              className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'analyze'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Analysis
            </button>
            
            <button
              onClick={() => setActiveTab('data')}
              className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'data'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Data Management
            </button>
          </div>
        </div>

        <div className="max-w-6xl mx-auto">
          {activeTab === 'analyze' && renderAnalysisTab()}
          {activeTab === 'data' && renderDataTab()}
        </div>
      </div>
    </div>
  );
};

export default MeetingAnalysisDashboard;