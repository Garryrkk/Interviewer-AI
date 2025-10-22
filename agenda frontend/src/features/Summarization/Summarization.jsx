import React, { useState, useCallback, useRef } from 'react';
import { Upload, Mic, FileAudio, Loader2, Play, Pause, Download, Trash2, Eye, Clock, Users, Target } from 'lucide-react';
import { API_BASE_URL, callEndpoint } from "../../services/apiConfig";
// api-configuration.js
const ENV = "development"; // or "production"

const BASE_URLS = {
  development: "http://127.0.0.1:8000",
  production: "https://api.myapp.com",
};

export const API_BASE_URL = BASE_URLS[ENV];
  
const summarizationService = {
  // Audio Upload Endpoints
  uploadAudio: async (file, meetingId = null) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const params = new URLSearchParams({ user_id: USER_ID });
    if (meetingId) params.append('meeting_id', meetingId);

    const response = await fetch(`${API_BASE_URL}/audio/upload?${params}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail || error.error || 'Upload failed');
    }

    const result = await response.json();
    // Handle both wrapped and direct responses
    return result.data || result;
  },

  uploadMeetingAudio: async (file, meetingId = null) => {
    const formData = new FormData();
    formData.append('audio_file', file);
    if (meetingId) formData.append('meeting_id', meetingId);

    const response = await fetch(`${API_BASE_URL}/summarization/upload-audio`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  },

  // Meeting Analysis Endpoints
  analyzeMeeting: async (audioFilePath, meetingContext = null, analysisType = 'post_meeting') => {
    const params = new URLSearchParams({
      audio_file_path: audioFilePath,
      user_id: USER_ID,
      analysis_type: analysisType
    });
    
    if (meetingContext) params.append('meeting_context', meetingContext);

    const response = await fetch(`${API_BASE_URL}/meetings/analyze?${params}`, {
      method: 'POST'
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Analysis failed' }));
      throw new Error(error.detail || error.error || 'Analysis failed');
    }

    const result = await response.json();
    return result.data || result;
  },

  analyzeMeetingAudio: async (audioFilePath, meetingContext = null, analysisType = 'post_meeting') => {
    const response = await fetch(`${API_BASE_URL}/summarization/analyze-meeting`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_file_path: audioFilePath,
        meeting_context: meetingContext,
        analysis_type: analysisType
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Analysis failed');
    }

    return response.json();
  },

  // Summary Generation Endpoints
  generateSummary: async (content, summaryType = 'detailed', includeActionItems = true, meetingId = null) => {
    const params = new URLSearchParams({
      content: content,
      summary_type: summaryType,
      user_id: USER_ID,
      include_action_items: includeActionItems.toString()
    });
    
    if (meetingId) params.append('meeting_id', meetingId);

    const response = await fetch(`${API_BASE_URL}/summaries/generate?${params}`, {
      method: 'POST'
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Summary generation failed' }));
      throw new Error(error.detail || error.error || 'Summary generation failed');
    }

    const result = await response.json();
    return result.data || result;
  },

  createSummary: async (content, summaryType = 'detailed', meetingId = null, includeActionItems = true) => {
    const response = await fetch(`${API_BASE_URL}/summarization/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: content,
        summary_type: summaryType,
        meeting_id: meetingId,
        include_action_items: includeActionItems
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Summary creation failed');
    }

    return response.json();
  },

  // Real-time Analysis Endpoints
  realTimeAnalysis: async (audioChunkPath, meetingContext = null) => {
    const params = new URLSearchParams({
      audio_chunk_path: audioChunkPath,
      user_id: USER_ID
    });
    
    if (meetingContext) params.append('meeting_context', meetingContext);

    const response = await fetch(`${API_BASE_URL}/meetings/real-time?${params}`, {
      method: 'POST'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Real-time analysis failed');
    }

    return response.json();
  },

  realTimeMeetingAnalysis: async (audioFilePath, meetingContext = null) => {
    const response = await fetch(`${API_BASE_URL}/summarization/real-time-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_file_path: audioFilePath,
        meeting_context: meetingContext
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Real-time analysis failed');
    }

    return response.json();
  },

  // Get Summary Endpoints
  getMeetingSummary: async (meetingId) => {
    const params = new URLSearchParams({ user_id: USER_ID });
    const response = await fetch(`${API_BASE_URL}/summaries/${meetingId}?${params}`);

    if (!response.ok) {
      if (response.status === 404) return null;
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch summary');
    }

    const result = await response.json();
    return result.data;
  },

  getSummaryByMeetingId: async (meetingId) => {
    const response = await fetch(`${API_BASE_URL}/summarization/meeting/${meetingId}/summary`);

    if (!response.ok) {
      if (response.status === 404) return null;
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch summary');
    }

    return response.json();
  },

  // User Summaries Endpoints
  getUserSummaries: async (limit = 10, offset = 0) => {
    const params = new URLSearchParams({
      user_id: USER_ID,
      limit: limit.toString(),
      offset: offset.toString()
    });

    const response = await fetch(`${API_BASE_URL}/summaries/user?${params}`);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to fetch summaries' }));
      throw new Error(error.detail || error.error || 'Failed to fetch summaries');
    }

    const result = await response.json();
    return result.data || result;
  },

  getAllUserSummaries: async (limit = 10, offset = 0) => {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString()
    });

    const response = await fetch(`${API_BASE_URL}/summarization/user/summaries?${params}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch summaries');
    }

    return response.json();
  },

  // Delete Summary Endpoints
  deleteSummary: async (meetingId) => {
    const params = new URLSearchParams({ user_id: USER_ID });
    const response = await fetch(`${API_BASE_URL}/summaries/${meetingId}?${params}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete summary');
    }

    return true;
  },

  deleteMeetingSummary: async (meetingId) => {
    const response = await fetch(`${API_BASE_URL}/summarization/meeting/${meetingId}/summary`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete summary');
    }

    return response.json();
  },

  // Batch Summarization
  batchSummarize: async (meetingIds) => {
    const response = await fetch(`${API_BASE_URL}/summarization/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        meeting_ids: meetingIds
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Batch summarization failed');
    }

    return response.json();
  },

  // Action Items CRUD
  createActionItem: async (item) => {
    const response = await fetch(`${API_BASE_URL}/actionitems/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(item)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create action item');
    }

    return response.json();
  },

  getActionItems: async () => {
    const response = await fetch(`${API_BASE_URL}/actionitems/`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch action items');
    }

    return response.json();
  },

  getActionItem: async (itemId) => {
    const response = await fetch(`${API_BASE_URL}/actionitems/${itemId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch action item');
    }

    return response.json();
  },

  updateActionItem: async (itemId, item) => {
    const response = await fetch(`${API_BASE_URL}/actionitems/${itemId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(item)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update action item');
    }

    return response.json();
  },

  deleteActionItem: async (itemId) => {
    const response = await fetch(`${API_BASE_URL}/actionitems/${itemId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete action item');
    }

    return response.json();
  },

  // Key Points CRUD
  createKeyPoint: async (point) => {
    const response = await fetch(`${API_BASE_URL}/keypoints/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(point)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create key point');
    }

    return response.json();
  },

  getKeyPoints: async () => {
    const response = await fetch(`${API_BASE_URL}/keypoints/`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch key points');
    }

    return response.json();
  },

  getKeyPoint: async (pointId) => {
    const response = await fetch(`${API_BASE_URL}/keypoints/${pointId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch key point');
    }

    return response.json();
  },

  updateKeyPoint: async (pointId, point) => {
    const response = await fetch(`${API_BASE_URL}/keypoints/${pointId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(point)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update key point');
    }

    return response.json();
  },

  deleteKeyPoint: async (pointId) => {
    const response = await fetch(`${API_BASE_URL}/keypoints/${pointId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete key point');
    }

    return response.json();
  },

  // Summary Types CRUD
  createSummaryType: async (summary) => {
    const response = await fetch(`${API_BASE_URL}/summarytypes/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(summary)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create summary type');
    }

    return response.json();
  },

  getSummaryTypes: async () => {
    const response = await fetch(`${API_BASE_URL}/summarytypes/`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch summary types');
    }

    return response.json();
  },

  getSummaryType: async (summaryId) => {
    const response = await fetch(`${API_BASE_URL}/summarytypes/${summaryId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch summary type');
    }

    return response.json();
  },

  updateSummaryType: async (summaryId, summary) => {
    const response = await fetch(`${API_BASE_URL}/summarytypes/${summaryId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(summary)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update summary type');
    }

    return response.json();
  },

  deleteSummaryType: async (summaryId) => {
    const response = await fetch(`${API_BASE_URL}/summarytypes/${summaryId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete summary type');
    }

    return response.json();
  },

  // Summary CRUD
  createSummaryRecord: async (summary) => {
    const response = await fetch(`${API_BASE_URL}/summary`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(summary)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create summary');
    }

    return response.json();
  },

  getSummaries: async () => {
    const response = await fetch(`${API_BASE_URL}/summary`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch summaries');
    }

    return response.json();
  },

  // Analysis Types CRUD
  createAnalysisType: async (analysis) => {
    const response = await fetch(`${API_BASE_URL}/analysistypes/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(analysis)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create analysis type');
    }

    return response.json();
  },

  getAnalysisTypes: async () => {
    const response = await fetch(`${API_BASE_URL}/analysistypes/`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch analysis types');
    }

    return response.json();
  },

  getAnalysisType: async (analysisId) => {
    const response = await fetch(`${API_BASE_URL}/analysistypes/${analysisId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch analysis type');
    }

    return response.json();
  },

  updateAnalysisType: async (analysisId, analysis) => {
    const response = await fetch(`${API_BASE_URL}/analysistypes/${analysisId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(analysis)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update analysis type');
    }

    return response.json();
  },

  deleteAnalysisType: async (analysisId) => {
    const response = await fetch(`${API_BASE_URL}/analysistypes/${analysisId}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete analysis type');
    }

    return response.json();
  },

  // Meeting Context
  createMeeting: async (meeting) => {
    const response = await fetch(`${API_BASE_URL}/meeting`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(meeting)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create meeting');
    }

    return response.json();
  },

  getMeetings: async () => {
    const response = await fetch(`${API_BASE_URL}/meeting`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch meetings');
    }

    return response.json();
  },

  // LLAVA Analysis Config
  createLLAVAConfig: async (config) => {
    const response = await fetch(`${API_BASE_URL}/llava-config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create LLAVA config');
    }

    return response.json();
  },

  getLLAVAConfigs: async () => {
    const response = await fetch(`${API_BASE_URL}/llava-config`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch LLAVA configs');
    }

    return response.json();
  },

  // Real-time Updates
  createRealtimeUpdate: async (update) => {
    const response = await fetch(`${API_BASE_URL}/realtime-update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(update)
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to create realtime update');
    }

    return response.json();
  },

  getRealtimeUpdates: async () => {
    const response = await fetch(`${API_BASE_URL}/realtime-update`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to fetch realtime updates');
    }

    return response.json();
  }
};

const BASE_URL = "http://localhost:8000/api/v1/summarization";
const MeetingSummarization = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [summaries, setSummaries] = useState([]);
  const [selectedSummary, setSelectedSummary] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [meetingContext, setMeetingContext] = useState('');
  const [summaryType, setSummaryType] = useState('brief');
  const [includeActionItems, setIncludeActionItems] = useState(true);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordingIntervalRef = useRef(null);

  // Handle file upload with proper error handling
  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('audio/')) {
      setError('Please select an audio file');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const uploadResult = await summarizationService.uploadAudio(file);
      setUploadedFile({
        name: file.name,
        file_path: uploadResult.file_path || uploadResult.audio_file_path,
        file_size: uploadResult.file_size || file.size,
        duration: uploadResult.duration,
        meeting_id: uploadResult.meeting_id,
        file: file
      });
      
      console.log('Upload successful:', uploadResult);
    } catch (err) {
      console.error('Upload error:', err);
      setError('Failed to upload audio file: ' + err.message);
    } finally {
      setIsUploading(false);
    }
  }, []);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], `recording-${Date.now()}.wav`, { type: 'audio/wav' });

        setIsUploading(true);
        try {
          const uploadResult = await summarizationService.uploadAudio(audioFile);
          setUploadedFile({
            ...uploadResult,
            name: audioFile.name,
            file: audioFile
          });
        } catch (err) {
          setError('Failed to process recording: ' + err.message);
        } finally {
          setIsUploading(false);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      recordingIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      setError('Failed to access microphone: ' + err.message);
    }
  }, []);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
      clearInterval(recordingIntervalRef.current);
    }
  }, [isRecording]);

  // Analyze meeting with better error handling and data validation
  const analyzeMeeting = useCallback(async () => {
    if (!uploadedFile || !uploadedFile.file_path) {
      setError('Please upload an audio file first');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      const analysisResult = await summarizationService.analyzeMeeting({
        audio_file_path: uploadedFile.file_path,
        meeting_context: meetingContext,
        analysis_type: 'post_meeting',
        include_sentiment: true,
        include_speakers: true
      });

      setAnalysisResult(analysisResult);
      setActiveTab('results');
    } catch (err) {
      console.error('Analysis error:', err);
      setError('Failed to analyze meeting: ' + err.message);
    } finally {
      setIsAnalyzing(false);
    }
  }, [uploadedFile, meetingContext]);

  // Generate summary
  const generateSummary = useCallback(async (content) => {
    try {
      const summaryResult = await summarizationService.generateSummary({
        content: content,
        summary_type: summaryType,
        include_action_items: includeActionItems,
        meeting_id: uploadedFile?.meeting_id
      });

      setSummaries(prev => [summaryResult, ...prev]);
      return summaryResult;
    } catch (err) {
      setError('Failed to generate summary: ' + err.message);
    }
  }, [summaryType, includeActionItems, uploadedFile]);

  // Load user summaries with better error handling
  const loadUserSummaries = useCallback(async () => {
    try {
      const userSummaries = await summarizationService.getUserSummaries();
      setSummaries(userSummaries);
    } catch (err) {
      console.error('Load summaries error:', err);
      setError('Failed to load summaries: ' + err.message);
    }
  }, []);

  // Delete summary
  const deleteSummary = useCallback(async (meetingId) => {
    try {
      const res = await fetch(`${BASE_URL}/meeting/${meetingId}/summary`, {
        method: "DELETE",
      });
      if (!res.ok) throw new Error(await res.text());
      setSummaries(prev => prev.filter(s => s.meeting_id !== meetingId));
    } catch (err) {
      setError('Failed to delete summary: ' + err.message);
    }
  }, []);

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Format file size
  const formatFileSize = (bytes) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  React.useEffect(() => {
    loadUserSummaries();
  }, [loadUserSummaries]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6" style={{ backgroundColor: '#1E1E2F', fontFamily: 'Roboto, sans-serif' }}>
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-100 mb-2" style={{ color: '#F8FAFC' }}>
            Meeting Summarization
          </h1>
          <p className="text-gray-400">Upload audio files or record meetings to generate AI-powered summaries and insights</p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-700 rounded-lg">
            <p className="text-red-200">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-red-300 hover:text-red-100 text-sm underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="flex space-x-1 bg-gray-800 p-1 rounded-lg" style={{ backgroundColor: '#2A2A3E' }}>
            {[
              { id: 'upload', label: 'Upload Audio', icon: Upload },
              { id: 'record', label: 'Record Meeting', icon: Mic },
              { id: 'results', label: 'Analysis Results', icon: FileAudio },
              { id: 'summaries', label: 'My Summaries', icon: Eye }
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${activeTab === tab.id
                    ? 'text-white shadow-md'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-700'
                    }`}
                  style={{
                    backgroundColor: activeTab === tab.id ? '#8F74D4' : 'transparent'
                  }}
                >
                  <Icon className="w-4 h-4" />
                  <span className="font-medium">{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="space-y-6">
              <div className="bg-gray-800 p-6 rounded-lg border border-gray-700" style={{ backgroundColor: '#2A2A3E' }}>
                <h3 className="text-lg font-semibold mb-4" style={{ color: '#F8FAFC' }}>Upload Meeting Audio</h3>

                <div className="space-y-4">
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-gray-500 cursor-pointer transition-colors"
                  >
                    <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                    <p className="text-lg font-medium mb-2" style={{ color: '#F8FAFC' }}>
                      Click to upload audio file
                    </p>
                    <p className="text-gray-400">Supports MP3, WAV, M4A, and other audio formats</p>
                  </div>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />

                  {/* Meeting Context */}
                  <div>
                    <label className="block text-sm font-medium mb-2" style={{ color: '#F8FAFC' }}>
                      Meeting Context (Optional)
                    </label>
                    <textarea
                      value={meetingContext}
                      onChange={(e) => setMeetingContext(e.target.value)}
                      placeholder="Provide context about the meeting (e.g., standup, project review, client meeting...)"
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 placeholder-gray-400 focus:outline-none focus:border-purple-400"
                      rows={3}
                    />
                  </div>

                  {/* Summary Options */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium mb-2" style={{ color: '#F8FAFC' }}>
                        Summary Type
                      </label>
                      <select
                        value={summaryType}
                        onChange={(e) => setSummaryType(e.target.value)}
                        className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:border-purple-400"
                      >
                        <option value="brief">Brief Summary</option>
                        <option value="detailed">Detailed Summary</option>
                        <option value="action_items">Action Items Focus</option>
                        <option value="key_points">Key Points</option>
                      </select>
                    </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="includeActionItems"
                        checked={includeActionItems}
                        onChange={(e) => setIncludeActionItems(e.target.checked)}
                        className="mr-2"
                      />
                      <label htmlFor="includeActionItems" className="text-sm" style={{ color: '#F8FAFC' }}>
                        Include Action Items
                      </label>
                    </div>
                  </div>
                </div>
              </div>

              {/* Uploaded File Info */}
              {uploadedFile && (
                <div className="bg-gray-800 p-6 rounded-lg border border-gray-700" style={{ backgroundColor: '#2A2A3E' }}>
                  <h3 className="text-lg font-semibold mb-4" style={{ color: '#F8FAFC' }}>Uploaded File</h3>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <FileAudio className="w-8 h-8 text-purple-400" />
                      <div>
                        <p className="font-medium" style={{ color: '#F8FAFC' }}>{uploadedFile.name}</p>
                        <p className="text-sm text-gray-400">
                          {uploadedFile.duration && `${formatTime(Math.floor(uploadedFile.duration))} • `}
                          {uploadedFile.file_size && formatFileSize(uploadedFile.file_size)}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={analyzeMeeting}
                      disabled={isAnalyzing}
                      className="px-6 py-2 rounded-lg font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{
                        backgroundColor: '#8F74D4',
                        color: '#F8FAFC'
                      }}
                    >
                      {isAnalyzing ? (
                        <div className="flex items-center space-x-2">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          <span>Analyzing...</span>
                        </div>
                      ) : (
                        'Analyze Meeting'
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Record Tab */}
          {activeTab === 'record' && (
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700" style={{ backgroundColor: '#2A2A3E' }}>
              <h3 className="text-lg font-semibold mb-4" style={{ color: '#F8FAFC' }}>Record Meeting</h3>

              <div className="text-center space-y-6">
                <div className="flex items-center justify-center">
                  <div className={`w-32 h-32 rounded-full flex items-center justify-center ${isRecording ? 'bg-red-600' : 'bg-gray-700'} transition-colors`}>
                    <Mic className={`w-16 h-16 ${isRecording ? 'text-white animate-pulse' : 'text-gray-400'}`} />
                  </div>
                </div>

                {isRecording && (
                  <div className="text-center">
                    <p className="text-2xl font-mono font-bold text-red-400">
                      {formatTime(recordingTime)}
                    </p>
                    <p className="text-gray-400">Recording in progress...</p>
                  </div>
                )}

                <div className="flex justify-center space-x-4">
                  {!isRecording ? (
                    <button
                      onClick={startRecording}
                      className="px-8 py-3 rounded-lg font-medium transition-all"
                      style={{
                        backgroundColor: '#8F74D4',
                        color: '#F8FAFC'
                      }}
                    >
                      <div className="flex items-center space-x-2">
                        <Mic className="w-5 h-5" />
                        <span>Start Recording</span>
                      </div>
                    </button>
                  ) : (
                    <button
                      onClick={stopRecording}
                      className="px-8 py-3 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 transition-all"
                    >
                      <div className="flex items-center space-x-2">
                        <Pause className="w-5 h-5" />
                        <span>Stop Recording</span>
                      </div>
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && analysisResult && (
            <div className="space-y-6">
              {/* Summary Card */}
              <div className="bg-gray-800 p-6 rounded-lg border border-gray-700" style={{ backgroundColor: '#2A2A3E' }}>
                <h3 className="text-lg font-semibold mb-4" style={{ color: '#F8FAFC' }}>Meeting Summary</h3>
                <p className="text-gray-300 leading-relaxed">{analysisResult.summary}</p>
              </div>

              {/* Key Points */}
              {analysisResult.key_points?.length > 0 && (
                <div className="bg-gray-800 p-6 rounded-lg border border-gray-700" style={{ backgroundColor: '#2A2A3E' }}>
                  <h3 className="text-lg font-semibold mb-4" style={{ color: '#F8FAFC' }}>Key Points</h3>
                  <ul className="space-y-2">
                    {analysisResult.key_points.map((point, index) => (
                      <li key={index} className="flex items-start space-x-3">
                        <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 flex-shrink-0"></div>
                        <span className="text-gray-300">{typeof point === 'string' ? point : point.point}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Action Items */}
              {analysisResult.action_items?.length > 0 && (
                <div className="bg-gray-800 p-6 rounded-lg border border-gray-700" style={{ backgroundColor: '#2A2A3E' }}>
                  <h3 className="text-lg font-semibold mb-4" style={{ color: '#F8FAFC' }}>Action Items</h3>
                  <div className="space-y-3">
                    {analysisResult.action_items.map((item, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <Target className="w-5 h-5 text-purple-400" />
                          <div>
                            <p className="font-medium" style={{ color: '#F8FAFC' }}>{typeof item === 'string' ? item : item.task}</p>
                            {item.assignee && (
                              <p className="text-sm text-gray-400">Assigned to: {item.assignee}</p>
                            )}
                          </div>
                        </div>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          item.priority === 'high' ? 'bg-red-900 text-red-200' :
                          item.priority === 'medium' ? 'bg-yellow-900 text-yellow-200' :
                          'bg-green-900 text-green-200'
                        }`}>
                          {item.priority}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {analysisResult.recommendations?.length > 0 && (
                <div className="bg-gray-800 p-6 rounded-lg border border-gray-700" style={{ backgroundColor: '#2A2A3E' }}>
                  <h3 className="text-lg font-semibold mb-4" style={{ color: '#F8FAFC' }}>AI Recommendations</h3>
                  <ul className="space-y-2">
                    {analysisResult.recommendations.map((rec, index) => (
                      <li key={index} className="text-gray-300">{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Summaries Tab */}
          {activeTab === 'summaries' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold" style={{ color: '#F8FAFC' }}>My Summaries</h3>
                <button
                  onClick={loadUserSummaries}
                  className="px-4 py-2 rounded-lg font-medium transition-all"
                  style={{
                    backgroundColor: '#8F74D4',
                    color: '#F8FAFC'
                  }}
                >
                  Refresh
                </button>
              </div>

              {summaries.length === 0 ? (
                <div className="text-center py-12">
                  <FileAudio className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                  <p className="text-gray-400">No summaries found</p>
                  <p className="text-gray-500 text-sm">Upload and analyze meetings to see summaries here</p>
                </div>
              ) : (
                <div className="grid gap-6">
                  {summaries.map((summary) => (
                    <div key={summary.summary_id} className="bg-gray-800 p-6 rounded-lg border border-gray-700" style={{ backgroundColor: '#2A2A3E' }}>
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <h4 className="font-semibold text-lg mb-2" style={{ color: '#F8FAFC' }}>
                            Meeting Summary
                          </h4>
                          <div className="flex items-center space-x-4 text-sm text-gray-400 mb-3">
                            <div className="flex items-center space-x-1">
                              <Clock className="w-4 h-4" />
                              <span>{new Date(summary.created_at || Date.now()).toLocaleDateString()}</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <FileAudio className="w-4 h-4" />
                              <span className="capitalize">{summary.summary_type}</span>
                            </div>
                          </div>
                          <p className="text-gray-300 line-clamp-3">{summary.summary_text}</p>
                        </div>
                        <div className="flex space-x-2 ml-4">
                          <button
                            onClick={() => setSelectedSummary(summary)}
                            className="p-2 text-gray-400 hover:text-gray-200 transition-colors"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => deleteSummary(summary.meeting_id)}
                            className="p-2 text-gray-400 hover:text-red-400 transition-colors"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>

                      {summary.action_items?.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-gray-700">
                          <p className="text-sm font-medium mb-2" style={{ color: '#F8FAFC' }}>
                            Action Items ({summary.action_items.length})
                          </p>
                          <div className="flex flex-wrap gap-2">
                            {summary.action_items.slice(0, 3).map((item, index) => {
                              const taskText = typeof item === 'string' ? item : item.task;
                              return (
                                <span key={index} className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300">
                                  {taskText.substring(0, 50)}{taskText.length > 50 ? '...' : ''}
                                </span>
                              );
                            })}
                            {summary.action_items.length > 3 && (
                              <span className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">
                                +{summary.action_items.length - 3} more
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Loading States */}
        {(isUploading || isAnalyzing) && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 flex items-center space-x-4">
              <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
              <div>
                <p className="font-medium" style={{ color: '#F8FAFC' }}>
                  {isUploading ? 'Uploading file...' : 'Analyzing meeting...'}
                </p>
                <p className="text-sm text-gray-400">This may take a few moments</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MeetingSummarization;