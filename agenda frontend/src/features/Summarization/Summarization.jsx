import React, { useState, useCallback, useRef } from 'react';
import { Upload, Mic, FileAudio, Loader2, Play, Pause, Download, Trash2, Eye, Clock, Users, Target } from 'lucide-react';
import { summarizationService } from './summarizationService';

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

  // File upload handler
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
        ...uploadResult,
        name: file.name,
        file: file
      });
    } catch (err) {
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

  // Analyze meeting
  const analyzeMeeting = useCallback(async () => {
    if (!uploadedFile) {
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

  // Load user summaries
  const loadUserSummaries = useCallback(async () => {
    try {
      const userSummaries = await summarizationService.getUserSummaries();
      setSummaries(userSummaries);
    } catch (err) {
      setError('Failed to load summaries: ' + err.message);
    }
  }, []);

  // Delete summary
  const deleteSummary = useCallback(async (meetingId) => {
    try {
      await summarizationService.deleteSummary(meetingId);
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
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                    activeTab === tab.id 
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
                          {formatFileSize(uploadedFile.file_size)}
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
                        <span className="text-gray-300">{point.point}</span>
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
                            <p className="font-medium" style={{ color: '#F8FAFC' }}>{item.task}</p>
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
                              <span>{new Date(summary.created_at).toLocaleDateString()}</span>
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
                            {summary.action_items.slice(0, 3).map((item, index) => (
                              <span key={index} className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-300">
                                {item.task.substring(0, 50)}{item.task.length > 50 ? '...' : ''}
                              </span>
                            ))}
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