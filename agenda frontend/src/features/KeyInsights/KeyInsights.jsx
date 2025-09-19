import React, { useState, useEffect } from 'react';
import { Upload, FileText, Brain, TrendingUp, Clock, Trash2, RefreshCw, Users, AlertCircle, CheckCircle } from 'lucide-react';

const KeyInsightsDashboard = () => {
  const [meetingContext, setMeetingContext] = useState('');
  const [meetingId, setMeetingId] = useState('');
  const [participants, setParticipants] = useState('');
  const [analysisFocus, setAnalysisFocus] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [statusPolling, setStatusPolling] = useState(false);
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

  // Simplify Insights
  const simplifyInsights = async () => {
    if (!insights) {
      setError('No insights to simplify');
      return;
    }

    setLoading(true);
    setError('');

    const requestData = {
      original_insight_id: insights.insight_id,
      original_insights: insights.key_insights,
      original_tips: insights.situation_tips,
      simplification_level: simplificationLevel
    };

    try {
      const response = await fetch(`${BASE_URL}/simplify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setInsights(prev => ({
        ...prev,
        simplified_insights: data.simplified_insights,
        simplified_tips: data.simplified_tips,
        simplification_level: data.simplification_level
      }));
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

  // Get Insights History
  const getInsightsHistory = async () => {
    if (!meetingId) {
      setError('Please provide a meeting ID');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${BASE_URL}/history/${meetingId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setInsightsHistory(data.insights_history);
    } catch (err) {
      setError(`Failed to get insights history: ${err.message}`);
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
    setError('');

    try {
      const response = await fetch(`${BASE_URL}/batch-analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          meeting_contexts: validContexts,
          meeting_ids: validIds
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setInsights(data);
    } catch (err) {
      setError(`Failed to perform batch analysis: ${err.message}`);
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
      setSelectedFile(file);
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
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                  Meeting ID
                </label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <input
                    type="text"
                    value={meetingId}
                    onChange={(e) => setMeetingId(e.target.value)}
                    placeholder="Enter meeting ID"
                    style={{
                      flex: 1,
                      padding: '12px',
                      backgroundColor: '#1F2937',
                      border: '1px solid #4B5563',
                      borderRadius: '8px',
                      color: '#F8FAFC',
                      fontSize: '14px'
                    }}
                  />
                  <button
                    onClick={generateMeetingId}
                    type="button"
                    style={{
                      backgroundColor: '#6B7280',
                      color: '#F8FAFC',
                      border: 'none',
                      padding: '12px',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      fontSize: '12px',
                      fontWeight: '500',
                      whiteSpace: 'nowrap'
                    }}
                  >
                    Generate
                  </button>
                </div>
              </div>

              <div>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                  Participants (comma-separated)
                </label>
                <input
                  type="text"
                  value={participants}
                  onChange={(e) => setParticipants(e.target.value)}
                  placeholder="John, Sarah, Mike"
                  style={{
                    width: '100%',
                    padding: '12px',
                    backgroundColor: '#1F2937',
                    border: '1px solid #4B5563',
                    borderRadius: '8px',
                    color: '#F8FAFC',
                    fontSize: '14px'
                  }}
                />
              </div>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                Analysis Focus
              </label>
              <input
                type="text"
                value={analysisFocus}
                onChange={(e) => setAnalysisFocus(e.target.value)}
                placeholder="e.g., Decision making, Team dynamics, Action items"
                style={{
                  width: '100%',
                  padding: '12px',
                  backgroundColor: '#1F2937',
                  border: '1px solid #4B5563',
                  borderRadius: '8px',
                  color: '#F8FAFC',
                  fontSize: '14px'
                }}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                Meeting Context/Transcript
              </label>
              <textarea
                value={meetingContext}
                onChange={(e) => setMeetingContext(e.target.value)}
                placeholder="Enter meeting context, transcript, or discussion points..."
                rows={6}
                style={{
                  width: '100%',
                  padding: '12px',
                  backgroundColor: '#1F2937',
                  border: '1px solid #4B5563',
                  borderRadius: '8px',
                  color: '#F8FAFC',
                  fontSize: '14px',
                  resize: 'vertical'
                }}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
                Upload Image (for facial expression analysis)
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                style={{
                  width: '100%',
                  padding: '12px',
                  backgroundColor: '#1F2937',
                  border: '1px solid #4B5563',
                  borderRadius: '8px',
                  color: '#F8FAFC',
                  fontSize: '14px'
                }}
              />
              {selectedFile && (
                <p style={{ fontSize: '14px', color: '#9CA3AF', marginTop: '4px' }}>
                  Selected: {selectedFile.name}
                </p>
              )}
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
            ))}

            <div style={{ display: 'flex', gap: '12px', marginBottom: '20px' }}>
              <button
                onClick={addBatchInput}
                style={{
                  backgroundColor: '#374151',
                  color: '#F8FAFC',
                  border: '1px solid #8F74D4',
                  padding: '8px 16px',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: '500',
                  fontSize: '14px'
                }}
              >
                Add Meeting
              </button>
            </div>

            <button
              onClick={batchAnalyzeInsights}
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
                opacity: loading ? 0.7 : 1
              }}
            >
              {loading ? (
                <RefreshCw style={{ marginRight: '8px', animation: 'spin 1s linear infinite' }} size={20} />
              ) : (
                <TrendingUp style={{ marginRight: '8px' }} size={20} />
              )}
              {loading ? 'Processing Batch...' : 'Analyze Batch'}
            </button>
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
                </div>
              ))}
            </div>
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
        </div>

        {/* Footer */}
        <div style={{ textAlign: 'center', marginTop: '40px', paddingTop: '20px', borderTop: '1px solid #4B5563' }}>
          <p style={{ color: '#9CA3AF', fontSize: '14px' }}>
            Key Insights Dashboard - Powered by LLAVA & Ollama
          </p>
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        button:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        
        input:focus, textarea:focus, select:focus {
          outline: none;
          border-color: #8F74D4;
          box-shadow: 0 0 0 2px rgba(143, 116, 212, 0.2);
        }
        
        .insight-card:hover {
          transform: translateY(-2px);
          transition: transform 0.2s ease;
        }
      `}</style>
    </div>
  );
};

export default KeyInsightsDashboard;Generated: {new Date(item.generated_at).toLocaleString()}
                    