import { useState } from "react";
import { sendQuickReply } from "./quickRespondUtils";
import { Activity, Send, MessageCircle, Sparkles } from 'lucide-react';
import { QuickRespond } from "../../services/aiService";

export default function QuickRespond() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSend() {
    if (!input.trim()) return;
    setLoading(true);
    const reply = await sendQuickReply(input);
    setResponse(reply);
    setLoading(false);
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Simplify a response
  async function handleSimplify() {
    if (!response) return;
    setLoading(true);
    try {
      const res = await fetch(`${BASE_URL}/simplify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ original_analysis: response })
      });
      const data = await res.json();
      setResponse(data.simplified_text);
    } catch (err) {
      console.error("Simplify failed:", err);
    } finally {
      setLoading(false);
    }
  }

  // Run advanced analysis
  async function handleAdvancedAnalysis() {
    setLoading(true);
    try {
      const res = await fetch(`${BASE_URL}/advanced`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input })
      });
      const data = await res.json();
      setResponse(data.full_analysis);
    } catch (err) {
      console.error("Advanced analysis failed:", err);
    } finally {
      setLoading(false);
    }
  }

  // Health check
  async function checkHealth() {
    try {
      const res = await fetch(`${BASE_URL}/health`);
      const data = await res.json();
      alert(`Service status: ${data.status}, response time: ${data.response_time_ms}ms`);
    } catch (err) {
      console.error("Health check failed:", err);
    }
  }

  // ================= Extra Routes =================

  // Batch analysis
  async function handleBatchAnalysis(files) {
    setLoading(true);
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("screenshots", file));
      const res = await fetch(`${BASE_URL}/batch-analyze`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResponse(JSON.stringify(data.batch_results, null, 2));
    } catch (err) {
      console.error("Batch analysis failed:", err);
    } finally {
      setLoading(false);
    }
  }

  // Analyze screenshot
  async function handleAnalyzeScreenshot(file) {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("screenshot", file);
      const res = await fetch(`${BASE_URL}/analyze-screenshot`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResponse(data.full_analysis || JSON.stringify(data));
    } catch (err) {
      console.error("Screenshot analysis failed:", err);
    } finally {
      setLoading(false);
    }
  }

  // Update meeting context
  async function handleUpdateContext(contextText) {
    try {
      const res = await fetch(`${BASE_URL}/context/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context: contextText }),
      });
      const data = await res.json();
      alert(data.message);
    } catch (err) {
      console.error("Update context failed:", err);
    }
  }

  // Clear meeting context
  async function handleClearContext() {
    try {
      const res = await fetch(`${BASE_URL}/context/clear`, { method: "DELETE" });
      const data = await res.json();
      alert(data.message);
    } catch (err) {
      console.error("Clear context failed:", err);
    }
  }



  const handleSimplify = async (analysisText) => {
    try {
      const result = await apiService.request('/simplify', {
        method: 'POST',
        body: JSON.stringify({
          original_analysis: analysisText,
          simplification_level: 1
        })
      });
      
      return result;
    } catch (error) {
      console.error('Simplification failed:', error);
      return { error: error.message };
    }
  };

  const updateMeetingContext = async (context) => {
    try {
      await apiService.request('/context/update', {
        method: 'POST',
        body: JSON.stringify(context)
      });
      
      alert('Meeting context updated successfully!');
    } catch (error) {
      console.error('Failed to update context:', error);
      alert('Failed to update meeting context');
    }
  };

  const clearMeetingContext = async () => {
    try {
      await apiService.request('/context/clear', {
        method: 'DELETE'
      });
      
      setMeetingContext('');
      setAudioTranscript('');
      alert('Meeting context cleared successfully!');
    } catch (error) {
      console.error('Failed to clear context:', error);
      alert('Failed to clear meeting context');
    }
  };

  const fetchPaginatedItems = async (page = 1, pageSize = 10) => {
    try {
      const result = await apiService.request(`/items?page=${page}&page_size=${pageSize}`);
      setPaginatedData(result);
    } catch (error) {
      console.error('Failed to fetch paginated data:', error);
    }
  };

  // Utility function to get urgency color
  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'HIGH': return 'text-red-500';
      case 'MEDIUM': return 'text-yellow-500';
      case 'LOW': return 'text-green-500';
      default: return 'text-gray-500';
    }
  };

  // Render Health Status
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
            <span className="text-sm text-gray-300">Overall</span>
          </div>
          
          <div className="flex items-center">
            {healthStatus.ollama ? 
              <CheckCircle className="w-5 h-5 text-green-500 mr-2" /> :
              <XCircle className="w-5 h-5 text-red-500 mr-2" />
            }
            <span className="text-sm text-gray-300">Ollama</span>
          </div>
          
          <div className="flex items-center">
            {healthStatus.llava_model ? 
              <CheckCircle className="w-5 h-5 text-green-500 mr-2" /> :
              <XCircle className="w-5 h-5 text-red-500 mr-2" />
            }
            <span className="text-sm text-gray-300">LLAVA</span>
          </div>
          
          <div className="flex items-center">
            {healthStatus.llama_model ? 
              <CheckCircle className="w-5 h-5 text-green-500 mr-2" /> :
              <XCircle className="w-5 h-5 text-red-500 mr-2" />
            }
            <span className="text-sm text-gray-300">Llama</span>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Input Section */}
          <div className="xl:col-span-2 space-y-6">
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-xl font-semibold mb-4 text-slate-200 flex items-center">
                <MessageCircle className="mr-3" size={20} />
                Compose Response
              </h3>
              
              <div className="space-y-4">
                <div>
                  <label className="text-slate-300 text-sm mb-2 block">Your Message</label>
                  <textarea
                    placeholder="Type your answer or response here..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    rows={8}
                    className="w-full bg-slate-900/80 border border-slate-600 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:border-green-500 focus:ring-2 focus:ring-green-500/20 focus:outline-none transition-colors resize-none"
                  />
                  <div className="flex justify-between mt-2">
                    <span className="text-slate-500 text-xs">Press Enter to send, Shift+Enter for new line</span>
                    <span className="text-slate-500 text-xs">{input.length} characters</span>
                  </div>
                </div>

                <div className="flex space-x-3">
                  <button 
                    onClick={handleSend} 
                    disabled={loading || !input.trim()}
                    className="flex-1 bg-gradient-to-r from-green-600 to-green-700 text-white py-3 px-6 rounded-lg hover:from-green-700 hover:to-green-800 transition-all flex items-center justify-center space-x-2 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <Send size={18} />
                        <span>Send Response</span>
                      </>
                    )}
                  </button>
                  
                  <button 
                    onClick={() => {setInput(""); setResponse("");}}
                    className="bg-slate-700 text-slate-300 py-3 px-6 rounded-lg hover:bg-slate-600 transition-colors font-medium"
                  >
                    Clear
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200">Quick Templates</h3>
              <div className="grid grid-cols-2 gap-3">
                {[
                  "Thank you for your question...",
                  "I understand your concern...",
                  "Let me clarify that point...",
                  "That's an excellent observation..."
                ].map((template, index) => (
                  <button
                    key={index}
                    onClick={() => setInput(template)}
                    className="text-left p-3 bg-slate-900/50 hover:bg-slate-700/50 rounded-lg transition-colors text-slate-300 text-sm border border-slate-700 hover:border-slate-600"
                  >
                    {template}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Response Section */}
          <div className="space-y-6">
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                <Sparkles className="mr-2" size={20} />
                AI Response
              </h3>
              
              <div className="bg-slate-900/80 p-4 rounded-lg min-h-64 max-h-96 overflow-y-auto border border-slate-700">
                {loading ? (
                  <div className="flex flex-col items-center justify-center py-8">
                    <div className="w-8 h-8 border-4 border-green-600/30 border-t-green-600 rounded-full animate-spin mb-4"></div>
                    <p className="text-slate-400 text-sm">AI is generating your response...</p>
                  </div>
                )}
                
                {/* Full Analysis */}
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
                
                {/* Metadata */}
                <div className="text-sm text-gray-400 flex justify-between">
                  <span>Confidence: {(analysisResult.confidence_score * 100).toFixed(1)}%</span>
                  <span>Session: {analysisResult.session_id}</span>
                </div>
              </div>
              
              {response && (
                <div className="mt-4 flex space-x-2">
                  <button 
                    onClick={() => navigator.clipboard.writeText(response)}
                    className="bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
                  >
                    Copy Response
                  </button>
                  <button 
                    onClick={() => setResponse("")}
                    className="bg-slate-700 text-slate-300 py-2 px-4 rounded-lg hover:bg-slate-600 transition-colors text-sm font-medium"
                  >
                    Clear Response
                  </button>
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

  // Render Configuration Tab
  const renderConfigTab = () => (
    <div className="space-y-6">
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-200 mb-4">Configuration Management</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium text-gray-300 mb-2">Ollama Configs</h4>
            <div className="bg-gray-700 rounded p-3 min-h-[100px]">
              {configs.ollama.length === 0 ? (
                <p className="text-gray-400 text-sm">No configurations found</p>
              ) : (
                configs.ollama.map((config, index) => (
                  <div key={index} className="text-gray-300 text-sm mb-1">
                    {config.base_url || 'Default Config'}
                  </div>
                ))
              )}
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-300 mb-2">Quick Respond Configs</h4>
            <div className="bg-gray-700 rounded p-3 min-h-[100px]">
              {configs.quickRespond.length === 0 ? (
                <p className="text-gray-400 text-sm">No configurations found</p>
              ) : (
                configs.quickRespond.map((config, index) => (
                  <div key={index} className="text-gray-300 text-sm mb-1">
                    Config {index + 1}
                  </div>
                ))
              )}
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-300 mb-2">Model Prompts</h4>
            <div className="bg-gray-700 rounded p-3 min-h-[100px]">
              {configs.modelPrompts.length === 0 ? (
                <p className="text-gray-400 text-sm">No prompts found</p>
              ) : (
                configs.modelPrompts.map((prompt, index) => (
                  <div key={index} className="text-gray-300 text-sm mb-1">
                    Prompt {index + 1}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
        
        <button
          onClick={fetchConfigs}
          className="mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
        >
          Refresh Configs
        </button>
      </div>

      {/* Urgency Levels */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-200 mb-4">Urgency Levels</h3>
        <div className="flex flex-wrap gap-2">
          {urgencyLevels.map((level, index) => (
            <span
              key={index}
              className={`px-3 py-1 rounded-full text-sm font-medium ${getUrgencyColor(level)}`}
            >
              {level}
            </span>
          ))}
        </div>
      </div>

      {/* Paginated Data Demo */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-200 mb-4">Paginated Items Demo</h3>
        
        <button
          onClick={() => fetchPaginatedItems()}
          className="mb-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
        >
          Fetch Items
        </button>
        
        {paginatedData && (
          <div>
            <div className="mb-4">
              <p className="text-gray-300">
                Page {paginatedData.page} of {Math.ceil(paginatedData.total / paginatedData.page_size)}
                ({paginatedData.total} total items)
              </p>
            </div>
            
            <div className="bg-gray-700 rounded p-3 max-h-40 overflow-y-auto">
              {paginatedData.items.map((item, index) => (
                <div key={index} className="text-gray-300 text-sm py-1">
                  {typeof item === 'string' ? item : JSON.stringify(item)}
                </div>
              ))}
            </div>
            
            <div className="flex gap-2 mt-4">
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
    <div style={{ fontFamily: 'Roboto, sans-serif', backgroundColor: '#1E1E2F', minHeight: '100vh', color: '#F8FAFC' }}>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Meeting Analysis Dashboard</h1>
          <p className="text-gray-400">AI-powered meeting screenshot analysis and insights</p>
        </div>

        {/* Health Status */}
        {renderHealthStatus()}

        {/* Navigation Tabs */}
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
              onClick={() => setActiveTab('config')}
              className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'config'
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Configuration
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="max-w-6xl mx-auto">
          {activeTab === 'analyze' && renderAnalysisTab()}
          {activeTab === 'config' && renderConfigTab()}
        </div>
      </div>
    </div>
  );
};

export default MeetingAnalysisDashboard;