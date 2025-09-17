import { useState } from "react";
import { FileText, Zap, Copy, Download } from 'lucide-react';
import { summarizeText } from "./summarizationUtils";
import { Summarization } from "../../services/aiService";

export default function Summarization() {
  const [input, setInput] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [autoGenerate, setAutoGenerate] = useState(false);
  const [includeTimestamps, setIncludeTimestamps] = useState(false);

  async function handleSummarize() {
    if (!input.trim()) return;
    setLoading(true);
    const result = await summarizeText(input);
    setSummary(result);
    setLoading(false);
  }

  const handleCopySummary = () => {
    navigator.clipboard.writeText(summary);
  };

  const handleDownloadSummary = () => {
    const blob = new Blob([summary], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'summary.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 font-sans p-8">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-4xl font-bold text-slate-100 mb-2">Summarization</h2>
            <p className="text-slate-400">AI-powered text summarization tool</p>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`px-3 py-1 text-white text-sm rounded-full ${
              loading ? 'bg-orange-600' : summary ? 'bg-green-600' : 'bg-slate-600'
            }`}>
              {loading ? 'Processing' : summary ? 'Complete' : 'Ready'}
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          
          {/* Input Section - Takes 2 columns */}
          <div className="xl:col-span-2 space-y-6">
            
            {/* Input Text Area */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-xl font-semibold mb-4 text-slate-200 flex items-center">
                <FileText className="mr-2" size={20} />
                Input Content
              </h3>
              <textarea
                placeholder="Paste your long text here for summarization..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                className="w-full h-64 bg-slate-900/80 text-slate-200 p-4 rounded-lg border border-slate-600 focus:border-orange-500 focus:outline-none resize-none transition-colors placeholder-slate-500"
              />
              <div className="mt-4 flex justify-between items-center">
                <span className="text-slate-400 text-sm">
                  {input.length} characters
                </span>
                <button
                  onClick={handleSummarize}
                  disabled={loading || !input.trim()}
                  className={`px-6 py-3 rounded-lg transition-all font-medium flex items-center space-x-2 ${
                    loading || !input.trim()
                      ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800 text-white shadow-lg'
                  }`}
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                      <span>Summarizing...</span>
                    </>
                  ) : (
                    <>
                      <Zap size={16} />
                      <span>Summarize</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Summary Output */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-slate-200 flex items-center">
                  <FileText className="mr-2" size={20} />
                  Generated Summary
                </h3>
                {summary && (
                  <div className="flex space-x-2">
                    <button
                      onClick={handleCopySummary}
                      className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors text-slate-300"
                      title="Copy to clipboard"
                    >
                      <Copy size={16} />
                    </button>
                    <button
                      onClick={handleDownloadSummary}
                      className="p-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors text-slate-300"
                      title="Download summary"
                    >
                      <Download size={16} />
                    </button>
                  </div>
                )}
              </div>
              <div className="bg-slate-900/80 p-6 rounded-lg min-h-64">
                {summary ? (
                  <div className="space-y-4">
                    <p className="text-slate-200 leading-relaxed">{summary}</p>
                    <div className="pt-4 border-t border-slate-700">
                      <span className="text-slate-400 text-sm">
                        Summary generated • {summary.length} characters
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <p className="text-slate-400 text-center">
                      Your summarized content will appear here...
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Settings Panel - Takes 1 column */}
          <div className="space-y-6">
            
            {/* Summary Settings */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200">Summary Settings</h3>
              <div className="space-y-4">
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={autoGenerate}
                    onChange={(e) => setAutoGenerate(e.target.checked)}
                    className="w-4 h-4 rounded bg-slate-700 border-slate-600 text-orange-600 focus:ring-orange-500"
                  />
                  <span className="text-slate-300 text-sm">Auto-generate on input</span>
                </label>
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={includeTimestamps}
                    onChange={(e) => setIncludeTimestamps(e.target.checked)}
                    className="w-4 h-4 rounded bg-slate-700 border-slate-600 text-orange-600 focus:ring-orange-500"
                  />
                  <span className="text-slate-300 text-sm">Include timestamps</span>
                </label>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={handleSummarize}
                  disabled={loading || !input.trim()}
                  className={`w-full py-3 px-4 rounded-lg transition-colors font-medium flex items-center justify-center space-x-2 ${
                    loading || !input.trim()
                      ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800 text-white'
                  }`}
                >
                  <FileText size={16} />
                  <span>Generate Summary</span>
                </button>
                <button
                  onClick={() => {
                    setInput("");
                    setSummary("");
                  }}
                  className="w-full bg-slate-700 text-slate-300 py-3 px-4 rounded-lg hover:bg-slate-600 transition-colors font-medium"
                >
                  Clear All
                </button>
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200">Statistics</h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-300">Input Length</span>
                  <span className="text-slate-400">{input.length} chars</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-300">Summary Length</span>
                  <span className="text-slate-400">{summary.length} chars</span>
                </div>
                {summary && input && (
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">Compression</span>
                    <span className="text-orange-400">
                      {Math.round((1 - summary.length / input.length) * 100)}%
                    </span>
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}