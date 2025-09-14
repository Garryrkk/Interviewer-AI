import { useState } from "react";
import { sendQuickReply } from "./quickRespondUtils";
import { Activity, Send, MessageCircle, Sparkles } from 'lucide-react';

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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-12 h-12 bg-gradient-to-r from-green-600 to-green-700 rounded-xl flex items-center justify-center">
              <Activity size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-100">Quick Response</h1>
              <p className="text-slate-400">AI-Powered Instant Reply System</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`px-3 py-1 text-white text-sm rounded-full ${loading ? 'bg-yellow-600' : 'bg-green-600'}`}>
              {loading ? 'Processing' : 'Ready'}
            </div>
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
                ) : response ? (
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2 mb-3">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span className="text-green-400 text-sm font-medium">Generated Response</span>
                    </div>
                    <p className="text-slate-200 leading-relaxed">{response}</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <MessageCircle size={48} className="text-slate-500 mb-3" />
                    <p className="text-slate-400 text-sm">AI response will appear here</p>
                    <p className="text-slate-500 text-xs mt-1">Type a message and click send to get started</p>
                  </div>
                )}
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
              )}
            </div>

            {/* Settings Panel */}
            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
              <h3 className="text-lg font-semibold mb-4 text-slate-200">Response Settings</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-slate-300 text-sm mb-2 block">Response Style</label>
                  <select className="w-full bg-slate-900/80 border border-slate-600 rounded-lg px-3 py-2 text-slate-100 text-sm focus:border-green-500 focus:outline-none">
                    <option>Professional</option>
                    <option>Casual</option>
                    <option>Technical</option>
                    <option>Creative</option>
                  </select>
                </div>
                <div>
                  <label className="text-slate-300 text-sm mb-2 block">Response Length</label>
                  <select className="w-full bg-slate-900/80 border border-slate-600 rounded-lg px-3 py-2 text-slate-100 text-sm focus:border-green-500 focus:outline-none">
                    <option>Short</option>
                    <option>Medium</option>
                    <option>Long</option>
                    <option>Detailed</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  <input type="checkbox" className="rounded border-slate-600" />
                  <span className="text-slate-300 text-sm">Include examples</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200">Performance Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500 mb-1">0.3s</div>
              <div className="text-slate-400 text-sm">Average Response Time</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500 mb-1">98%</div>
              <div className="text-slate-400 text-sm">Accuracy Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-500 mb-1">{response ? "1" : "0"}</div>
              <div className="text-slate-400 text-sm">Responses Generated</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}