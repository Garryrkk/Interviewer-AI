import { useEffect, useState } from "react";
import { startListening } from "./handsFreeUtils";
import { sendQuickReply } from "../QuickRespond/quickRespondUtils";
import { HandsFreeMode } from "../../services/aiService";
import { Zap, Mic, Activity, Volume2, User, Bot } from 'lucide-react';

export default function HandsFreeMode() {
  const [messages, setMessages] = useState([]); // chat history
  const [listening, setListening] = useState(false);

  useEffect(() => {
    if (listening) {
      // Start speech recognition
      startListening(async (transcript) => {
        // Add interviewer's text
        setMessages((prev) => [...prev, { from: "interviewer", text: transcript }]);

        // AI generates quick response
        const aiReply = await sendQuickReply(transcript);
        setMessages((prev) => [...prev, { from: "ai", text: aiReply }]);
      });
    }
  }, [listening]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-slate-100">Hands Free Mode</h2>
        <div className="flex space-x-2">
          <div className={`px-3 py-1 text-white text-sm rounded-full ${listening ? 'bg-green-600' : 'bg-slate-600'}`}>
            {listening ? 'Active' : 'Standby'}
          </div>
        </div>
      </div>

      {/* Control Panel */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
            <Zap className="mr-2" size={20} />
            Voice Command Control
          </h3>
          <div className="text-center space-y-4">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-colors ${
              listening ? 'bg-green-600 animate-pulse' : 'bg-slate-900'
            }`}>
              <Mic size={32} className="text-white" />
            </div>
            <button 
              onClick={() => setListening(!listening)}
              className={`w-full py-3 px-4 rounded-lg transition-colors font-medium ${
                listening ? 'bg-red-600 hover:bg-red-700' : 'bg-yellow-600 hover:bg-yellow-700'
              } text-white`}
            >
              {listening ? "Stop Hands-Free Mode" : "Start Hands-Free Mode"}
            </button>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
            <Activity className="mr-2" size={20} />
            Processing Status
          </h3>
          <div className="text-center space-y-4">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-colors ${
              listening ? 'bg-blue-600 animate-pulse' : 'bg-slate-900'
            }`}>
              <Activity size={32} className="text-white" />
            </div>
            <div className="text-sm text-slate-300">
              {listening ? "Listening for commands..." : "Waiting for activation"}
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
            <Volume2 className="mr-2" size={20} />
            Response System
          </h3>
          <div className="text-center space-y-4">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-colors ${
              messages.length > 0 ? 'bg-purple-600' : 'bg-slate-900'
            }`}>
              <Volume2 size={32} className="text-white" />
            </div>
            <div className="text-sm text-slate-300">
              {messages.length > 0 ? `${messages.length} interactions` : "No responses yet"}
            </div>
          </div>
        </div>
      </div>

      {/* Chat History */}
      <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
        <h3 className="text-xl font-semibold mb-6 text-slate-200">Conversation History</h3>
        <div className="bg-slate-900/80 p-6 rounded-lg min-h-96 max-h-96 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="text-center text-slate-400 py-12">
              <Mic size={64} className="text-slate-600 mx-auto mb-4" />
              <p className="text-lg mb-2">Ready for hands-free interaction</p>
              <p className="text-sm">Voice commands and AI responses will appear here in real-time</p>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, index) => (
                <div key={index} className={`flex items-start space-x-3 p-4 rounded-lg ${
                  msg.from === "ai" ? "bg-blue-900/30 border-l-4 border-blue-500" : "bg-green-900/30 border-l-4 border-green-500"
                }`}>
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    msg.from === "ai" ? "bg-blue-600" : "bg-green-600"
                  }`}>
                    {msg.from === "ai" ? <Bot size={16} className="text-white" /> : <User size={16} className="text-white" />}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="font-medium text-slate-200">
                        {msg.from === "ai" ? "AI Assistant" : "Interviewer"}
                      </span>
                      <span className="text-xs text-slate-500">
                        {new Date().toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-slate-300">{msg.text}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Feature Description */}
      {!listening && messages.length === 0 && (
        <div className="bg-slate-800/50 backdrop-blur p-8 rounded-xl text-center border border-slate-700">
          <Zap size={64} className="text-yellow-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-4 text-slate-200">Voice Command Interface</h3>
          <p className="text-slate-400 mb-6 text-base max-w-2xl mx-auto">
            Control the entire application using voice commands. Perfect for hands-free operation during presentations or interviews.
            The AI will automatically respond to detected speech with intelligent replies.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <Mic className="text-blue-400 mx-auto mb-2" size={24} />
              <h4 className="font-medium text-slate-200 mb-2">Voice Detection</h4>
              <p className="text-sm text-slate-400">Automatically detects and transcribes speech</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <Activity className="text-green-400 mx-auto mb-2" size={24} />
              <h4 className="font-medium text-slate-200 mb-2">AI Processing</h4>
              <p className="text-sm text-slate-400">Generates intelligent responses in real-time</p>
            </div>
            <div className="bg-slate-900/50 p-4 rounded-lg">
              <Volume2 className="text-purple-400 mx-auto mb-2" size={24} />
              <h4 className="font-medium text-slate-200 mb-2">Audio Output</h4>
              <p className="text-sm text-slate-400">Provides spoken responses for complete hands-free experience</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}