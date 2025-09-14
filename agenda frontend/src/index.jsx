import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css"; // only if you have styles
import { 
  Camera, 
  Zap, 
  Activity, 
  FileText, 
  Key, 
  Mic, 
  Eye, 
  EyeOff, 
  Monitor,
  Volume2,
  Circle,
  Home,
  ChevronLeft,
  ChevronRight,
  Settings,
  User
} from 'lucide-react';

const InvisibleAI = () => {
  const [activeFeature, setActiveFeature] = useState('dashboard');
  const [isRecording, setIsRecording] = useState(false);
  const [isInvisible, setIsInvisible] = useState(false);
  const [micStatus, setMicStatus] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const features = [
    { id: 'imagerecognition', name: 'Image Recognition', icon: Camera, color: 'bg-blue-600' },
    { id: 'handsfree', name: 'Hands Free', icon: Zap, color: 'bg-yellow-600' },
    { id: 'quickresponse', name: 'Quick Response', icon: Activity, color: 'bg-green-600' },
    { id: 'summarization', name: 'Summarization', icon: FileText, color: 'bg-orange-600' },
    { id: 'keyinsights', name: 'Key Insights', icon: Key, color: 'bg-red-600' },
    { id: 'voicerecognition', name: 'Voice Recognition', icon: Mic, color: 'bg-pink-600' },
    { id: 'invisibility', name: 'Invisibility', icon: Eye, color: 'bg-purple-600' }
  ];

  const FeatureCard = ({ feature, isActive, onClick }) => {
    const Icon = feature.icon;
    return (
      <div
        className={`p-4 rounded-xl cursor-pointer transition-all duration-300 border-2 ${
          isActive 
            ? `${feature.color} text-white shadow-lg border-white/20 scale-105` 
            : 'bg-slate-800/50 text-slate-300 hover:bg-slate-700/70 border-slate-700 hover:border-slate-600'
        }`}
        onClick={onClick}
      >
        <div className="flex flex-col items-center space-y-2">
          <Icon size={28} />
          <span className="text-xs font-medium text-center leading-tight">{feature.name}</span>
        </div>
      </div>
    );
  };

  const WindowControls = () => (
    <div className="flex items-center space-x-2">
      <button className="w-3 h-3 bg-red-500 rounded-full hover:bg-red-600 transition-colors"></button>
      <button className="w-3 h-3 bg-yellow-500 rounded-full hover:bg-yellow-600 transition-colors"></button>
      <button className="w-3 h-3 bg-green-500 rounded-full hover:bg-green-600 transition-colors"></button>
    </div>
  );

  const ImageRecognitionPanel = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-slate-100">Image Recognition</h2>
        <div className="flex space-x-2">
          <div className="px-3 py-1 bg-blue-600 text-white text-sm rounded-full">Active</div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
            <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
              <Camera className="mr-2" size={20} />
              Camera Capture
            </h3>
            <div className="bg-slate-900/80 p-6 rounded-lg mb-4 min-h-64 flex items-center justify-center border-2 border-dashed border-slate-600">
              <div className="text-center">
                <Camera size={64} className="text-slate-500 mx-auto mb-2" />
                <p className="text-slate-400">Camera feed will appear here</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <button className="bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors font-medium">
                Start Camera
              </button>
              <button className="bg-slate-700 text-slate-300 py-3 px-4 rounded-lg hover:bg-slate-600 transition-colors font-medium">
                Capture
              </button>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
            <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
              <Monitor className="mr-2" size={20} />
              Screen Capture
            </h3>
            <div className="bg-slate-900/80 p-6 rounded-lg mb-4 min-h-64 flex items-center justify-center border-2 border-dashed border-slate-600">
              <div className="text-center">
                <Monitor size={64} className="text-slate-500 mx-auto mb-2" />
                <p className="text-slate-400">Screen capture will appear here</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <button className="bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors font-medium">
                Capture Screen
              </button>
              <button className="bg-slate-700 text-slate-300 py-3 px-4 rounded-lg hover:bg-slate-600 transition-colors font-medium">
                Analyze
              </button>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200">Recognition Results</h3>
          <div className="bg-slate-900/80 p-4 rounded-lg min-h-96">
            <div className="space-y-4">
              <div className="border-b border-slate-700 pb-3">
                <h4 className="font-medium text-slate-300 mb-2">Detected Objects</h4>
                <p className="text-slate-500 text-sm">No objects detected yet</p>
              </div>
              <div className="border-b border-slate-700 pb-3">
                <h4 className="font-medium text-slate-300 mb-2">Text Recognition</h4>
                <p className="text-slate-500 text-sm">No text detected yet</p>
              </div>
              <div>
                <h4 className="font-medium text-slate-300 mb-2">Confidence Score</h4>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full w-0"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const VoiceRecognitionPanel = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold text-slate-100">Voice Recognition</h2>
        <div className="flex space-x-2">
          <div className={`px-3 py-1 text-white text-sm rounded-full ${micStatus ? 'bg-green-600' : 'bg-slate-600'}`}>
            {micStatus ? 'Listening' : 'Standby'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200">Prep Voice</h3>
          <div className="text-center space-y-4">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-colors ${
              micStatus ? 'bg-green-600 animate-pulse' : 'bg-slate-900'
            }`}>
              <Mic size={32} className="text-white" />
            </div>
            <button 
              onClick={() => setMicStatus(!micStatus)}
              className={`w-full py-3 px-4 rounded-lg transition-colors font-medium ${
                micStatus ? 'bg-red-600 hover:bg-red-700' : 'bg-pink-600 hover:bg-pink-700'
              } text-white`}
            >
              {micStatus ? 'Stop Prep' : 'Start Prep'}
            </button>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200">Processing</h3>
          <div className="text-center space-y-4">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-colors ${
              isRecording ? 'bg-red-600 animate-pulse' : 'bg-slate-900'
            }`}>
              <Activity size={32} className="text-white" />
            </div>
            <button 
              onClick={() => setIsRecording(!isRecording)}
              className={`w-full py-3 px-4 rounded-lg transition-colors font-medium ${
                isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-pink-600 hover:bg-pink-700'
              } text-white`}
            >
              {isRecording ? 'Stop' : 'Start'}
            </button>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200">Recognition</h3>
          <div className="text-center space-y-4">
            <div className="w-20 h-20 bg-slate-900 rounded-full flex items-center justify-center mx-auto">
              <Volume2 size={32} className="text-slate-500" />
            </div>
            <button className="w-full bg-pink-600 text-white py-3 px-4 rounded-lg hover:bg-pink-700 transition-colors font-medium">
              Recognize
            </button>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-lg font-semibold mb-4 text-slate-200">Live Transcription</h3>
          <div className="bg-slate-900/80 p-4 rounded-lg min-h-32 max-h-48 overflow-y-auto">
            <p className="text-slate-400 text-sm">Voice transcription will appear here in real-time...</p>
          </div>
        </div>
      </div>    
    </div>
  );

  const Dashboard = () => (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-4xl font-bold text-slate-100 mb-2">Dashboard</h2>
          <p className="text-slate-400">AI Assistant Control Center</p>
        </div>
        <div className="flex items-center space-x-4">
          <button 
            onClick={() => setIsInvisible(!isInvisible)}
            className={`flex items-center space-x-2 py-3 px-6 rounded-xl transition-colors font-medium ${
              isInvisible ? 'bg-green-600 hover:bg-green-700 shadow-lg' : 'bg-slate-700 hover:bg-slate-600'
            } text-white`}
          >
            {isInvisible ? <EyeOff size={20} /> : <Eye size={20} />}
            <span>{isInvisible ? 'Invisible Mode ON' : 'Invisible Mode OFF'}</span>
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-4 xl:grid-cols-7 gap-4">
        {features.map((feature) => (
          <FeatureCard
            key={feature.id}
            feature={feature}
            isActive={activeFeature === feature.id}
            onClick={() => setActiveFeature(feature.id)}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        <div className="xl:col-span-2 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
            <h3 className="text-xl font-semibold mb-6 text-slate-200">Quick Actions</h3>
            <div className="space-y-4">
              <button className="w-full bg-gradient-to-r from-yellow-600 to-yellow-700 text-white py-4 px-6 rounded-lg hover:from-yellow-700 hover:to-yellow-800 transition-all flex items-center space-x-3 font-medium">
                <Zap size={20} />
                <span>Quick Response Mode</span>
              </button>
              <button className="w-full bg-gradient-to-r from-orange-600 to-orange-700 text-white py-4 px-6 rounded-lg hover:from-orange-700 hover:to-orange-800 transition-all flex items-center space-x-3 font-medium">
                <FileText size={20} />
                <span>Generate Summary</span>
              </button>
              <button className="w-full bg-gradient-to-r from-red-600 to-red-700 text-white py-4 px-6 rounded-lg hover:from-red-700 hover:to-red-800 transition-all flex items-center space-x-3 font-medium">
                <Key size={20} />
                <span>Extract Key Insights</span>
              </button>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
            <h3 className="text-xl font-semibold mb-6 text-slate-200">Performance Metrics</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-slate-300 text-sm">Processing Speed</span>
                  <span className="text-slate-400 text-sm">95%</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full w-[95%]"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-slate-300 text-sm">Accuracy Rate</span>
                  <span className="text-slate-400 text-sm">89%</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full w-[89%]"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-slate-300 text-sm">Response Time</span>
                  <span className="text-slate-400 text-sm">0.3s</span>
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div className="bg-purple-500 h-2 rounded-full w-[75%]"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
          <h3 className="text-xl font-semibold mb-6 text-slate-200">System Status</h3>
          <div className="space-y-6">
            {[
              { name: 'Voice Recognition', status: micStatus, color: 'green' },
              { name: 'Image Processing', status: true, color: 'blue' },
              { name: 'AI Assistant', status: true, color: 'purple' },
              { name: 'Network Connection', status: true, color: 'green' },
              { name: 'Security Layer', status: isInvisible, color: isInvisible ? 'green' : 'orange' }
            ].map((item, index) => (
              <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-slate-900/50">
                <span className="text-slate-300 font-medium">{item.name}</span>
                <div className="flex items-center space-x-2">
                  <Circle 
                    size={8} 
                    className={`fill-current ${
                      item.status 
                        ? item.color === 'green' ? 'text-green-500' 
                        : item.color === 'blue' ? 'text-blue-500'
                        : 'text-purple-500'
                        : 'text-orange-500'
                    }`} 
                  />
                  <span className={`text-sm font-medium ${
                    item.status ? 'text-green-400' : 'text-orange-400'
                  }`}>
                    {item.status ? 'Active' : 'Standby'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderActiveFeature = () => {
    switch (activeFeature) {
      case 'imagerecognition':
        return <ImageRecognitionPanel />;
      case 'voicerecognition':
        return <VoiceRecognitionPanel />;
      case 'handsfree':
        return (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-slate-100">Hands Free Mode</h2>
            <div className="bg-slate-800/50 backdrop-blur p-12 rounded-xl text-center border border-slate-700">
              <Zap size={96} className="text-yellow-500 mx-auto mb-6" />
              <h3 className="text-2xl font-semibold mb-4 text-slate-200">Voice Command Interface</h3>
              <p className="text-slate-400 mb-8 text-lg max-w-2xl mx-auto">
                Control the entire application using voice commands. Perfect for hands-free operation during presentations or interviews.
              </p>
              <button className="bg-gradient-to-r from-yellow-600 to-yellow-700 text-white py-4 px-8 rounded-lg hover:from-yellow-700 hover:to-yellow-800 transition-all font-medium text-lg">
                Enable Hands Free Mode
              </button>
            </div>
          </div>
        );
      case 'quickresponse':
        return (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-slate-100">Quick Response</h2>
            <div className="bg-slate-800/50 backdrop-blur p-8 rounded-xl border border-slate-700">
              <h3 className="text-xl font-semibold mb-6 text-slate-200">Rapid AI Response System</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-slate-900/80 p-6 rounded-lg">
                  <h4 className="font-semibold text-slate-200 mb-4">Response Configuration</h4>
                  <div className="space-y-4">
                    <div>
                      <label className="text-slate-300 text-sm">Response Delay (ms)</label>
                      <input type="range" min="100" max="2000" defaultValue="500" className="w-full mt-2" />
                    </div>
                    <div>
                      <label className="text-slate-300 text-sm">Confidence Threshold</label>
                      <input type="range" min="0" max="100" defaultValue="80" className="w-full mt-2" />
                    </div>
                  </div>
                </div>
                <div className="bg-slate-900/80 p-6 rounded-lg">
                  <h4 className="font-semibold text-slate-200 mb-4">Quick Responses</h4>
                  <p className="text-slate-400 mb-4">AI will provide instant responses to detected questions</p>
                  <button className="w-full bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 transition-colors font-medium">
                    Activate Quick Response
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      case 'summarization':
        return (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-slate-100">Summarization</h2>
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              <div className="xl:col-span-2 bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                <h3 className="text-xl font-semibold mb-4 text-slate-200">Content Summary</h3>
                <div className="bg-slate-900/80 p-6 rounded-lg min-h-96">
                  <p className="text-slate-400">Interview summaries and key points will appear here as the conversation progresses...</p>
                </div>
              </div>
              <div className="space-y-4">
                <div className="bg-slate-800/50 backdrop-blur p-4 rounded-xl border border-slate-700">
                  <h4 className="font-semibold text-slate-200 mb-3">Summary Settings</h4>
                  <div className="space-y-3">
                    <label className="flex items-center space-x-2">
                      <input type="checkbox" className="rounded" />
                      <span className="text-slate-300 text-sm">Auto-generate</span>
                    </label>
                    <label className="flex items-center space-x-2">
                      <input type="checkbox" className="rounded" />
                      <span className="text-slate-300 text-sm">Include timestamps</span>
                    </label>
                  </div>
                </div>
                <button className="w-full bg-orange-600 text-white py-3 px-4 rounded-lg hover:bg-orange-700 transition-colors font-medium">
                  Generate Summary
                </button>
              </div>
            </div>
          </div>
        );
      case 'keyinsights':
        return (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-slate-100">Key Insights</h2>
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
              <div className="xl:col-span-3 bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                <h3 className="text-xl font-semibold mb-4 text-slate-200">Important Points & Insights</h3>
                <div className="bg-slate-900/80 p-6 rounded-lg min-h-96">
                  <div className="space-y-4">
                    <div className="border-l-4 border-red-500 pl-4">
                      <h4 className="font-medium text-slate-300">Key Insight Example</h4>
                      <p className="text-slate-400 text-sm mt-1">Important points and insights will be highlighted here with different priorities</p>
                    </div>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="bg-slate-800/50 backdrop-blur p-4 rounded-xl border border-slate-700">
                  <h4 className="font-semibold text-slate-200 mb-3">Insight Categories</h4>
                  <div className="space-y-2">
                    {['Critical', 'Important', 'Relevant', 'Notable'].map((category, i) => (
                      <div key={i} className="flex justify-between text-sm">
                        <span className="text-slate-300">{category}</span>
                        <span className="text-slate-500">0</span>
                      </div>
                    ))}
                  </div>
                </div>
                <button className="w-full bg-red-600 text-white py-3 px-4 rounded-lg hover:bg-red-700 transition-colors font-medium">
                  Extract Insights
                </button>
              </div>
            </div>
          </div>
        );
      case 'invisibility':
        return (
          <div className="space-y-6">
            <h2 className="text-3xl font-bold text-slate-100">Invisibility Mode</h2>
            <div className="bg-slate-800/50 backdrop-blur p-12 rounded-xl text-center border border-slate-700">
              {isInvisible ? (
                <EyeOff size={96} className="text-green-500 mx-auto mb-6" />
              ) : (
                <Eye size={96} className="text-slate-500 mx-auto mb-6" />
              )}
              <h3 className="text-2xl font-semibold mb-4 text-slate-200">
                {isInvisible ? 'Invisible Mode Active' : 'Invisible Mode Inactive'}
              </h3>
              <p className="text-slate-400 mb-8 text-lg max-w-2xl mx-auto">
                {isInvisible 
                  ? 'The AI assistant is running invisibly in the background, monitoring and analyzing without detection' 
                  : 'Activate invisible mode for stealth operation during sensitive conversations or interviews'}
              </p>
              <button 
                onClick={() => setIsInvisible(!isInvisible)}
                className={`py-4 px-8 rounded-lg transition-all font-medium text-lg ${
                  isInvisible 
                    ? 'bg-red-600 hover:bg-red-700 text-white' 
                    : 'bg-purple-600 hover:bg-purple-700 text-white'
                }`}
              >
                {isInvisible ? 'Deactivate Invisibility' : 'Activate Invisibility'}
              </button>
            </div>
          </div>
        );
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 font-sans">
      {/* Desktop Window Frame */}
      <div className="bg-slate-800/50 backdrop-blur border-b border-slate-700 px-6 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <WindowControls />
            <div className="flex items-center space-x-3 ml-4">
              <div className="w-8 h-8 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg flex items-center justify-center">
                <Eye size={18} className="text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold">Invisible AI</h1>
                <p className="text-xs text-slate-400">v2.1 Desktop Edition</p>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1 bg-slate-700/50 rounded-full">
              <Circle size={6} className="text-green-500 fill-current" />
              <span className="text-xs text-slate-300">Connected</span>
            </div>
            <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
              <Settings size={18} />
            </button>
            <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
              <User size={18} />
            </button>
          </div>
        </div>
      </div>

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Collapsible Sidebar */}
        <aside className={`bg-slate-800/30 backdrop-blur border-r border-slate-700 transition-all duration-300 ${
          sidebarCollapsed ? 'w-16' : 'w-64'
        }`}>
          <div className="p-4">
            <button
              onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
              className="w-full flex items-center justify-center p-2 hover:bg-slate-700/50 rounded-lg transition-colors mb-4"
            >
              {sidebarCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
            </button>
          </div>
          
          <nav className="px-2 space-y-1">
            <button
              onClick={() => setActiveFeature('dashboard')}
              className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-colors ${
                activeFeature === 'dashboard' ? 'bg-purple-600 text-white' : 'text-slate-300 hover:bg-slate-700/50'
              }`}
              title="Dashboard"
            >
              <Home size={20} />
              {!sidebarCollapsed && <span className="font-medium">Dashboard</span>}
            </button>
            
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <button
                  key={feature.id}
                  onClick={() => setActiveFeature(feature.id)}
                  className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-colors ${
                    activeFeature === feature.id 
                      ? `${feature.color} text-white` 
                      : 'text-slate-300 hover:bg-slate-700/50'
                  }`}
                  title={feature.name}
                >
                  <Icon size={20} />
                  {!sidebarCollapsed && <span className="font-medium">{feature.name}</span>}
                </button>
              );
            })}
          </nav>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto">
          <div className="p-8">
            {renderActiveFeature()}
          </div>
        </main>
      </div>
    </div>
  );
};

// Main App Component that integrates with your existing App.jsx
const AppWithInvisibleAI = () => {
  const [showInvisibleAI, setShowInvisibleAI] = React.useState(false);

  // You can toggle between your existing App and InvisibleAI
  if (showInvisibleAI) {
    return <InvisibleAI />;
  }

  return (
    <div>
      <div className="fixed top-4 right-4 z-50">
        <button
          onClick={() => setShowInvisibleAI(true)}
          className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg shadow-lg transition-colors"
        >
          Switch to Invisible AI
        </button>
      </div>
      <App />
    </div>
  );
};

// Render the application
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <AppWithInvisibleAI />
  </React.StrictMode>
);

export default InvisibleAI;