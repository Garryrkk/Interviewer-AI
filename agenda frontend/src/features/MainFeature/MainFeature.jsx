import React, { useState, useEffect } from 'react';
import { 
  MessageSquare, 
  Zap, 
  FileText, 
  Mic, 
  Brain, 
  Settings,
  ChevronDown,
  ChevronUp,
  Play,
  Square,
  RotateCcw
} from 'lucide-react';
import HiddenAnswers from './HiddenAnswers';

const MainFeature = () => {
  const [activeFeature, setActiveFeature] = useState('hidden-answers');
  const [isExpanded, setIsExpanded] = useState(true);
  const [featureStatus, setFeatureStatus] = useState({
    'hidden-answers': 'ready',
    'quick-respond': 'coming-soon',
    'summarization': 'coming-soon',
    'hands-free': 'coming-soon',
    'key-insights': 'coming-soon'
  });

  const features = [
    {
      id: 'hidden-answers',
      name: 'Hidden Answers',
      icon: MessageSquare,
      description: 'Send AI-generated responses to your private overlay window',
      color: 'indigo',
      status: 'ready'
    },
    {
      id: 'quick-respond',
      name: 'Quick Respond',
      icon: Zap,
      description: 'Generate instant responses to interview questions',
      color: 'yellow',
      status: 'coming-soon'
    },
    {
      id: 'summarization',
      name: 'Smart Summarization',
      icon: FileText,
      description: 'Summarize long conversations and key points',
      color: 'green',
      status: 'coming-soon'
    },
    {
      id: 'hands-free',
      name: 'Hands Free Mode',
      icon: Mic,
      description: 'Voice-activated AI assistance during interviews',
      color: 'purple',
      status: 'coming-soon'
    },
    {
      id: 'key-insights',
      name: 'Key Insights',
      icon: Brain,
      description: 'Extract and highlight important information',
      color: 'blue',
      status: 'coming-soon'
    }
  ];

  const getColorClasses = (color, variant = 'primary') => {
    const colorMap = {
      indigo: {
        primary: 'bg-indigo-600 hover:bg-indigo-700 text-white',
        secondary: 'bg-indigo-100 text-indigo-800 border-indigo-200',
        accent: 'text-indigo-600'
      },
      yellow: {
        primary: 'bg-yellow-500 hover:bg-yellow-600 text-white',
        secondary: 'bg-yellow-100 text-yellow-800 border-yellow-200',
        accent: 'text-yellow-600'
      },
      green: {
        primary: 'bg-green-600 hover:bg-green-700 text-white',
        secondary: 'bg-green-100 text-green-800 border-green-200',
        accent: 'text-green-600'
      },
      purple: {
        primary: 'bg-purple-600 hover:bg-purple-700 text-white',
        secondary: 'bg-purple-100 text-purple-800 border-purple-200',
        accent: 'text-purple-600'
      },
      blue: {
        primary: 'bg-blue-600 hover:bg-blue-700 text-white',
        secondary: 'bg-blue-100 text-blue-800 border-blue-200',
        accent: 'text-blue-600'
      }
    };
    return colorMap[color]?.[variant] || colorMap.indigo[variant];
  };

  const handleFeatureSelect = (featureId) => {
    const feature = features.find(f => f.id === featureId);
    if (feature.status === 'ready') {
      setActiveFeature(featureId);
    }
  };

  const renderFeatureContent = () => {
    switch (activeFeature) {
      case 'hidden-answers':
        return <HiddenAnswers />;
      case 'quick-respond':
        return (
          <div className="text-center py-12">
            <Zap className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Quick Respond</h3>
            <p className="text-gray-500 mb-4">Generate instant, contextual responses to interview questions</p>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 max-w-md mx-auto">
              <p className="text-yellow-800 text-sm">This feature is coming soon! It will integrate with the overlay system to provide quick AI responses.</p>
            </div>
          </div>
        );
      case 'summarization':
        return (
          <div className="text-center py-12">
            <FileText className="w-16 h-16 text-green-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Smart Summarization</h3>
            <p className="text-gray-500 mb-4">Summarize conversations and extract key information</p>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 max-w-md mx-auto">
              <p className="text-green-800 text-sm">Coming soon! This will help you summarize long discussions and highlight important points in your overlay.</p>
            </div>
          </div>
        );
      case 'hands-free':
        return (
          <div className="text-center py-12">
            <Mic className="w-16 h-16 text-purple-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Hands Free Mode</h3>
            <p className="text-gray-500 mb-4">Voice-activated AI assistance for seamless interviews</p>
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 max-w-md mx-auto">
              <p className="text-purple-800 text-sm">Voice control integration coming soon! Activate AI features without touching your keyboard.</p>
            </div>
          </div>
        );
      case 'key-insights':
        return (
          <div className="text-center py-12">
            <Brain className="w-16 h-16 text-blue-500 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-700 mb-2">Key Insights</h3>
            <p className="text-gray-500 mb-4">AI-powered analysis and insight extraction</p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 max-w-md mx-auto">
              <p className="text-blue-800 text-sm">Advanced AI analysis coming soon! Extract key insights and talking points automatically.</p>
            </div>
          </div>
        );
      default:
        return <HiddenAnswers />;
    }
  };

  return (
    <div className="main-feature-container min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">AI Interview Assistant</h1>
              <p className="text-gray-600 mt-1">Private AI features for interview success</p>
            </div>
            <div className="flex items-center gap-4">
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:text-gray-900 transition-colors"
              >
                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                {isExpanded ? 'Collapse' : 'Expand'}
              </button>
              <Settings className="w-5 h-5 text-gray-400 hover:text-gray-600 cursor-pointer transition-colors" />
            </div>
          </div>
        </div>
      </div>

      {/* Feature Selection Tabs */}
      {isExpanded && (
        <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-6">
            <div className="flex overflow-x-auto scrollbar-hide">
              {features.map((feature) => {
                const Icon = feature.icon;
                const isActive = activeFeature === feature.id;
                const isAvailable = feature.status === 'ready';
                
                return (
                  <button
                    key={feature.id}
                    onClick={() => handleFeatureSelect(feature.id)}
                    disabled={!isAvailable}
                    className={`
                      flex items-center gap-3 px-6 py-4 border-b-2 transition-all duration-200 whitespace-nowrap
                      ${isActive 
                        ? `border-${feature.color}-500 ${getColorClasses(feature.color, 'accent')} bg-${feature.color}-50` 
                        : 'border-transparent text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                      }
                      ${!isAvailable ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                    `}
                  >
                    <Icon className="w-5 h-5" />
                    <div className="text-left">
                      <div className="font-medium">{feature.name}</div>
                      {!isAvailable && (
                        <div className="text-xs text-gray-400">Coming Soon</div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Feature Status Bar */}
      <div className="bg-gray-50 border-b border-gray-200 py-2">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              <span className="text-gray-600">
                Active Feature: <span className="font-medium text-gray-900">
                  {features.find(f => f.id === activeFeature)?.name || 'Hidden Answers'}
                </span>
              </span>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-green-700">Ready</span>
              </div>
            </div>
            <div className="flex items-center gap-3 text-gray-500">
              <span>All outputs → Private Overlay</span>
              <div className="w-1 h-4 bg-gray-300"></div>
              <span>Screen Share Safe</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {renderFeatureContent()}
      </div>

      {/* Footer Info */}
      <div className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-gray-600">
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">How It Works</h4>
              <p>All AI features send their outputs to a private overlay window that only you can see. Your interviewer will never see the AI assistance on their shared screen.</p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">Global Hotkeys</h4>
              <ul className="space-y-1">
                <li><code className="bg-gray-100 px-1 rounded">Ctrl+Shift+Space</code> - Toggle overlay</li>
                <li><code className="bg-gray-100 px-1 rounded">Ctrl+Shift+O</code> - Toggle click-through</li>
                <li><code className="bg-gray-100 px-1 rounded">Ctrl+Shift+T</code> - Toggle theme</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">Integration Status</h4>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>Electron Overlay</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span>IPC Communication</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                  <span>AI Features (Expanding)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainFeature;