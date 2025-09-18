import React, { useState, useEffect, useRef } from 'react';
import { Send, Eye, EyeOff, Settings, Trash2, Copy, Check, Circle } from 'lucide-react';
import { HiddenAnswers } from '../../services/mainFeature';

const HiddenAnswers = () => {
  const [answer, setAnswer] = useState('');
  const [isOverlayVisible, setIsOverlayVisible] = useState(true);
  const [overlaySettings, setOverlaySettings] = useState({
    position: 'top-right',
    opacity: 0.9,
    theme: 'dark'
  });
  const [isCopied, setIsCopied] = useState(false);
  const [isElectronAvailable, setIsElectronAvailable] = useState(false);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (window.agenda && window.agenda.sendAnswer) {
      setIsElectronAvailable(true);
    }
  }, []);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [answer]);

  const sendToOverlay = () => {
    if (!answer.trim()) {
      alert('Please enter an answer to send to the overlay.');
      return;
    }

    try {
      if (isElectronAvailable) {
        // Sending to overlay via ipc
        window.agenda.sendAnswer({
          text: answer,
          timestamp: new Date().toISOString(),
          settings: overlaySettings
        });
        console.log('Answer sent to overlay:', answer);
      } else {
        console.log('Electron is not available - answer would be:', answer);
        alert('Overlay feature requires Electron app. Answer logged to console.');
      }
    } catch (error) {
      console.error('Error sending answer to overlay:', error);
      alert('Failed to send answer to overlay.');
    }
  };

  const clearAnswer = () => {
    setAnswer('');
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const copyToClipboard = async () => {
    if (!answer.trim()) return;

    try {
      await navigator.clipboard.writeText(answer);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const toggleOverlayVisibility = () => {
    setIsOverlayVisible(!isOverlayVisible);
    if (isElectronAvailable && window.agenda.toggleOverlay) {
      window.agenda.toggleOverlay();
    }
  };

  const handleKeyDown = (e) => {
    // Ctrl + Enter to send
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      sendToOverlay();
    }
    // Ctrl + Shift + C to clear
    if (e.ctrlKey && e.shiftKey && e.key === 'C') {
      e.preventDefault();
      clearAnswer();
    }
  };

  const formatAnswer = (text) => {
    return text
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0)
      .map((line, index) => {
        if (line.startsWith('.') || line.startsWith('-') || line.startsWith('*')) {
          return `<li class="numbered-point">${line}</li>`;
        }
        if (/^\d+\./.test(line)) {
          return `<li class="numbered-point">${line}</li>`;
        }
        return `<p>${line}</p>`;
      })
      .join('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 p-6">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="bg-slate-800/50 backdrop-blur p-8 rounded-xl border border-slate-700">
          <div className="flex items-center space-x-4 mb-4">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl flex items-center justify-center">
              <Eye size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-100">Hidden Answers</h1>
              <p className="text-slate-400">Send answers to your private overlay. Only you can see them during screen sharing.</p>
            </div>
          </div>
          
          {!isElectronAvailable && (
            <div className="bg-amber-900/20 border border-amber-600/30 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <Circle size={8} className="text-amber-500 fill-current" />
                <span className="text-amber-400 font-medium">Electron app required for overlay feature</span>
              </div>
            </div>
          )}
        </div>

        {/* Status indicators */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
            <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
              <Settings className="mr-2" size={20} />
              System Status
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/50">
                <span className="text-slate-300 font-medium">Electron Connection</span>
                <div className="flex items-center space-x-2">
                  <Circle 
                    size={8} 
                    className={`fill-current ${isElectronAvailable ? 'text-green-500' : 'text-red-500'}`} 
                  />
                  <span className={`text-sm font-medium ${isElectronAvailable ? 'text-green-400' : 'text-red-400'}`}>
                    {isElectronAvailable ? 'Connected' : 'Unavailable'}
                  </span>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/50">
                <span className="text-slate-300 font-medium">Overlay Status</span>
                <div className="flex items-center space-x-2">
                  {isOverlayVisible ? <Eye className="w-4 h-4 text-green-500" /> : <EyeOff className="w-4 h-4 text-slate-500" />}
                  <span className={`text-sm font-medium ${isOverlayVisible ? 'text-green-400' : 'text-slate-400'}`}>
                    {isOverlayVisible ? 'Visible' : 'Hidden'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
            <h3 className="text-lg font-semibold mb-4 text-slate-200">Quick Stats</h3>
            <div className="space-y-4">
              <div className="flex justify-between">
                <span className="text-slate-300">Characters</span>
                <span className="text-slate-400 font-mono">{answer.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-300">Words</span>
                <span className="text-slate-400 font-mono">{answer.trim() ? answer.trim().split(/\s+/).length : 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-300">Lines</span>
                <span className="text-slate-400 font-mono">{answer.split('\n').length}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Main input area */}
        <div className="bg-slate-800/50 backdrop-blur p-8 rounded-xl border border-slate-700 space-y-6">
          <h3 className="text-xl font-semibold text-slate-200">Answer Input</h3>
          
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter your answer or AI-generated response here...

Tips:
• Use bullet points for key insights
• Press Ctrl+Enter to send quickly
• Keep answers concise for better overlay display"
              className="w-full min-h-48 max-h-96 p-6 bg-slate-900/80 border-2 border-slate-700 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-colors resize-none text-slate-100 placeholder-slate-500"
              style={{ lineHeight: '1.6' }}
            />
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap gap-4">
            <button
              onClick={sendToOverlay}
              disabled={!answer.trim() || !isElectronAvailable}
              className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg hover:from-purple-700 hover:to-purple-800 disabled:from-slate-600 disabled:to-slate-700 disabled:cursor-not-allowed transition-all font-medium shadow-lg"
            >
              <Send className="w-5 h-5" />
              Send to Overlay
            </button>

            <button
              onClick={toggleOverlayVisibility}
              disabled={!isElectronAvailable}
              className="flex items-center gap-3 px-6 py-3 bg-slate-700 text-white rounded-lg hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {isOverlayVisible ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              {isOverlayVisible ? 'Hide Overlay' : 'Show Overlay'}
            </button>

      <button
  onClick={copyToClipboard}
  className="flex items-center gap-3 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium"
>
  {isCopied ? <Check className="w-5 h-5" /> : <Copy className="w-5 h-5" />}
  {isCopied ? 'Copied!' : 'Copy Text'}
</button>

<button
  onClick={clearAnswer}
  className="flex items-center gap-3 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
>
  <Trash2 className="w-5 h-5" />
  Clear
</button>

          </div>
        </div>

        {/* Overlay settings panel */}
        <div className="bg-slate-800/50 backdrop-blur p-8 rounded-xl border border-slate-700">
          <div className="flex items-center gap-3 mb-6">
            <Settings className="w-6 h-6 text-slate-400" />
            <h3 className="text-xl font-semibold text-slate-200">Overlay Settings</h3>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="bg-slate-900/50 p-6 rounded-lg">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Position
              </label>
              <select
                value={overlaySettings.position}
                onChange={(e) => setOverlaySettings(prev => ({ ...prev, position: e.target.value }))}
                className="w-full p-3 bg-slate-800 border border-slate-600 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 text-slate-100"
              >
                <option value="top-left">Top Left</option>
                <option value="top-right">Top Right</option>
                <option value="bottom-left">Bottom Left</option>
                <option value="bottom-right">Bottom Right</option>
                <option value="center">Center</option>
              </select>
            </div>

            <div className="bg-slate-900/50 p-6 rounded-lg">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Opacity ({Math.round(overlaySettings.opacity * 100)}%)
              </label>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.1"
                value={overlaySettings.opacity}
                onChange={(e) => setOverlaySettings(prev => ({ ...prev, opacity: parseFloat(e.target.value) }))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-slate-500 mt-2">
                <span>10%</span>
                <span>100%</span>
              </div>
            </div>

            <div className="bg-slate-900/50 p-6 rounded-lg">
              <label className="block text-sm font-medium text-slate-300 mb-3">
                Theme
              </label>
              <select
                value={overlaySettings.theme}
                onChange={(e) => setOverlaySettings(prev => ({ ...prev, theme: e.target.value }))}
                className="w-full p-3 bg-slate-800 border border-slate-600 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 text-slate-100"
              >
                <option value="dark">Dark</option>
                <option value="light">Light</option>
                <option value="auto">Auto</option>
              </select>
            </div>
          </div>
        </div>

        {/* Keyboard shortcuts help */}
        <div className="bg-gradient-to-r from-amber-900/20 to-orange-900/20 backdrop-blur p-6 rounded-xl border border-amber-600/30">
          <h4 className="text-lg font-semibold text-amber-300 mb-4 flex items-center">
            <Circle size={8} className="text-amber-500 fill-current mr-2" />
            Keyboard Shortcuts
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-amber-200">Send to overlay</span>
                <code className="bg-amber-900/30 text-amber-300 px-2 py-1 rounded text-xs font-mono">Ctrl+Enter</code>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-amber-200">Clear text</span>
                <code className="bg-amber-900/30 text-amber-300 px-2 py-1 rounded text-xs font-mono">Ctrl+Shift+C</code>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-amber-200">Toggle overlay (global)</span>
                <code className="bg-amber-900/30 text-amber-300 px-2 py-1 rounded text-xs font-mono">Ctrl+Shift+Space</code>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-amber-200">Toggle theme (global)</span>
                <code className="bg-amber-900/30 text-amber-300 px-2 py-1 rounded text-xs font-mono">Ctrl+Shift+T</code>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #9333ea;
          cursor: pointer;
        }
        .slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          border-radius: 50%;
          background: #9333ea;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  );
};

export default HiddenAnswers;