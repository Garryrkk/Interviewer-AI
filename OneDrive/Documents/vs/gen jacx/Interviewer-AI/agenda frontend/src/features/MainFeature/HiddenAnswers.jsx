import React, { useState, useEffect, useRef } from 'react';
import { Send, Eye, EyeOff, Settings, Trash2, Copy, Check } from 'lucide-react';

const HiddenAnswers = () => {
  const [answer, setAnswer] = useState('');
  const [isOverlayVisible, setIsOverlayVisible] = useState(true); // Fixed typo
  const [overlaySettings, setOverlaySettings] = useState({
    position: 'top-right',
    opacity: 0.9,
    theme: 'dark'
  });
  const [isCopied, setIsCopied] = useState(false); // Fixed camelCase
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
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px'; // Fixed typo
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
          settings: overlaySettings // Fixed capitalization
        });
        console.log('Answer sent to overlay:', answer);
      } else {
        console.log('Electron is not available - answer would be:', answer);
        alert('Overlay feature requires Electron app. Answer logged to console.');
      }
    } catch (error) {
      console.error('Error sending answer to overlay:', error); // Fixed string concatenation
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
      setTimeout(() => setIsCopied(false), 2000); // Fixed function name
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
    if (e.ctrlKey && e.shiftKey && e.key === 'C') { // Fixed key code
      e.preventDefault();
      clearAnswer();
    }
  };

  const formatAnswer = (text) => {
    return text
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0) // Fixed typo
      .map((line, index) => {
        if (line.startsWith('.') || line.startsWith('-') || line.startsWith('*')) {
          return `<li class="numbered-point">${line}</li>`; // Fixed template literal
        }
        if (/^\d+\./.test(line)) {
          return `<li class="numbered-point">${line}</li>`; // Fixed template literal and removed extra >
        }
        return `<p>${line}</p>`; // Added return for non-list items
      })
      .join('');
  };

  return (
    <div className="hidden-answers-container max-w-4xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg shadow-lg">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Hidden Answers
        </h2>
        <p className="text-gray-600 text-sm">
          Send answers to your private overlay. Only you can see them during screen sharing.
          {!isElectronAvailable && (
            <span className="text-amber-600 font-medium ml-2">
              (Electron app required for overlay feature)
            </span>
          )}
        </p>
      </div>

      {/* Status indicators */}
      <div className="flex items-center gap-4 mb-4 p-3 bg-white/50 rounded-lg">
        <div className="flex items-center gap-2">
          <div className={`w-3 h-3 rounded-full ${isElectronAvailable ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm font-medium">
            Electron: {isElectronAvailable ? 'Connected' : 'Unavailable'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {isOverlayVisible ? <Eye className="w-4 h-4 text-green-600" /> : <EyeOff className="w-4 h-4 text-gray-400" />}
          <span className="text-sm font-medium">
            Overlay: {isOverlayVisible ? 'Visible' : 'Hidden'}
          </span>
        </div>
      </div>

      {/* Main input area */}
      <div className="space-y-4">
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
            className="w-full min-h-40 max-h-80 p-4 border-2 border-indigo-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-colors resize-none"
            style={{ lineHeight: '1.5' }}
          />
          <div className="absolute bottom-3 right-3 text-xs text-gray-400">
            {answer.length} characters
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex flex-wrap gap-3">
          <button
            onClick={sendToOverlay}
            disabled={!answer.trim() || !isElectronAvailable}
            className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
            Send to Overlay
          </button>

          <button
            onClick={toggleOverlayVisibility}
            disabled={!isElectronAvailable}
            className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isOverlayVisible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            {isOverlayVisible ? 'Hide Overlay' : 'Show Overlay'}
          </button>

          <button
            onClick={copyToClipboard}
            disabled={!answer.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isCopied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
            {isCopied ? 'Copied!' : 'Copy Text'}
          </button>

          <button
            onClick={clearAnswer}
            disabled={!answer.trim()}
            className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear
          </button>
        </div>
      </div>

      {/* Overlay settings panel */}
      <div className="mt-6 p-4 bg-white/60 rounded-lg">
        <div className="flex items-center gap-2 mb-3">
          <Settings className="w-4 h-4 text-gray-600" />
          <h3 className="font-semibold text-gray-700">Overlay Settings</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Position
            </label>
            <select
              value={overlaySettings.position}
              onChange={(e) => setOverlaySettings(prev => ({ ...prev, position: e.target.value }))}
              className="w-full p-2 border border-gray-300 rounded-md focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="top-left">Top Left</option>
              <option value="top-right">Top Right</option>
              <option value="bottom-left">Bottom Left</option>
              <option value="bottom-right">Bottom Right</option>
              <option value="center">Center</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Opacity ({Math.round(overlaySettings.opacity * 100)}%)
            </label>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.1"
              value={overlaySettings.opacity}
              onChange={(e) => setOverlaySettings(prev => ({ ...prev, opacity: parseFloat(e.target.value) }))}
              className="w-full"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Theme
            </label>
            <select
              value={overlaySettings.theme}
              onChange={(e) => setOverlaySettings(prev => ({ ...prev, theme: e.target.value }))}
              className="w-full p-2 border border-gray-300 rounded-md focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            >
              <option value="dark">Dark</option>
              <option value="light">Light</option>
              <option value="auto">Auto</option>
            </select>
          </div>
        </div>
      </div>

      {/* Keyboard shortcuts help */}
      <div className="mt-4 p-3 bg-amber-50 rounded-lg border border-amber-200">
        <h4 className="font-semibold text-amber-800 mb-2">Keyboard Shortcuts</h4>
        <div className="text-sm text-amber-700 space-y-1">
          <div><code className="bg-amber-100 px-1 rounded">Ctrl+Enter</code> - Send to overlay</div>
          <div><code className="bg-amber-100 px-1 rounded">Ctrl+Shift+C</code> - Clear text</div>
          <div><code className="bg-amber-100 px-1 rounded">Ctrl+Shift+Space</code> - Toggle overlay visibility (global)</div>
          <div><code className="bg-amber-100 px-1 rounded">Ctrl+Shift+O</code> - Toggle click-through (global)</div>
          <div><code className="bg-amber-100 px-1 rounded">Ctrl+Shift+T</code> - Toggle theme (global)</div>
        </div>
      </div>
    </div>
  );
};

export default HiddenAnswers;