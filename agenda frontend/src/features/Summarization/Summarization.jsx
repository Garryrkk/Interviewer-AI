import React, { useState, useRef, useEffect } from 'react';
import { 
  Mic, 
  MicOff, 
  Copy, 
  Download, 
  Play, 
  Pause,
  RefreshCw,
  CheckSquare,
  BarChart3,
  Settings,
  Volume2,
  Square,
  Users
} from 'lucide-react';

const AudioMeetingApp = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioData, setAudioData] = useState(null);
  const [summary, setSummary] = useState('');
  const [transcript, setTranscript] = useState('');
  const [loading, setLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [meetingStatus, setMeetingStatus] = useState('idle'); // idle, recording, processing, complete
  
  // Settings
  const [autoSummarize, setAutoSummarize] = useState(true);
  const [includeTimestamps, setIncludeTimestamps] = useState(false);
  const [summaryLength, setSummaryLength] = useState('medium');
  const [summaryStyle, setSummaryStyle] = useState('bullet-points');
  
  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);
  const timerRef = useRef(null);

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setAudioData(event.data);
        }
      };
      
      mediaRecorderRef.current.onstop = () => {
        processAudio();
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setMeetingStatus('recording');
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
      
      // Simulate audio level detection
      const levelInterval = setInterval(() => {
        setAudioLevel(Math.random() * 100);
      }, 100);
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    setIsRecording(false);
    setMeetingStatus('processing');
    
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
  };

  // Process audio (simulate AI processing)
  const processAudio = () => {
    setLoading(true);
    
    // Simulate processing time
    setTimeout(() => {
      // Mock transcript
      const mockTranscript = "Welcome everyone to today's meeting. Let's start by reviewing the quarterly results. Sales have increased by 15% compared to last quarter. We need to discuss the marketing strategy for the next quarter and allocate resources accordingly. The development team has completed the main features and we're on track for the product launch.";
      
      // Mock summary based on settings
      let mockSummary = "";
      if (summaryStyle === 'bullet-points') {
        mockSummary = "• Quarterly results reviewed - 15% sales increase\n• Marketing strategy discussion needed for next quarter\n• Resource allocation to be determined\n• Development team completed main features\n• Product launch on track";
      } else if (summaryStyle === 'key-highlights') {
        mockSummary = "Key Highlights:\n1. Sales increased 15% this quarter\n2. Marketing strategy planning required\n3. Product launch remains on schedule";
      } else {
        mockSummary = "The meeting covered quarterly results showing a 15% increase in sales. Discussion focused on marketing strategy for the upcoming quarter and resource allocation. The development team reported completion of main features with the product launch remaining on track.";
      }
      
      if (includeTimestamps) {
        mockSummary = `[${new Date().toLocaleTimeString()}] ` + mockSummary;
      }
      
      setTranscript(mockTranscript);
      setSummary(mockSummary);
      setLoading(false);
      setMeetingStatus('complete');
    }, 3000);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleCopy = async (text) => {
    await navigator.clipboard.writeText(text);
  };

  const handleDownload = (content, filename) => {
    const element = document.createElement('a');
    const file = new Blob([content], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = filename;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const clearAll = () => {
    setAudioData(null);
    setSummary('');
    setTranscript('');
    setRecordingTime(0);
    setMeetingStatus('idle');
    setAudioLevel(0);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="max-w-7xl mx-auto p-6">
        
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-slate-300 bg-clip-text text-transparent mb-2">
                AI Meeting Assistant
              </h1>
              <p className="text-slate-400 text-lg">Record, transcribe, and summarize your meetings in real-time</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                meetingStatus === 'recording' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                meetingStatus === 'processing' ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30' : 
                meetingStatus === 'complete' ? 'bg-green-500/20 text-green-300 border border-green-500/30' : 
                'bg-slate-500/20 text-slate-300 border border-slate-500/30'
              }`}>
                {meetingStatus === 'recording' ? (
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
                    <span>Recording {formatTime(recordingTime)}</span>
                  </div>
                ) : meetingStatus === 'processing' ? (
                  <div className="flex items-center space-x-2">
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Processing</span>
                  </div>
                ) : meetingStatus === 'complete' ? (
                  <div className="flex items-center space-x-2">
                    <CheckSquare className="w-4 h-4" />
                    <span>Complete</span>
                  </div>
                ) : (
                  'Ready'
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* Recording and Results Section - Takes 3 columns */}
          <div className="lg:col-span-3 space-y-6">
            
            {/* Recording Control */}
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <Users className="w-5 h-5 mr-2 text-blue-400" />
                  Live Meeting Recording
                </h2>
                <div className="text-sm text-slate-400">
                  {formatTime(recordingTime)}
                </div>
              </div>
              
              {/* Recording Interface */}
              <div className="text-center py-8">
                <div className="mb-6">
                  <div className={`w-32 h-32 mx-auto rounded-full border-4 transition-all duration-300 flex items-center justify-center ${
                    isRecording 
                      ? 'border-red-500 bg-red-500/10 shadow-lg shadow-red-500/20' 
                      : 'border-slate-600 bg-slate-700/50'
                  }`}>
                    <button
                      onClick={isRecording ? stopRecording : startRecording}
                      disabled={loading}
                      className={`w-20 h-20 rounded-full transition-all duration-300 flex items-center justify-center ${
                        isRecording 
                          ? 'bg-red-500 hover:bg-red-600 shadow-lg' 
                          : loading 
                            ? 'bg-slate-600 cursor-not-allowed' 
                            : 'bg-blue-600 hover:bg-blue-700 shadow-lg'
                      }`}
                    >
                      {loading ? (
                        <RefreshCw className="w-8 h-8 animate-spin text-white" />
                      ) : isRecording ? (
                        <Square className="w-8 h-8 text-white" />
                      ) : (
                        <Mic className="w-8 h-8 text-white" />
                      )}
                    </button>
                  </div>
                  
                  {/* Audio Level Indicator */}
                  {isRecording && (
                    <div className="mt-4">
                      <div className="w-64 h-2 mx-auto bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-green-500 to-yellow-500 transition-all duration-100"
                          style={{ width: `${audioLevel}%` }}
                        ></div>
                      </div>
                      <p className="text-slate-400 text-sm mt-2">Audio Level</p>
                    </div>
                  )}
                </div>
                
                <div className="space-y-2">
                  <h3 className="text-lg font-medium text-white">
                    {isRecording ? 'Recording in Progress' : loading ? 'Processing Audio' : 'Ready to Record'}
                  </h3>
                  <p className="text-slate-400">
                    {isRecording 
                      ? 'Click the square button to stop recording' 
                      : loading 
                        ? 'Generating transcript and summary...' 
                        : 'Click the microphone to start recording your meeting'
                    }
                  </p>
                </div>
              </div>

              <div className="flex items-center justify-center space-x-4 pt-4 border-t border-slate-700/50">
                <button
                  onClick={clearAll}
                  className="px-4 py-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded-lg transition-all font-medium border border-slate-600/30"
                >
                  Clear All
                </button>
              </div>
            </div>

            {/* Transcript */}
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <Volume2 className="w-5 h-5 mr-2 text-purple-400" />
                  Live Transcript
                </h2>
                {transcript && (
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleCopy(transcript)}
                      className="p-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded-lg transition-all border border-slate-600/30"
                      title="Copy transcript"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDownload(transcript, 'transcript.txt')}
                      className="p-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded-lg transition-all border border-slate-600/30"
                      title="Download transcript"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
              
              <div className="bg-slate-900/60 border border-slate-700/30 rounded-xl p-6 min-h-[200px]">
                {transcript ? (
                  <div className="space-y-4">
                    <p className="text-slate-200 leading-relaxed">{transcript}</p>
                    <div className="pt-4 border-t border-slate-700/50">
                      <div className="text-sm text-slate-400">
                        Transcript generated • {transcript.length} characters
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <Volume2 className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                      <p className="text-slate-400 text-lg">Live transcript will appear here...</p>
                      <p className="text-slate-500 text-sm mt-2">Start recording to see real-time transcription</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Summary Output */}
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-green-400" />
                  AI Summary
                </h2>
                {summary && (
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleCopy(summary)}
                      className="p-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded-lg transition-all border border-slate-600/30"
                      title="Copy summary"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDownload(summary, 'summary.txt')}
                      className="p-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 rounded-lg transition-all border border-slate-600/30"
                      title="Download summary"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  </div>
                )}
              </div>
              
              <div className="bg-slate-900/60 border border-slate-700/30 rounded-xl p-6 min-h-[200px]">
                {summary ? (
                  <div className="space-y-4">
                    <div className="prose prose-slate max-w-none">
                      <pre className="text-slate-200 leading-relaxed whitespace-pre-wrap font-sans">{summary}</pre>
                    </div>
                    <div className="pt-4 border-t border-slate-700/50">
                      <div className="text-sm text-slate-400">
                        Summary generated • {summary.length} characters
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <BarChart3 className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                      <p className="text-slate-400 text-lg">AI-generated summary will appear here...</p>
                      <p className="text-slate-500 text-sm mt-2">Complete a recording to get an intelligent summary</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Settings Sidebar - Takes 1 column */}
          <div className="space-y-6">
            
            {/* Recording Settings */}
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-xl">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Settings className="w-5 h-5 mr-2 text-purple-400" />
                Settings
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Summary Length
                  </label>
                  <select
                    value={summaryLength}
                    onChange={(e) => setSummaryLength(e.target.value)}
                    className="w-full bg-slate-900/60 border border-slate-600/50 text-white rounded-lg px-3 py-2 focus:border-purple-500/50 focus:outline-none focus:ring-2 focus:ring-purple-500/20"
                  >
                    <option value="short">Short (2-3 sentences)</option>
                    <option value="medium">Medium (1 paragraph)</option>
                    <option value="long">Long (2-3 paragraphs)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-2">
                    Summary Style
                  </label>
                  <select
                    value={summaryStyle}
                    onChange={(e) => setSummaryStyle(e.target.value)}
                    className="w-full bg-slate-900/60 border border-slate-600/50 text-white rounded-lg px-3 py-2 focus:border-purple-500/50 focus:outline-none focus:ring-2 focus:ring-purple-500/20"
                  >
                    <option value="paragraph">Paragraph</option>
                    <option value="bullet-points">Bullet Points</option>
                    <option value="key-highlights">Key Highlights</option>
                  </select>
                </div>

                <div className="space-y-3 pt-2">
                  <label className="flex items-center space-x-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={autoSummarize}
                      onChange={(e) => setAutoSummarize(e.target.checked)}
                      className="w-4 h-4 rounded bg-slate-700 border-slate-600 text-purple-600 focus:ring-purple-500 focus:ring-offset-0"
                    />
                    <span className="text-slate-300 text-sm">Auto-summarize after recording</span>
                  </label>
                  <label className="flex items-center space-x-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includeTimestamps}
                      onChange={(e) => setIncludeTimestamps(e.target.checked)}
                      className="w-4 h-4 rounded bg-slate-700 border-slate-600 text-purple-600 focus:ring-purple-500 focus:ring-offset-0"
                    />
                    <span className="text-slate-300 text-sm">Include timestamps</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Session Stats */}
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-xl">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
                Session Stats
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 text-sm">Recording Time</span>
                  <span className="text-white font-medium">{formatTime(recordingTime)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 text-sm">Transcript Length</span>
                  <span className="text-white font-medium">{transcript.length.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 text-sm">Summary Length</span>
                  <span className="text-white font-medium">{summary.length.toLocaleString()}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400 text-sm">Status</span>
                  <span className={`font-medium capitalize ${
                    meetingStatus === 'recording' ? 'text-red-400' :
                    meetingStatus === 'processing' ? 'text-orange-400' :
                    meetingStatus === 'complete' ? 'text-green-400' :
                    'text-slate-400'
                  }`}>
                    {meetingStatus}
                  </span>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-slate-800/40 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-xl">
              <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
              <div className="space-y-3">
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={loading}
                  className={`w-full py-3 px-4 rounded-lg font-medium transition-all flex items-center justify-center space-x-2 ${
                    loading 
                      ? 'bg-slate-700/50 text-slate-500 cursor-not-allowed border border-slate-600/30'
                      : isRecording 
                        ? 'bg-red-600 hover:bg-red-700 text-white shadow-lg border border-red-500/30'
                        : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg border border-blue-500/30'
                  }`}
                >
                  {loading ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : isRecording ? (
                    <>
                      <Square className="w-4 h-4" />
                      <span>Stop Recording</span>
                    </>
                  ) : (
                    <>
                      <Mic className="w-4 h-4" />
                      <span>Start Recording</span>
                    </>
                  )}
                </button>
                
                <button
                  onClick={clearAll}
                  disabled={isRecording}
                  className={`w-full py-3 px-4 rounded-lg transition-all font-medium border ${
                    isRecording 
                      ? 'bg-slate-700/30 text-slate-500 cursor-not-allowed border-slate-600/20'
                      : 'bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 border-slate-600/30'
                  }`}
                >
                  Clear Session
                </button>

                {summary && (
                  <button
                    onClick={() => handleDownload(summary, 'meeting-summary.txt')}
                    className="w-full bg-green-600/20 hover:bg-green-600/30 text-green-300 py-3 px-4 rounded-lg transition-all font-medium border border-green-500/30"
                  >
                    Download Summary
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AudioMeetingApp;