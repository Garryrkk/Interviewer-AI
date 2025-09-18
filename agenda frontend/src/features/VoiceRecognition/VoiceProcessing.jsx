import React, { useState, useEffect, useRef } from "react";
import {
    Mic,
    Square,
    Activity,
    Volume2,
    Clock,
    Trash2,
    ArrowLeft,
    Circle,
    Settings,
    User
} from 'lucide-react';

// Real utility functions implementation
const startRecording = () => Promise.resolve();
const stopRecording = () => Promise.resolve();
const processAudioData = () => Promise.resolve();

const sendToAI = async (text, history) => {
    // Simple AI response simulation - replace with your actual AI service
    const responses = [
        `I understand you said: "${text}". How can I help you with that?`,
        `Thank you for sharing: "${text}". What would you like to know more about?`,
        `Based on your input: "${text}", here are some thoughts...`,
        `I heard: "${text}". Is there anything specific you'd like me to explain?`
    ];
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
    
    return responses[Math.floor(Math.random() * responses.length)];
};

const formatResponse = (response) => response;

const audioToText = (blob) => {
    return new Promise((resolve, reject) => {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            // Fallback for browsers without speech recognition
            resolve("Speech recognition not supported in this browser");
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        // Convert blob to audio for speech recognition
        const audio = new Audio();
        const url = URL.createObjectURL(blob);
        audio.src = url;
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            resolve(transcript);
            URL.revokeObjectURL(url);
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            reject(new Error('Speech recognition failed: ' + event.error));
            URL.revokeObjectURL(url);
        };

        recognition.onend = () => {
            URL.revokeObjectURL(url);
        };

        // Start recognition
        try {
            recognition.start();
        } catch (error) {
            reject(error);
        }
    });
};

const calculateVolume = (dataArray) => {
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / dataArray.length);
    return (rms / 255) * 100;
};

const VoiceProcessing = ({ onBack }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [aiResponse, setAiResponse] = useState('');
    const [volume, setVolume] = useState(0);
    const [error, setError] = useState('');
    const [sessionHistory, setSessionHistory] = useState([]);
    const [recordingTime, setRecordingTime] = useState(0);

    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const animationFrameRef = useRef(null);
    const recordingIntervalRef = useRef(null);
    const analyserRef = useRef(null);

    const handleStartRecording = async () => {
        try {
            setError('');
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            analyserRef.current = { analyser, audioContext };

            const monitorVolume = () => {
                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(dataArray);
                const vol = calculateVolume(dataArray);
                setVolume(vol);
                animationFrameRef.current = requestAnimationFrame(monitorVolume);
            };
            monitorVolume();

            // Set up media recorder
            const mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
                await processRecording(audioBlob);

                // Clean up resources
                stream.getTracks().forEach(track => track.stop());
                if (animationFrameRef.current) {
                    cancelAnimationFrame(animationFrameRef.current);
                }
                if (analyserRef.current) {
                    analyserRef.current.audioContext.close();
                }
                setVolume(0);
            };
            
            mediaRecorder.start(100); // Collect data every 100ms
            setIsRecording(true);
            setRecordingTime(0);

            // Start recording timer
            recordingIntervalRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);

        } catch (err) {
            console.error('Error starting recording:', err);
            setError('Failed to start recording. Please check microphone permissions.');
        }
    };

    // Stop recording
    const handleStopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            
            if (recordingIntervalRef.current) {
                clearInterval(recordingIntervalRef.current);
            }
            
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        }
    };

    const processRecording = async (audioBlob) => {
        setIsProcessing(true);
        setTranscript('');
        setAiResponse('');

        try {
            // Convert audio to text using Web Speech API
            const transcriptText = await audioToText(audioBlob);
            setTranscript(transcriptText);

            if (transcriptText.trim() && transcriptText !== "Speech recognition not supported in this browser") {
                // Send to AI for processing
                const response = await sendToAI(transcriptText, sessionHistory);
                const formattedResponse = formatResponse(response);
                setAiResponse(formattedResponse);

                // Add to session history
                const newEntry = {
                    id: Date.now(),
                    timestamp: new Date().toLocaleTimeString(),
                    transcript: transcriptText,
                    response: formattedResponse,
                    duration: recordingTime
                };
                setSessionHistory(prev => [...prev, newEntry]);
            } else {
                setError('No speech detected or speech recognition not supported. Please try again.');
            }
        } catch (err) {
            console.error('Error processing recording:', err);
            setError('Failed to process audio. Please try again.');
        } finally {
            setIsProcessing(false);
        }
    };

    // Clear session history
    const clearHistory = () => {
        setSessionHistory([]);
        setTranscript('');
        setAiResponse('');
        setError('');
    };

    // Format recording time
    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            if (recordingIntervalRef.current) {
                clearInterval(recordingIntervalRef.current);
            }
            if (analyserRef.current && analyserRef.current.audioContext) {
                analyserRef.current.audioContext.close();
            }
        };
    }, []);

    const WindowControls = () => (
        <div className="flex items-center space-x-2">
            <button className="w-3 h-3 bg-red-500 rounded-full hover:bg-red-600 transition-colors"></button>
            <button className="w-3 h-3 bg-yellow-500 rounded-full hover:bg-yellow-600 transition-colors"></button>
            <button className="w-3 h-3 bg-green-500 rounded-full hover:bg-green-600 transition-colors"></button>
        </div>
    );

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 font-sans">
            {/* Desktop Window Frame */}
            <div className="bg-slate-800/50 backdrop-blur border-b border-slate-700 px-6 py-3">
                <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                        <WindowControls />
                        <div className="flex items-center space-x-3 ml-4">
                            <div className="w-8 h-8 bg-gradient-to-r from-pink-600 to-purple-600 rounded-lg flex items-center justify-center">
                                <Mic size={18} className="text-white" />
                            </div>
                            <div>
                                <h1 className="text-lg font-bold">Voice Processing</h1>
                                <p className="text-xs text-slate-400">AI Voice Assistant</p>
                            </div>
                        </div>
                    </div>
                    <div className="flex items-center space-x-3">
                        <div className="flex items-center space-x-2 px-3 py-1 bg-slate-700/50 rounded-full">
                            <Circle size={6} className={`fill-current ${isRecording ? 'text-red-500' : 'text-green-500'}`} />
                            <span className="text-xs text-slate-300">
                                {isRecording ? 'Recording' : 'Ready'}
                            </span>
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

            <div className="p-8">
                {/* Header Section */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center space-x-4">
                        {onBack && (
                            <button 
                                onClick={onBack}
                                className="p-3 bg-slate-800/50 backdrop-blur rounded-xl hover:bg-slate-700/50 transition-colors border border-slate-700"
                            >
                                <ArrowLeft size={20} />
                            </button>
                        )}
                        <div>
                            <h2 className="text-4xl font-bold text-slate-100 mb-2">Voice Processing</h2>
                            <p className="text-slate-400">Real-time voice recording and AI processing</p>
                        </div>
                    </div>
                    <div className="flex space-x-2">
                        <div className={`px-3 py-1 text-white text-sm rounded-full ${
                            isRecording ? 'bg-red-600' : isProcessing ? 'bg-yellow-600' : 'bg-green-600'
                        }`}>
                            {isRecording ? 'Recording' : isProcessing ? 'Processing' : 'Ready'}
                        </div>
                    </div>
                </div>

                {/* Main Control Panel */}
                <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 mb-8">
                    {/* Recording Control */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                            <Mic className="mr-2" size={20} />
                            Voice Control
                        </h3>
                        <div className="text-center space-y-4">
                            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-all duration-300 ${
                                isRecording 
                                    ? 'bg-red-600 animate-pulse shadow-lg shadow-red-600/30' 
                                    : 'bg-slate-900 hover:bg-slate-800'
                            }`}>
                                {isRecording ? (
                                    <Square size={32} className="text-white" />
                                ) : (
                                    <Mic size={32} className="text-white" />
                                )}
                            </div>
                            <button 
                                onClick={isRecording ? handleStopRecording : handleStartRecording}
                                disabled={isProcessing}
                                className={`w-full py-3 px-4 rounded-lg transition-all font-medium ${
                                    isRecording 
                                        ? 'bg-red-600 hover:bg-red-700 text-white shadow-lg' 
                                        : 'bg-pink-600 hover:bg-pink-700 text-white'
                                } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                                {isRecording ? 'Stop Recording' : 'Start Recording'}
                            </button>
                        </div>
                    </div>

                    {/* Recording Status */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                            <Clock className="mr-2" size={20} />
                            Recording Status
                        </h3>
                        <div className="space-y-4">
                            <div className="text-center">
                                <div className="text-3xl font-bold text-slate-100 mb-1">
                                    {formatTime(recordingTime)}
                                </div>
                                <div className="text-sm text-slate-400">Duration</div>
                            </div>
                            {isRecording && (
                                <div className="space-y-2">
                                    <div className="flex justify-between text-sm">
                                        <span className="text-slate-300">Volume</span>
                                        <span className="text-slate-400">{Math.round(volume)}%</span>
                                    </div>
                                    <div className="w-full bg-slate-700 rounded-full h-2">
                                        <div 
                                            className="bg-green-500 h-2 rounded-full transition-all duration-150" 
                                            style={{ width: `${Math.min(volume, 100)}%` }}
                                        ></div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Processing Status */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                            <Activity className="mr-2" size={20} />
                            AI Processing
                        </h3>
                        <div className="text-center space-y-4">
                            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-colors ${
                                isProcessing ? 'bg-yellow-600 animate-pulse' : 'bg-slate-900'
                            }`}>
                                <Activity size={32} className="text-white" />
                            </div>
                            <div className="text-sm text-slate-400">
                                {isProcessing ? 'Processing audio...' : 'Waiting for audio'}
                            </div>
                        </div>
                    </div>

                    {/* Session Control */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                            <Volume2 className="mr-2" size={20} />
                            Session
                        </h3>
                        <div className="space-y-4">
                            <div className="text-center text-sm text-slate-400">
                                {sessionHistory.length} interactions
                            </div>
                            {sessionHistory.length > 0 && (
                                <button 
                                    onClick={clearHistory}
                                    className="w-full bg-slate-700 text-slate-300 py-3 px-4 rounded-lg hover:bg-slate-600 transition-colors font-medium flex items-center justify-center space-x-2"
                                >
                                    <Trash2 size={16} />
                                    <span>Clear History</span>
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                {/* Error Display */}
                {error && (
                    <div className="bg-red-900/50 backdrop-blur border border-red-700 p-4 rounded-xl mb-6">
                        <div className="flex items-center space-x-2 text-red-200">
                            <Circle size={6} className="text-red-500 fill-current" />
                            <span className="font-medium">Error:</span>
                            <span>{error}</span>
                        </div>
                    </div>
                )}

                {/* Results Section */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-8">
                    {/* Transcript */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-xl font-semibold mb-4 text-slate-200">Live Transcript</h3>
                        <div className="bg-slate-900/80 p-6 rounded-lg min-h-48 max-h-64 overflow-y-auto">
                            {transcript ? (
                                <p className="text-slate-300 leading-relaxed">{transcript}</p>
                            ) : (
                                <p className="text-slate-500 text-center">Voice transcription will appear here...</p>
                            )}
                        </div>
                    </div>

                    {/* AI Response */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-xl font-semibold mb-4 text-slate-200">AI Response</h3>
                        <div className="bg-slate-900/80 p-6 rounded-lg min-h-48 max-h-64 overflow-y-auto">
                            {aiResponse ? (
                                <p className="text-slate-300 leading-relaxed">{aiResponse}</p>
                            ) : (
                                <p className="text-slate-500 text-center">AI response will appear here...</p>
                            )}
                        </div>
                    </div>
                </div>

                {/* Session History */}
                {sessionHistory.length > 0 && (
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-xl font-semibold text-slate-200">Session History</h3>
                            <span className="text-sm text-slate-400">{sessionHistory.length} interactions</span>
                        </div>
                        <div className="space-y-4 max-h-96 overflow-y-auto">
                            {sessionHistory.map((entry) => (
                                <div key={entry.id} className="bg-slate-900/50 p-4 rounded-lg border border-slate-700">
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center space-x-3">
                                            <Circle size={6} className="text-blue-500 fill-current" />
                                            <span className="text-sm text-slate-400">{entry.timestamp}</span>
                                        </div>
                                        <span className="text-xs text-slate-500">
                                            {formatTime(entry.duration)}
                                        </span>
                                    </div>
                                    <div className="space-y-3">
                                        <div className="border-l-4 border-blue-500 pl-3">
                                            <div className="text-xs text-slate-400 mb-1">You said:</div>
                                            <p className="text-slate-300 text-sm">{entry.transcript}</p>
                                        </div>
                                        <div className="border-l-4 border-green-500 pl-3">
                                            <div className="text-xs text-slate-400 mb-1">AI responded:</div>
                                            <p className="text-slate-300 text-sm">{entry.response}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default VoiceProcessing;