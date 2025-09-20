import React, { useState, useEffect, useRef } from "react";
import { Mic, Activity, Volume2, ChevronLeft, Settings, Circle } from 'lucide-react';

// PreOpVoice Component
const PreOpVoice = ({ onComplete, micPermission, setMicPermission }) => {
    const [isPrepping, setIsPrepping] = useState(false);
    const [prepStatus, setPrepStatus] = useState('idle');
    
    const handleStartPrep = async () => {
        setIsPrepping(true);
        setPrepStatus('requesting');
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setMicPermission('granted');
            setPrepStatus('ready');
            
            // Stop the stream after getting permission
            stream.getTracks().forEach(track => track.stop());
            
            setTimeout(() => {
                setIsPrepping(false);
                onComplete();
            }, 1500);
        } catch (error) {
            setMicPermission('denied');
            setPrepStatus('error');
            setIsPrepping(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-4xl font-bold text-slate-100 mb-2">Voice Recognition Setup</h1>
                    <p className="text-slate-400">Prepare your microphone for voice recognition</p>
                </div>

                {/* Main Setup Card */}
                <div className="bg-slate-800/50 backdrop-blur p-8 rounded-xl border border-slate-700">
                    <div className="text-center space-y-6">
                        {/* Microphone Visual */}
                        <div className={`w-32 h-32 rounded-full flex items-center justify-center mx-auto transition-all duration-300 ${
                            isPrepping ? 'bg-pink-600 animate-pulse shadow-lg shadow-pink-600/30' : 'bg-slate-900 border-2 border-slate-600'
                        }`}>
                            <Mic size={48} className="text-white" />
                        </div>

                        {/* Status Display */}
                        <div className="space-y-3">
                            <h2 className="text-2xl font-semibold text-slate-200">
                                {prepStatus === 'idle' && 'Ready to Setup Microphone'}
                                {prepStatus === 'requesting' && 'Requesting Microphone Access...'}
                                {prepStatus === 'ready' && 'Microphone Ready!'}
                                {prepStatus === 'error' && 'Microphone Access Denied'}
                            </h2>
                            
                            <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium ${
                                micPermission === 'granted' ? 'bg-green-600/20 text-green-400 border border-green-600/30' :
                                micPermission === 'denied' ? 'bg-red-600/20 text-red-400 border border-red-600/30' :
                                'bg-slate-600/20 text-slate-400 border border-slate-600/30'
                            }`}>
                                <Circle 
                                    size={8} 
                                    className={`fill-current ${
                                        micPermission === 'granted' ? 'text-green-500' :
                                        micPermission === 'denied' ? 'text-red-500' :
                                        'text-slate-500'
                                    }`} 
                                />
                                <span>
                                    {micPermission === 'granted' ? 'Permission Granted' :
                                     micPermission === 'denied' ? 'Permission Denied' :
                                     'Permission Required'}
                                </span>
                            </div>
                        </div>

                        {/* Instructions */}
                        <div className="bg-slate-900/50 p-6 rounded-lg border border-slate-700">
                            <h3 className="text-lg font-medium text-slate-200 mb-3">Setup Instructions</h3>
                            <div className="text-slate-400 space-y-2 text-left">
                                <p>• Click "Start Prep" to begin microphone setup</p>
                                <p>• Allow microphone access when prompted</p>
                                <p>• Ensure your microphone is working properly</p>
                                <p>• Wait for confirmation before proceeding</p>
                            </div>
                        </div>

                        {/* Action Button */}
                        <button 
                            onClick={handleStartPrep}
                            disabled={isPrepping}
                            className={`py-4 px-8 rounded-xl transition-all font-medium text-lg ${
                                isPrepping 
                                    ? 'bg-slate-600 text-slate-400 cursor-not-allowed' 
                                    : 'bg-gradient-to-r from-pink-600 to-pink-700 hover:from-pink-700 hover:to-pink-800 text-white shadow-lg hover:shadow-pink-600/30'
                            }`}
                        >
                            {isPrepping ? 'Setting up...' : 'Start Prep'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

// VoiceProcessing Component
const VoiceProcessing = ({ onBackToSetup, isReady }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [transcription, setTranscription] = useState('');
    const [processingStatus, setProcessingStatus] = useState('ready');
    const [accuracy, setAccuracy] = useState(94);
    const [processingSpeed, setProcessingSpeed] = useState(87);
    const [audioQuality, setAudioQuality] = useState(91);
    const [interimTranscript, setInterimTranscript] = useState('');
    
    const recognitionRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const streamRef = useRef(null);

    useEffect(() => {
        // Initialize Speech Recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.webkitSpeechRecognition || window.SpeechRecognition;
            recognitionRef.current = new SpeechRecognition();
            
            recognitionRef.current.continuous = true;
            recognitionRef.current.interimResults = true;
            recognitionRef.current.lang = 'en-US';

            recognitionRef.current.onstart = () => {
                setProcessingStatus('recording');
                setTranscription('Listening for voice input...');
            };

            recognitionRef.current.onresult = (event) => {
                let finalTranscript = '';
                let interimTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    const confidence = event.results[i][0].confidence;
                    
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                        // Update accuracy based on confidence
                        setAccuracy(Math.round(confidence * 100));
                    } else {
                        interimTranscript += transcript;
                    }
                }

                if (finalTranscript) {
                    setTranscription(prev => prev + ' ' + finalTranscript);
                }
                setInterimTranscript(interimTranscript);
            };

            recognitionRef.current.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                setProcessingStatus('error');
                setTranscription('Error occurred during voice recognition: ' + event.error);
            };

            recognitionRef.current.onend = () => {
                setIsRecording(false);
                setIsProcessing(false);
                setProcessingStatus('completed');
            };
        }

        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    const startAudioRecording = async () => {
        try {
            streamRef.current = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });

            mediaRecorderRef.current = new MediaRecorder(streamRef.current, {
                mimeType: 'audio/webm;codecs=opus'
            });

            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                    // Analyze audio quality
                    analyzeAudioQuality(event.data);
                }
            };

            mediaRecorderRef.current.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                processAudioData(audioBlob);
            };

            mediaRecorderRef.current.start(100); // Collect data every 100ms
        } catch (error) {
            console.error('Error starting audio recording:', error);
        }
    };

    const analyzeAudioQuality = (audioData) => {
        // Simulate audio quality analysis based on data size
        const quality = Math.min(100, Math.max(60, audioData.size / 1000 * 20));
        setAudioQuality(Math.round(quality));
    };

    const processAudioData = async (audioBlob) => {
        setIsProcessing(true);
        setProcessingStatus('processing');
        
        // Simulate processing time based on audio size
        const processingTime = Math.max(500, audioBlob.size / 1000);
        const speedScore = Math.max(70, 100 - (processingTime / 100));
        setProcessingSpeed(Math.round(speedScore));

        // In a real implementation, you would send audioBlob to your backend
        // const formData = new FormData();
        // formData.append('audio', audioBlob);
        // const response = await fetch('/api/voice-recognition', {
        //     method: 'POST',
        //     body: formData
        // });
        
        setTimeout(() => {
            setIsProcessing(false);
        }, Math.min(processingTime, 2000));
    };

    const handleStartRecording = async () => {
        setIsRecording(true);
        setTranscription('');
        setInterimTranscript('');
        
        // Start both speech recognition and audio recording
        if (recognitionRef.current) {
            recognitionRef.current.start();
        }
        await startAudioRecording();
    };

    const handleStopRecording = () => {
        setIsRecording(false);
        
        if (recognitionRef.current) {
            recognitionRef.current.stop();
        }
        
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-8">
            <div className="max-w-6xl mx-auto">
                {/* Header with Back Button */}
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center space-x-4">
                        <button
                            onClick={onBackToSetup}
                            className="p-3 bg-slate-800/50 hover:bg-slate-700/70 rounded-xl border border-slate-700 transition-colors"
                        >
                            <ChevronLeft size={20} className="text-slate-300" />
                        </button>
                        <div>
                            <h1 className="text-4xl font-bold text-slate-100 mb-2">Voice Processing</h1>
                            <p className="text-slate-400">Real-time voice recognition and processing</p>
                        </div>
                    </div>
                    <div className="flex items-center space-x-2">
                        <div className={`px-3 py-1 text-white text-sm rounded-full ${
                            isReady ? 'bg-green-600' : 'bg-slate-600'
                        }`}>
                            {isReady ? 'Ready' : 'Not Ready'}
                        </div>
                    </div>
                </div>

                {/* Processing Grid */}
                <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
                    {/* Recording Control */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                            <Mic className="mr-2" size={20} />
                            Recording
                        </h3>
                        <div className="text-center space-y-4">
                            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-all duration-300 ${
                                isRecording ? 'bg-red-600 animate-pulse shadow-lg shadow-red-600/30' : 'bg-slate-900 border-2 border-slate-600'
                            }`}>
                                <Mic size={32} className="text-white" />
                            </div>
                           <button
                            onClick={isRecording ? handleStopRecording : handleStartRecording}
                            className={`w-full py-3 px-4 rounded-lg transition-all font-medium ${
                                isRecording
                                ? 'bg-red-600 hover:bg-red-700 shadow-lg hover:shadow-red-600/30'
                                : 'bg-pink-600 hover:bg-pink-700 shadow-lg hover:shadow-pink-600/30'
                            } text-white`}
                            >
                            {isRecording ? 'Stop Recording' : 'Start Recording'}
                            </button>

                        </div>
                    </div>

                    {/* Processing Status */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                            <Activity className="mr-2" size={20} />
                            Processing
                        </h3>
                        <div className="text-center space-y-4">
                            <div className={`w-20 h-20 rounded-full flex items-center justify-center mx-auto transition-all duration-300 ${
                                isProcessing ? 'bg-yellow-600 animate-pulse shadow-lg shadow-yellow-600/30' : 'bg-slate-900 border-2 border-slate-600'
                            }`}>
                                <Activity size={32} className="text-white" />
                            </div>
                            <div className={`px-3 py-1 text-white text-sm rounded-full ${
                                processingStatus === 'recording' ? 'bg-red-600' :
                                processingStatus === 'processing' ? 'bg-yellow-600' :
                                processingStatus === 'completed' ? 'bg-green-600' :
                                processingStatus === 'error' ? 'bg-red-600' :
                                'bg-slate-600'
                            }`}>
                                {processingStatus === 'recording' ? 'Recording' :
                                 processingStatus === 'processing' ? 'Processing' :
                                 processingStatus === 'completed' ? 'Completed' :
                                 processingStatus === 'error' ? 'Error' :
                                 'Ready'}
                            </div>
                        </div>
                    </div>

                    {/* Recognition Output */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200 flex items-center">
                            <Volume2 className="mr-2" size={20} />
                            Recognition
                        </h3>
                        <div className="text-center space-y-4">
                            <div className="w-20 h-20 bg-slate-900 border-2 border-slate-600 rounded-full flex items-center justify-center mx-auto">
                                <Volume2 size={32} className="text-slate-500" />
                            </div>
                            <button 
                                className="w-full bg-pink-600 text-white py-3 px-4 rounded-lg hover:bg-pink-700 transition-colors font-medium"
                                onClick={() => {
                                    if (transcription) {
                                        navigator.clipboard.writeText(transcription);
                                        alert('Transcription copied to clipboard!');
                                    }
                                }}
                            >
                                Copy Results
                            </button>
                        </div>
                    </div>

                    {/* Live Transcription */}
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-lg font-semibold mb-4 text-slate-200">Live Transcription</h3>
                        <div className="bg-slate-900/80 p-4 rounded-lg min-h-32 max-h-48 overflow-y-auto">
                            <p className="text-slate-300 text-sm mb-2">
                                {transcription || 'Voice transcription will appear here in real-time...'}
                            </p>
                            {interimTranscript && (
                                <p className="text-slate-500 italic text-sm">
                                    {interimTranscript}
                                </p>
                            )}
                        </div>
                    </div>
                </div>

                {/* Performance Metrics */}
                <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-xl font-semibold mb-6 text-slate-200">Performance Metrics</h3>
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span className="text-slate-300 text-sm">Recognition Accuracy</span>
                                    <span className="text-slate-400 text-sm">{accuracy}%</span>
                                </div>
                                <div className="w-full bg-slate-700 rounded-full h-2">
                                    <div 
                                        className="bg-green-500 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${accuracy}%` }}
                                    ></div>
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span className="text-slate-300 text-sm">Processing Speed</span>
                                    <span className="text-slate-400 text-sm">{processingSpeed}%</span>
                                </div>
                                <div className="w-full bg-slate-700 rounded-full h-2">
                                    <div 
                                        className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${processingSpeed}%` }}
                                    ></div>
                                </div>
                            </div>
                            <div>
                                <div className="flex justify-between mb-2">
                                    <span className="text-slate-300 text-sm">Audio Quality</span>
                                    <span className="text-slate-400 text-sm">{audioQuality}%</span>
                                </div>
                                <div className="w-full bg-slate-700 rounded-full h-2">
                                    <div 
                                        className="bg-purple-500 h-2 rounded-full transition-all duration-500"
                                        style={{ width: `${audioQuality}%` }}
                                    ></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                        <h3 className="text-xl font-semibold mb-6 text-slate-200">System Status</h3>
                        <div className="space-y-4">
                            {[
                                { name: 'Microphone', status: isReady, color: 'green' },
                                { name: 'Audio Processing', status: !!mediaRecorderRef.current, color: 'blue' },
                                { name: 'Speech Recognition', status: !!recognitionRef.current, color: 'purple' },
                                { name: 'Network Connection', status: navigator.onLine, color: 'green' }
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
        </div>
    );
};

// Main VoiceRecognition Component
const VoiceRecognition = () => {
    const [currentStep, setCurrentStep] = useState('preop');
    const [isReady, setIsReady] = useState(false);
    const [micPermission, setMicPermission] = useState(null);

    useEffect(() => {
        navigator.permissions.query({name: 'microphone'}).then((result) => {
            setMicPermission(result.state);
        });
    }, []);

    const handlePreOpComplete = () => {
        setIsReady(true);
        setCurrentStep('processing');
    };

    const handleBackToPreop = () => {
        setIsReady(false);
        setCurrentStep('preop');
    };

    const renderCurrentStep = () => {
        switch (currentStep) {
            case 'preop':
                return(
                    <PreOpVoice
                        onComplete={handlePreOpComplete}
                        micPermission={micPermission}
                        setMicPermission={setMicPermission}
                    />
                );
            case 'processing':
                return(
                    <VoiceProcessing
                        onBackToSetup={handleBackToPreop}
                        isReady={isReady}
                    />
                );
            default:
                return null;
        }
    };

    return (
        <div>
            {renderCurrentStep()}
        </div>
    );
}

export default VoiceRecognition;