import React, { useState, useEffect, useRef } from "react";
import { checkingMicrophonePermission, calibrateAudio, detectNoise } from './voiceUtils';
import { 
  Mic, 
  MicOff, 
  Activity, 
  CheckCircle, 
  AlertCircle, 
  Volume2,
  Settings,  
  Play,
  Square
} from 'lucide-react';
import { PreOpVoice } from "../../services/voiceService";

const PreOpVoice = ({ onComplete, micPermission, setMicPermission }) => {
    const [currentCheck, setCurrentCheck] = useState('permission');
    const [permissionStatus, setPermissionStatus] = useState('pending');
    const [calibrationStatus, setCalibrationStatus] = useState('pending');
    const [noiseLevel, setNoiseLevel] = useState(0);
    const [isCalibrating, setIsCalibrating] = useState(false);
    const [audioContext, setAudioContext] = useState(null);
    const [mediaStream, setMediaStream] = useState(null);
    const [calibrationData, setCalibrationData] = useState(null);
    const [allChecksComplete, setAllChecksComplete] = useState(false);

    // Voice transcription states
    const [isRecording, setIsRecording] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [audioChunks, setAudioChunks] = useState([]);

    const animationRef = useRef();
    const analyserRef = useRef();

    useEffect(() => {
        if (micPermission === 'granted') {
            setPermissionStatus('granted');
            setCurrentCheck('calibration');
        }
    }, [micPermission]);

    useEffect(() => {
        return () => {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
            }
            if (audioContext) {
                audioContext.close();
            }
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        };
    }, [mediaStream, audioContext, mediaRecorder]);

    const requestMicrophonePermission = async () => {
        setPermissionStatus('requesting');

        try {
            const permission = await checkingMicrophonePermission();
            setPermissionStatus(permission);
            setMicPermission(permission);

            if (permission === 'granted') {
                setTimeout(() => {
                    setCurrentCheck('calibration');
                }, 1000);
            }
        } catch (error) {
            console.error('Error requesting microphone permission:', error);
            setPermissionStatus('denied');
        }
    };

    const startCalibration = async () => {
        setIsCalibrating(true);
        setCalibrationStatus('calibrating');

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            setMediaStream(stream);
            const context = new (window.AudioContext || window.webkitAudioContext)();
            setAudioContext(context);

            const source = context.createMediaStreamSource(stream);
            const analyser = context.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;

            source.connect(analyser);
            analyserRef.current = analyser;
            
            // Setup MediaRecorder for transcription
            const recorder = new MediaRecorder(stream);
            setMediaRecorder(recorder);

            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    setAudioChunks(prev => [...prev, event.data]);
                }
            };

            recorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                sendAudioToBackend(audioBlob);
            };
            
            // Start noise detection
            monitorNoise();

            // Calibrate for 3 secs
            setTimeout(() => {
                finishCalibration();
            }, 3000);

        } catch (error) {
            console.error('Error during calibration:', error);
            setCalibrationStatus('failed');
            setIsCalibrating(false);
        }
    };

    const monitorNoise = () => {
        if (!analyserRef.current) return;

        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);

        const checkNoise = () => {
            analyserRef.current.getByteFrequencyData(dataArray);

            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            setNoiseLevel(Math.round((average / 255) * 100));

            if (isCalibrating) {
                animationRef.current = requestAnimationFrame(checkNoise);
            }
        };

        checkNoise();
    };

    const finishCalibration = () => {
        setIsCalibrating(false);
        setCalibrationStatus('completed');

        const data = calibrateAudio(noiseLevel);
        setCalibrationData(data);

        setTimeout(() => {
            setCurrentCheck('complete');
            setAllChecksComplete(true);
        }, 1000);
    };

    // Voice transcription functions
    const startRecording = () => {
        if (mediaRecorder && mediaRecorder.state === 'inactive') {
            setAudioChunks([]);
            setIsRecording(true);
            mediaRecorder.start();
        }
    };

    const stopRecording = () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            setIsRecording(false);
            mediaRecorder.stop();
        }
    };

    const sendAudioToBackend = async (audioBlob) => {
        try {
            const formData = new FormData();
            formData.append('audio', audioBlob);
            
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });
            
            const { transcript: newTranscript } = await response.json();
            setTranscript(newTranscript);
        } catch (error) {
            console.error('Error transcribing audio:', error);
            setTranscript('Error: Could not transcribe audio');
        }
    };

    const handleComplete = () => {
        if (allChecksComplete) {
            onComplete();
        }
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'granted':
            case 'completed':
                return 'text-green-500';
            case 'requesting':
            case 'calibrating':
                return 'text-yellow-500';
            case 'denied':
            case 'failed':
                return 'text-red-500';
            default:
                return 'text-slate-400';
        }
    };

    const getStatusIcon = (status, isActive = false) => {
        switch (status) {
            case 'granted':
            case 'completed':
                return <CheckCircle size={24} className="text-green-500" />;
            case 'requesting':
            case 'calibrating':
                return <Activity size={24} className={`text-yellow-500 ${isActive ? 'animate-spin' : ''}`} />;
            case 'denied':
            case 'failed':
                return <AlertCircle size={24} className="text-red-500" />;
            default:
                return <div className="w-6 h-6 border-2 border-slate-500 rounded-full"></div>;
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 font-sans">
            {/* Header */}
            <div className="bg-slate-800/50 backdrop-blur border-b border-slate-700 px-6 py-4">
                <div className="flex items-center space-x-4">
                    <div className="w-8 h-8 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg flex items-center justify-center">
                        <Mic size={18} className="text-white" />
                    </div>
                    <div>
                        <h1 className="text-xl font-bold">Voice Setup Assistant</h1>
                        <p className="text-sm text-slate-400">Preparing your microphone for optimal performance</p>
                    </div>
                </div>
            </div>

            <div className="p-8">
                {/* Progress Steps */}
                <div className="mb-12">
                    <div className="flex items-center justify-center space-x-8">
                        <div className="flex flex-col items-center space-y-2">
                            <div className={`w-16 h-16 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                                currentCheck === 'permission' 
                                    ? 'border-pink-500 bg-pink-600/20' 
                                    : permissionStatus === 'granted' 
                                        ? 'border-green-500 bg-green-600/20' 
                                        : 'border-slate-600 bg-slate-800/50'
                            }`}>
                                {permissionStatus === 'granted' 
                                    ? <CheckCircle size={24} className="text-green-500" />
                                    : <Mic size={24} className={currentCheck === 'permission' ? 'text-pink-500' : 'text-slate-400'} />
                                }
                            </div>
                            <span className={`text-sm font-medium ${
                                currentCheck === 'permission' ? 'text-pink-400' : 'text-slate-400'
                            }`}>
                                Permission
                            </span>
                        </div>

                        <div className={`h-0.5 w-24 transition-colors duration-300 ${
                            permissionStatus === 'granted' ? 'bg-green-500' : 'bg-slate-600'
                        }`}></div>

                        <div className="flex flex-col items-center space-y-2">
                            <div className={`w-16 h-16 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                                currentCheck === 'calibration' 
                                    ? 'border-pink-500 bg-pink-600/20' 
                                    : calibrationStatus === 'completed' 
                                        ? 'border-green-500 bg-green-600/20' 
                                        : 'border-slate-600 bg-slate-800/50'
                            }`}>
                                {calibrationStatus === 'completed' 
                                    ? <CheckCircle size={24} className="text-green-500" />
                                    : calibrationStatus === 'calibrating'
                                        ? <Activity size={24} className="text-pink-500 animate-pulse" />
                                        : <Settings size={24} className={currentCheck === 'calibration' ? 'text-pink-500' : 'text-slate-400'} />
                                }
                            </div>
                            <span className={`text-sm font-medium ${
                                currentCheck === 'calibration' ? 'text-pink-400' : 'text-slate-400'
                            }`}>
                                Calibration
                            </span>
                        </div>

                        <div className={`h-0.5 w-24 transition-colors duration-300 ${
                            allChecksComplete ? 'bg-green-500' : 'bg-slate-600'
                        }`}></div>

                        <div className="flex flex-col items-center space-y-2">
                            <div className={`w-16 h-16 rounded-full flex items-center justify-center border-2 transition-all duration-300 ${
                                allChecksComplete 
                                    ? 'border-green-500 bg-green-600/20' 
                                    : 'border-slate-600 bg-slate-800/50'
                            }`}>
                                {allChecksComplete 
                                    ? <CheckCircle size={24} className="text-green-500" />
                                    : <Play size={24} className="text-slate-400" />
                                }
                            </div>
                            <span className={`text-sm font-medium ${
                                allChecksComplete ? 'text-green-400' : 'text-slate-400'
                            }`}>
                                Ready
                            </span>
                        </div>
                    </div>
                </div>

                {/* Main Content Area */}
                <div className="max-w-4xl mx-auto">
                    {currentCheck === 'permission' && (
                        <div className="bg-slate-800/50 backdrop-blur p-8 rounded-xl border border-slate-700 text-center">
                            <div className="mb-6">
                                <div className={`w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-4 transition-all duration-300 ${
                                    permissionStatus === 'requesting' 
                                        ? 'bg-yellow-600/20 animate-pulse' 
                                        : permissionStatus === 'granted'
                                            ? 'bg-green-600/20'
                                            : 'bg-slate-900'
                                }`}>
                                    {permissionStatus === 'denied' 
                                        ? <MicOff size={48} className="text-red-500" />
                                        : <Mic size={48} className={
                                            permissionStatus === 'granted' 
                                                ? 'text-green-500' 
                                                : permissionStatus === 'requesting'
                                                    ? 'text-yellow-500'
                                                    : 'text-slate-400'
                                        } />
                                    }
                                </div>
                                <h2 className="text-2xl font-bold mb-2 text-slate-100">Microphone Permission</h2>
                                <p className="text-slate-400 mb-6">
                                    {permissionStatus === 'pending' && 'We need access to your microphone to continue'}
                                    {permissionStatus === 'requesting' && 'Requesting microphone access...'}
                                    {permissionStatus === 'granted' && 'Microphone access granted! Moving to calibration...'}
                                    {permissionStatus === 'denied' && 'Microphone access denied. Please enable it in your browser settings.'}
                                </p>
                            </div>

                            {permissionStatus === 'pending' && (
                                <button 
                                    onClick={requestMicrophonePermission}
                                    className="bg-gradient-to-r from-pink-600 to-pink-700 text-white py-4 px-8 rounded-lg hover:from-pink-700 hover:to-pink-800 transition-all font-medium text-lg shadow-lg"
                                >
                                    Grant Microphone Permission
                                </button>
                            )}

                            {permissionStatus === 'denied' && (
                                <div className="bg-red-900/30 border border-red-700/50 p-4 rounded-lg">
                                    <p className="text-red-300 text-sm">
                                        Please refresh the page and allow microphone access when prompted, or check your browser settings.
                                    </p>
                                </div>
                            )}
                        </div>
                    )}

                    {currentCheck === 'calibration' && (
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            {/* Calibration Control */}
                            <div className="lg:col-span-2 bg-slate-800/50 backdrop-blur p-8 rounded-xl border border-slate-700">
                                <h2 className="text-2xl font-bold mb-6 text-slate-100">Audio Calibration</h2>
                                
                                <div className="text-center mb-8">
                                    <div className={`w-32 h-32 rounded-full flex items-center justify-center mx-auto mb-6 transition-all duration-300 ${
                                        isCalibrating 
                                            ? 'bg-pink-600/20 animate-pulse' 
                                            : calibrationStatus === 'completed'
                                                ? 'bg-green-600/20'
                                                : 'bg-slate-900'
                                    }`}>
                                        {calibrationStatus === 'completed' 
                                            ? <CheckCircle size={64} className="text-green-500" />
                                            : isCalibrating
                                                ? <Activity size={64} className="text-pink-500 animate-spin" />
                                                : <Volume2 size={64} className="text-slate-400" />
                                        }
                                    </div>

                                    <div className="space-y-4">
                                        <div>
                                            <div className="flex justify-between mb-2">
                                                <span className="text-slate-300">Noise Level</span>
                                                <span className="text-slate-400">{noiseLevel}%</span>
                                            </div>
                                            <div className="w-full bg-slate-700 rounded-full h-3">
                                                <div 
                                                    className={`h-3 rounded-full transition-all duration-300 ${
                                                        noiseLevel > 70 ? 'bg-red-500' : noiseLevel > 40 ? 'bg-yellow-500' : 'bg-green-500'
                                                    }`}
                                                    style={{ width: `${Math.min(noiseLevel, 100)}%` }}
                                                ></div>
                                            </div>
                                        </div>

                                        {calibrationStatus === 'pending' && (
                                            <button 
                                                onClick={startCalibration}
                                                className="bg-gradient-to-r from-pink-600 to-pink-700 text-white py-4 px-8 rounded-lg hover:from-pink-700 hover:to-pink-800 transition-all font-medium text-lg shadow-lg"
                                            >
                                                Start Calibration
                                            </button>
                                        )}

                                        {isCalibrating && (
                                            <div className="bg-yellow-900/30 border border-yellow-700/50 p-4 rounded-lg">
                                                <p className="text-yellow-300 text-sm">
                                                    Calibrating... Please speak normally for a few seconds.
                                                </p>
                                            </div>
                                        )}

                                        {calibrationStatus === 'completed' && (
                                            <div className="bg-green-900/30 border border-green-700/50 p-4 rounded-lg">
                                                <p className="text-green-300 text-sm">
                                                    Calibration completed successfully!
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Status Panel */}
                            <div className="bg-slate-800/50 backdrop-blur p-6 rounded-xl border border-slate-700">
                                <h3 className="text-lg font-semibold mb-4 text-slate-200">Setup Status</h3>
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/50">
                                        <span className="text-slate-300 font-medium">Permission</span>
                                        <div className="flex items-center space-x-2">
                                            {getStatusIcon(permissionStatus)}
                                            <span className={`text-sm font-medium ${getStatusColor(permissionStatus)}`}>
                                                {permissionStatus === 'granted' ? 'Granted' : 
                                                 permissionStatus === 'requesting' ? 'Requesting...' : 
                                                 permissionStatus === 'denied' ? 'Denied' : 'Pending'}
                                            </span>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/50">
                                        <span className="text-slate-300 font-medium">Calibration</span>
                                        <div className="flex items-center space-x-2">
                                            {getStatusIcon(calibrationStatus, isCalibrating)}
                                            <span className={`text-sm font-medium ${getStatusColor(calibrationStatus)}`}>
                                                {calibrationStatus === 'completed' ? 'Complete' : 
                                                 calibrationStatus === 'calibrating' ? 'In Progress' : 
                                                 calibrationStatus === 'failed' ? 'Failed' : 'Pending'}
                                            </span>
                                        </div>
                                    </div>

                                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/50">
                                        <span className="text-slate-300 font-medium">Noise Level</span>
                                        <div className="flex items-center space-x-2">
                                            <div className={`w-2 h-2 rounded-full ${
                                                noiseLevel > 70 ? 'bg-red-500' : 
                                                noiseLevel > 40 ? 'bg-yellow-500' : 
                                                'bg-green-500'
                                            }`}></div>
                                            <span className="text-sm font-medium text-slate-400">{noiseLevel}%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {currentCheck === 'complete' && allChecksComplete && (
                        <div className="bg-slate-800/50 backdrop-blur p-12 rounded-xl border border-slate-700 text-center">
                            <div className="mb-6">
                                <div className="w-24 h-24 bg-green-600/20 rounded-full flex items-center justify-center mx-auto mb-4">
                                    <CheckCircle size={48} className="text-green-500" />
                                </div>
                                <h2 className="text-3xl font-bold mb-4 text-slate-100">Setup Complete!</h2>
                                <p className="text-slate-400 mb-8 text-lg">
                                    Your voice recognition system is now ready for optimal performance.
                                </p>
                                
                                {calibrationData && (
                                    <div className="bg-slate-900/50 p-6 rounded-lg mb-6 max-w-md mx-auto">
                                        <h4 className="font-semibold text-slate-200 mb-3">Calibration Results</h4>
                                        <div className="space-y-2 text-sm">
                                            <div className="flex justify-between">
                                                <span className="text-slate-400">Background Noise:</span>
                                                <span className="text-slate-300">{noiseLevel}%</span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-slate-400">Quality:</span>
                                                <span className="text-green-400">Excellent</span>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Voice Transcription Section */}
                                <div className="bg-slate-900/50 p-6 rounded-lg mb-6 max-w-2xl mx-auto">
                                    <h4 className="font-semibold text-slate-200 mb-4">Test Voice Transcription</h4>
                                    
                                    <div className="flex justify-center space-x-4 mb-4">
                                        <button 
                                            onClick={startRecording}
                                            disabled={isRecording}
                                            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
                                                isRecording 
                                                    ? 'bg-gray-600 cursor-not-allowed' 
                                                    : 'bg-red-600 hover:bg-red-700'
                                            } text-white`}
                                        >
                                            <Mic size={16} />
                                            <span>Start Recording</span>
                                        </button>
                                        
                                        <button 
                                            onClick={stopRecording}
                                            disabled={!isRecording}
                                            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
                                                !isRecording 
                                                    ? 'bg-gray-600 cursor-not-allowed' 
                                                    : 'bg-red-600 hover:bg-red-700'
                                            } text-white`}
                                        >
                                            <Square size={16} />
                                            <span>Stop Recording</span>
                                        </button>
                                    </div>

                                    {isRecording && (
                                        <div className="bg-red-900/30 border border-red-700/50 p-3 rounded-lg mb-4">
                                            <div className="flex items-center justify-center space-x-2">
                                                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                                                <span className="text-red-300 text-sm">Recording in progress...</span>
                                            </div>
                                        </div>
                                    )}

                                    {transcript && (
                                        <div className="bg-slate-800 p-4 rounded-lg border border-slate-600">
                                            <h5 className="text-slate-300 text-sm mb-2">Transcript:</h5>
                                            <p className="text-slate-100">{transcript}</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            <button 
                                onClick={handleComplete}
                                className="bg-gradient-to-r from-green-600 to-green-700 text-white py-4 px-8 rounded-lg hover:from-green-700 hover:to-green-800 transition-all font-medium text-lg shadow-lg"
                            >
                                Continue to Application
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PreOpVoice;