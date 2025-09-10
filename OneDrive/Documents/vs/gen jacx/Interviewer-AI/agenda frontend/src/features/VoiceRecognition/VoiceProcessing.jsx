import React, { useState, useEffect, useRef } from "react";
import {
    startRecording,
    stopRecording,
    processAudioData,
    sendToAI,
    formatResponse,
    audioToText,
    calculateVolume
} from './VoiceRecognitionUtils';

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
    const animationFrameRef = useRef(null); // Fixed typo: was "animationFrameref"
    const recordingIntervalRef = useRef(null);
    const analyserRef = useRef(null); // Fixed typo: was "analserRef"

    const handleStartRecording = async () => { // Fixed typo: was "handleStartTecording"
        try {
            setError('');
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            const audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            source.connect(analyser);
            analyserRef.current = analyser; // Fixed reference name

            const monitorVolume = () => {
                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                analyser.getByteFrequencyData(dataArray);
                const vol = calculateVolume(dataArray);
                setVolume(vol);
                animationFrameRef.current = requestAnimationFrame(monitorVolume); // Fixed reference name
            };
            monitorVolume();

            // Set up media recorder (fixed comment spacing)
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) { // Fixed spacing in if statement
                    audioChunksRef.current.push(event.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                await processRecording(audioBlob); // Fixed function name: was "ScriptProcessRecording"

                // Clean up resources (fixed comment)
                stream.getTracks().forEach(track => track.stop());
                if (animationFrameRef.current) { // Fixed reference name
                    cancelAnimationFrame(animationFrameRef.current);
                }
                setVolume(0);
            };
            
            mediaRecorder.start(100); // Collect data every 100ms
            setIsRecording(true);
            setRecordingTime(0);

            // Start recording timer (fixed comment spacing)
            recordingIntervalRef.current = setInterval(() => { // Fixed spacing in arrow function
                setRecordingTime(prev => prev + 1);
            }, 1000);

        } catch (err) {
            console.error('Error starting recording:', err);
            setError('Failed to start recording. Please check microphone permissions.');
        }
    };

    // Stop recording (fixed comment spacing)
    const handleStopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            
            if (recordingIntervalRef.current) {
                clearInterval(recordingIntervalRef.current);
            }
            
            if (animationFrameRef.current) { // Fixed reference name
                cancelAnimationFrame(animationFrameRef.current);
            }
        }
    };

    const processRecording = async (audioBlob) => {
        setIsProcessing(true);
        setTranscript('');
        setAiResponse('');

        try {
            // Convert audio to text using Web Speech API or external service
            const transcriptText = await audioToText(audioBlob);
            setTranscript(transcriptText);

            if (transcriptText.trim()) {
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
                setError('No speech detected. Please try again.');
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
            if (animationFrameRef.current) { // Fixed reference name
                cancelAnimationFrame(animationFrameRef.current);
            }
            if (recordingIntervalRef.current) {
                clearInterval(recordingIntervalRef.current);
            }
        };
    }, []);

    // Added missing return statement for the component JSX
    return (
        <div className="voice-processing">
            <div className="controls">
                <button 
                    onClick={isRecording ? handleStopRecording : handleStartRecording}
                    disabled={isProcessing}
                    className={`record-button ${isRecording ? 'recording' : ''}`}
                >
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                </button>
                
                {isRecording && (
                    <div className="recording-info">
                        <div className="recording-time">
                            Recording: {formatTime(recordingTime)}
                        </div>
                        <div className="volume-meter">
                            Volume: {Math.round(volume)}%
                        </div>
                    </div>
                )}
                
                {isProcessing && <div className="processing">Processing audio...</div>}
                
                {error && <div className="error">{error}</div>}
                
                {sessionHistory.length > 0 && (
                    <button onClick={clearHistory} className="clear-button">
                        Clear History
                    </button>
                )}
            </div>

            <div className="results">
                {transcript && (
                    <div className="transcript">
                        <h3>Transcript:</h3>
                        <p>{transcript}</p>
                    </div>
                )}
                
                {aiResponse && (
                    <div className="ai-response">
                        <h3>AI Response:</h3>
                        <p>{aiResponse}</p>
                    </div>
                )}
            </div>

            <div className="session-history">
                {sessionHistory.length > 0 && (
                    <>
                        <h3>Session History:</h3>
                        {sessionHistory.map((entry) => (
                            <div key={entry.id} className="history-entry">
                                <div className="entry-header">
                                    <span className="timestamp">{entry.timestamp}</span>
                                    <span className="duration">({formatTime(entry.duration)})</span>
                                </div>
                                <div className="entry-transcript">
                                    <strong>You:</strong> {entry.transcript}
                                </div>
                                <div className="entry-response">
                                    <strong>AI:</strong> {entry.response}
                                </div>
                            </div>
                        ))}
                    </>
                )}
            </div>
            
            {onBack && (
                <button onClick={onBack} className="back-button">
                    Back
                </button>
            )}
        </div>
    );
};

export default VoiceProcessing;