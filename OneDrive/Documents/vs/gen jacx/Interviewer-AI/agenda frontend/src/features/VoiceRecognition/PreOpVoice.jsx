import React, { useState, useEffect, useRef } from "react";
import { checkingMicrophonePermission, calibrateAudio, detectNoise } from './voiceUtils';

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
        };
    }, [mediaStream, audioContext]);

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

    const handleComplete = () => {
        if (allChecksComplete) {
            onComplete();
        }
    };

    return (
        <div>
            {/* Your JSX content here */}
            <div>Current Check: {currentCheck}</div>
            <div>Permission Status: {permissionStatus}</div>
            <div>Calibration Status: {calibrationStatus}</div>
            <div>Noise Level: {noiseLevel}%</div>
            
            {currentCheck === 'permission' && permissionStatus === 'pending' && (
                <button onClick={requestMicrophonePermission}>
                    Request Microphone Permission
                </button>
            )}
            
            {currentCheck === 'calibration' && calibrationStatus === 'pending' && (
                <button onClick={startCalibration}>
                    Start Calibration
                </button>
            )}
            
            {allChecksComplete && (
                <button onClick={handleComplete}>
                    Complete Setup
                </button>
            )}
        </div>
    );
};

export default PreOpVoice;