import React, {useState, useEffect} from "react";
import PreOpVoice from './PreOpVoice'
import VoiceProcessing from './VoiceProcessing'

const VoiceRecognition = () => {
    // Fixed typo: "currentSteo" -> "currentStep"
    const [currentStep, setCurrentStep] = useState('preop');
    const [isReady, setIsReady] = useState(false);
    const [micPermission, setMicPermission] = useState(null);

    // Fixed syntax: "useEffect (( => {" -> "useEffect(() => {"
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
        setIsReady(false); // Fixed: should set to false when going back to preop
        setCurrentStep('preop'); // Fixed typo: "proceesing" -> "preop"
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

    // Fixed: Added missing return statement
    return (
        <div>
            {renderCurrentStep()}
        </div>
    );
}

export default VoiceRecognition;