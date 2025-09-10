export const startRecording = async () => {
  try{
    const stream = await navigator.mediaDevices.getUserMedia({
        audio:{
         echoCancellation: true,
            noiseSuppression: true,
            sampleRate: 44100,
            channelCount: 1   
        }
    });
    return stream;
  
}catch (error){
    throw new Error('Failed to access microphone: ${error.message}');
}
};

export const stopRecording = (mediaRecorder) => {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
};

////audio proceesing ki utils
export const processAudioData = (audioBlob) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Failed to process audio data'));
    reader.readAsArrayBuffer(audioBlob);
  });
};

export const calculateVolume = (dataArray) => {
  if (!dataArray || dataArray.length === 0) return 0;
  
  const sum = dataArray.reduce((acc, value) => acc + value, 0);
  const average = sum / dataArray.length;
  return Math.min(100, Math.max(0, (average / 255) * 100));
};

export const normalizeAudio = (audioData) => {
  const maxValue = Math.max(...audioData.map(Math.abs));
  if (maxValue === 0) return audioData;
  
  const normalizedData = audioData.map(sample => sample / maxValue);
  return normalizedData;
};

export const audioToBase64 = (audioBlob) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = () => reject(new Error('Failed to convert audio to base64'));
    reader.readAsDataURL(audioBlob);
  });
};

export const audioToText = (audioBlob) => {
  return new Promise((resolve, reject) => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      setTimeout(() => {
        resolve("This is a simulated transcription since Web Speech API is not available.");
      }, 1000);
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    const audio = new Audio();
    audio.src = URL.createObjectURL(audioBlob);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      resolve(transcript);
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      resolve("Speech recognition failed. Please try again.");
    };

    recognition.onend = () => {
      URL.revokeObjectURL(audio.src);
    };

    try {
      recognition.start();
    } catch (error) {
      console.error('Recognition start error:', error);
      resolve("Unable to start speech recognition.");
    }
  });
};

export const sendToAI = async (text, conversationHistory = []) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      const responses = [
        `I heard you say: "${text}". This is an AI-generated response based on your voice input.`,
        `Processing your request: "${text}". Here's what I understand: This appears to be a voice command or question that I can help you with.`,
        `Voice input received: "${text}". Based on the context and conversation history, I can provide relevant information or assistance.`,
        `Thank you for your voice input: "${text}". I'm processing this information and providing a contextual response.`
      ];
      const randomResponse = responses[Math.floor(Math.random() * responses.length)];
      
      let contextualResponse = randomResponse;
      if (conversationHistory.length > 0) {
        contextualResponse += `\n\nBased on our previous conversation (${conversationHistory.length} exchanges), I'm providing a more contextual response.`;
      }
      
      resolve(contextualResponse);
    }, 1500); 
  });
};

export const formatResponse = (response) => {
  if (!response) return '';
  
  let formatted = response.trim();
  
  if (formatted && !formatted.match(/[.!?]$/)) {
    formatted += '.';
  }

  formatted = formatted.charAt(0).toUpperCase() + formatted.slice(1);
  
  return formatted;
};

export const checkAudioQuality = (audioData) => {
  if (!audioData || audioData.length === 0) {
    return { quality: 'poor', reason: 'No audio data' };
  }
  
  const maxAmplitude = Math.max(...audioData.map(Math.abs));
  const avgAmplitude = audioData.reduce((sum, val) => sum + Math.abs(val), 0) / audioData.length;
  
  if (maxAmplitude < 0.01) {
    return { quality: 'poor', reason: 'Audio too quiet' };
  }
  
  if (maxAmplitude > 0.95) {
    return { quality: 'poor', reason: 'Audio clipping detected' };
  }
  
  if (avgAmplitude < 0.001) {
    return { quality: 'poor', reason: 'Very low signal' };
  }
  
  return { quality: 'good', reason: 'Audio quality acceptable' };
};

export const detectNoise = (frequencyData) => {
  if (!frequencyData || frequencyData.length === 0) return false;
  
  const highFreqSum = frequencyData.slice(Math.floor(frequencyData.length * 0.7)).reduce((a, b) => a + b, 0);
  const totalSum = frequencyData.reduce((a, b) => a + b, 0);
  
  const highFreqRatio = highFreqSum / totalSum;
  return highFreqRatio > 0.3; 
};

export const formatDuration = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

export const saveRecording = (audioBlob, filename = 'recording.wav') => {
  const url = URL.createObjectURL(audioBlob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const checkMicrophonePermission = async () => {
  try {
    const result = await navigator.permissions.query({ name: 'microphone' });
    return result.state; 
  } catch (error) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach(track => track.stop());
      return 'granted';
    } catch (micError) {
      return 'denied';
    }
  }
};

export const createAudioContext = () => {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  return new AudioContext();
};

export const filterAudio = (audioData, filterType = 'lowpass') => {
  
  const filtered = [...audioData];
  const alpha = 0.8; 
  
  if (filterType === 'lowpass') {
    for (let i = 1; i < filtered.length; i++) {
      filtered[i] = alpha * filtered[i-1] + (1 - alpha) * filtered[i];
    }
  }
  
  return filtered;
};

export const convertAudioFormat = (audioBlob, targetFormat = 'wav') => {

  return Promise.resolve(audioBlob);
};

export const saveSession = (sessionData) => {
  try {
    const sessions = JSON.parse(sessionStorage.getItem('voiceSessions') || '[]');
    sessions.push({
      ...sessionData,
      id: Date.now(),
      timestamp: new Date().toISOString()
    });
    
    if (sessions.length > 50) {
      sessions.splice(0, sessions.length - 50);
    }
    
    sessionStorage.setItem('voiceSessions', JSON.stringify(sessions));
    return true;
  } catch (error) {
    console.error('Failed to save session:', error);
    return false;
  }
};

export const loadSessions = () => {
  try {
    return JSON.parse(sessionStorage.getItem('voiceSessions') || '[]');
  } catch (error) {
    console.error('Failed to load sessions:', error);
    return [];
  }
};

export const sanitizeText = (text) => {
  if (!text) return '';
  return text.replace(/[^\w\s.,!?-]/g, '').trim();
};

export const extractKeywords = (text) => {
  if (!text) return [];
  
  const words = text.toLowerCase().split(/\s+/);
  const stopWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'];
  
  return words
    .filter(word => word.length > 2 && !stopWords.includes(word))
    .reduce((acc, word) => {
      acc[word] = (acc[word] || 0) + 1;
      return acc;
    }, {});
};

export const handleVoiceError = (error) => {
  const errorMap = {
    'NotAllowedError': 'Microphone access denied. Please allow microphone permissions.',
    'NotFoundError': 'No microphone found. Please connect a microphone.',
    'NotSupportedError': 'Voice recognition not supported in this browser.',
    'NetworkError': 'Network error occurred. Please check your connection.',
    'AbortError': 'Recording was interrupted.',
    'InvalidStateError': 'Invalid recording state.',
    'SecurityError': 'Security error. Please use HTTPS.',
  };
  
  return errorMap[error.name] || `Voice processing error: ${error.message}`;
};

export const measurePerformance = (startTime, operation) => {
  const endTime = performance.now();
  const duration = endTime - startTime;
  console.log(`${operation} took ${duration.toFixed(2)}ms`);
  return duration;
};