// src/services/voiceService.js
import React, { useState, useEffect, useRef } from "react";
import api from "./apiConfig";


/* ------------------------------------------------------------------
   🔹 Core API functions (as before)
-------------------------------------------------------------------*/

/**
 * Upload audio blob to the server for speech-to-text.
 * @param {Blob} audioBlob
 */
export async function transcribeAudio(audioBlob) {
  const formData = new FormData();
  formData.append("file", audioBlob, "recording.webm");

  const { data } = await api.post("/voice/transcribe", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data; // { transcript: "recognized text" }
}

/**
 * Start real-time voice recognition (browser fallback).
 * @param {(text:string)=>void} onData - callback for live transcript
 */
export function startLiveVoiceRecognition(onData) {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map((result) => result[0].transcript)
      .join("");
    onData(transcript);
  };

  recognition.start();
  return recognition;
}

/* ------------------------------------------------------------------
   🔹 UI Components (React)
   Each one is independent but can reuse the API functions above.
-------------------------------------------------------------------*/

/**
 * 🟢 PreOpVoice:
 * Pre-operation check — tests if microphone & browser speech API work.
 */
export function PreOpVoice() {
  const [status, setStatus] = useState("Checking microphone…");

  useEffect(() => {
    if (!navigator.mediaDevices || !window.SpeechRecognition) {
      setStatus("❌ Browser does not support Speech Recognition.");
      return;
    }
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(() => setStatus("✅ Mic access granted, ready to go!"))
      .catch(() => setStatus("❌ Mic access denied."));
  }, []);

  return (
    <div className="p-4 bg-white shadow-md rounded-md">
      <h2 className="text-lg font-semibold mb-2">Pre-Operation Voice Check</h2>
      <p>{status}</p>
    </div>
  );
}

/**
 * 🎤 VoiceRecognition:
 * Live, real-time voice-to-text using Web Speech API.
 */
export function VoiceRecognition() {
  const [transcript, setTranscript] = useState("");
  const recognitionRef = useRef(null);

  useEffect(() => {
    if (!window.SpeechRecognition && !window.webkitSpeechRecognition) {
      setTranscript("Browser does not support live recognition.");
      return;
    }
    recognitionRef.current = startLiveVoiceRecognition(setTranscript);
    return () => recognitionRef.current && recognitionRef.current.stop();
  }, []);

  return (
    <div className="p-4 bg-white shadow-md rounded-md">
      <h2 className="text-lg font-semibold mb-2">Live Voice Recognition</h2>
      <p className="whitespace-pre-wrap">{transcript}</p>
    </div>
  );
}

/**
 * ⚡ VoiceProcessing:
 * Records short clips, sends to backend for AI processing (speech-to-text).
 */
export function VoiceProcessing() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    chunksRef.current = [];
    mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data);
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(chunksRef.current, { type: "audio/webm" });
      const result = await transcribeAudio(audioBlob);
      setTranscript(result.transcript || "(No text recognized)");
    };
    mediaRecorder.start();
    mediaRecorderRef.current = mediaRecorder;
    setRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  return (
    <div className="p-4 bg-white shadow-md rounded-md">
      <h2 className="text-lg font-semibold mb-2">Voice Processing</h2>
      <button
        onClick={recording ? stopRecording : startRecording}
        className={`px-4 py-2 rounded-md text-white ${
          recording ? "bg-red-500" : "bg-green-500"
        }`}
      >
        {recording ? "Stop Recording" : "Start Recording"}
      </button>
      {transcript && (
        <p className="mt-3 text-gray-700">
          <strong>Transcript:</strong> {transcript}
        </p>
      )}
    </div>
  );
}
       