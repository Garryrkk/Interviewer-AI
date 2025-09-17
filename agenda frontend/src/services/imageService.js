// src/services/imageService.js
import api from "./apiConfig";

/**
 * ===========================
 * 1️⃣  API: Send an image to backend for analysis
 * ===========================
 * @param {File|Blob} imageFile
 * @returns {Promise<Object>} { labels: [...], text: "detected text", ... }
 */
export async function analyzeImage(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);

  const { data } = await api.post("/image/analyze", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/**
 * ===========================
 * 2️⃣  CAMERA CAPTURE (Webcam)
 * ===========================
 * Opens the user's webcam, returns a single captured frame as a Blob.
 * @returns {Promise<Blob>}
 */
export async function captureFromCamera() {
  return new Promise(async (resolve, reject) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.createElement("video");
      video.srcObject = stream;
      video.play();

      // Wait for the video to be ready
      video.onloadedmetadata = () => {
        // Create a canvas to draw the current frame
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);

        // Stop the camera stream after capture
        stream.getTracks().forEach((track) => track.stop());

        canvas.toBlob((blob) => {
          resolve(blob);
        }, "image/png");
      };
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * ===========================
 * 3️⃣  SCREEN CAPTURE (Entire Screen or Window)
 * ===========================
 * Opens a browser prompt to choose a screen/window to capture.
 * Returns a Blob of the screenshot.
 * @returns {Promise<Blob>}
 */
export async function captureScreen() {
  return new Promise(async (resolve, reject) => {
    try {
      // Prompt user to share screen
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { cursor: "always" }
      });
      const track = stream.getVideoTracks()[0];
      const imageCapture = new ImageCapture(track);

      // Grab a still frame
      const bitmap = await imageCapture.grabFrame();

      // Draw to canvas
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(bitmap, 0, 0);

      // Stop screen sharing after capture
      track.stop();

      canvas.toBlob((blob) => {
        resolve(blob);
      }, "image/png");
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * ===========================
 * 4️⃣  Convenience: Capture & Send in One Step
 * ===========================
 * Example: take a screenshot and directly send it to the AI backend.
 * @returns {Promise<Object>} analysis result
 */
export async function captureScreenAndAnalyze() {
  const screenBlob = await captureScreen();
  return analyzeImage(screenBlob);
}

export async function captureCameraAndAnalyze() {
  const cameraBlob = await captureFromCamera();
  return analyzeImage(cameraBlob);
}
