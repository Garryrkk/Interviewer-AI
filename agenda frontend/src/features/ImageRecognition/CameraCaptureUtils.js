/**
 * Utils for camera-based expression detection
 */

export async function captureCameraFrame(videoEl) {
  if (!videoEl) return null;

  const canvas = document.createElement("canvas");
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.8));
  return blob;
}

export async function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    if (!blob) return resolve(null);
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * detectExpression
 * Mock detection: randomly returns "happy", "confused", or "stressed"
 * Replace this with a real ML/vision API.
 */
export async function detectExpression(base64Image) {
  await new Promise((r) => setTimeout(r, 500)); // simulate processing

  const labels = [
    { label: "happy", confidence: Math.random().toFixed(2) },
    { label: "confused", confidence: Math.random().toFixed(2) },
    { label: "stressed", confidence: Math.random().toFixed(2) },
  ];

  // Pick max
  const expr = labels.reduce((a, b) => (parseFloat(a.confidence) > parseFloat(b.confidence) ? a : b));
  return expr;
}
