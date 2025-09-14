/**
 * Hello Bimari Read From Here Yey
 * Utilities for screen capture & image recognition
 * - captureScreenFrame: take a frame from a video element (screen share)
 * - blobToBase64: helper to convert to base64
 * - recognizeImage: placeholder for calling a real image recognition API
 * - generateInsightFromRecognition: normalize recognition result to an "insight" object
 * - formatRecognitionResult: turn insight into a short prompt for QuickRespond
 *
 * NOTE: replace recognizeImage() with your real backend call when ready.
 */

export async function captureScreenFrame(videoEl) {
  try {
    if (!videoEl || !videoEl.srcObject) return null;

    const track = videoEl.srcObject.getVideoTracks()[0];
    const settings = track.getSettings ? track.getSettings() : {};
    const width = settings.width || videoEl.videoWidth || 1280;
    const height = settings.height || videoEl.videoHeight || 720;
     
    // Create an offscreen canvas to draw current frame
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    // Draw the current frame
    ctx.drawImage(videoEl, 0, 0, width, height);

    // Convert to Blob
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.8));
    return blob;
  } catch (err) {
    console.error("captureScreenFrame error:", err);
    return null;
  }
}

export async function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    if (!blob) return resolve(null);
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]); // return only base64 data portion
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * recognizeImage
 * - Placeholder local recognition that returns labels, bounding boxes, and OCR text
 * - Replace this with a fetch() call to your backend image recognition endpoint
 */
export async function recognizeImage(base64Image) {
  // If no image provided, return empty
  if (!base64Image) return { labels: [], ocr: "", raw: null };

  // MOCKED behaviour: simple heuristics based on image "hash" -> random labels
  // In real use: send base64Image to backend (DeepVision, Cloud Vision, your LLM with vision)
  await new Promise((r) => setTimeout(r, 600)); // simulate network/processing delay

  // Mock sample response
  const sampleLabels = [
    { name: "presentation slide", confidence: 0.88 },
    { name: "chart", confidence: 0.76 },
    { name: "logo", confidence: 0.45 },
    { name: "text block", confidence: 0.82 },
  ];

  // Mock OCR text (in real life this would be from OCR)
  const sampleOCR = "Quarterly revenue increased by 18% compared to last quarter.";

  return {
    labels: sampleLabels,
    ocr: sampleOCR,
    raw: null,
  };
}

/**
 * generateInsightFromRecognition
 * Convert a recognition response into a normalized insight object
 * similar in shape to your partner's KeyInsights objects.
 */
export function generateInsightFromRecognition(recognition) {
  const id = `rec-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`;
  const labels = recognition?.labels || [];
  const topLabel = labels.length ? labels[0] : null;
  const ocr = recognition?.ocr || "";

  const title = topLabel ? `Detected: ${topLabel.name}` : (ocr ? "Detected text on screen" : "Unknown content");
  const summaryParts = [];
  if (topLabel) summaryParts.push(`${topLabel.name} (${(topLabel.confidence || 0).toFixed(2)})`);
  if (ocr) summaryParts.push(`Text: "${ocr.slice(0, 120)}${ocr.length > 120 ? '…' : ''}"`);

  const summary = summaryParts.join(" · ") || "No clear detection";
  const category = topLabel ? (topLabel.confidence > 0.8 ? "important" : "trend") : (ocr ? "recent" : "general");
  const confidence = topLabel ? topLabel.confidence : (ocr ? 0.6 : 0.2);

  return {
    id,
    title,
    summary,
    details: { labels, ocr, raw: recognition?.raw },
    category,
    confidence,
    timestamp: new Date().toISOString(),
  };
}

/**
 * formatRecognitionResult
 * Create a compact prompt text that can be sent to QuickRespond or your chat AI
 */
export function formatRecognitionResult(insight) {
  const lines = [];
  lines.push(`I captured a screenshot. Top detection: ${insight.title}.`);
  if (insight.details?.labels?.length) {
    lines.push(
      "Labels: " +
        insight.details.labels
          .map((l) => `${l.name} (${(l.confidence || 0).toFixed(2)})`)
          .slice(0, 5)
          .join(", ")
    );
  }
  if (insight.details?.ocr) {
    lines.push(`Text on screen: "${insight.details.ocr}"`);
  }
  lines.push(`Confidence: ${(insight.confidence || 0).toFixed(2)}. Please provide a quick understandable answer for a candidate asked 'what is this?'`);
  return lines.join("\n");
}

/**
 * Example helper you can reuse if you want to compute a priority score for recognition items
 */
export function recognitionPriority(insight) {
  let score = (insight.confidence || 0) * 3; // base
  if (insight.category === "important") score += 1;
  if (insight.details?.ocr) score += 0.5;
  return Math.min(Math.round(score), 5);
}
