// src/services/electronService.js
export function sendHiddenAnswerToOverlay(answer) {
  if (window?.electronAPI?.sendHiddenAnswer) {
    window.electronAPI.sendHiddenAnswer({ answer, time: Date.now() });
  }
}
