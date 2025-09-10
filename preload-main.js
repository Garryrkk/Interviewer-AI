const { contextBridge, ipcRenderer } = require('electron');

// Expose secure API to main window renderer
contextBridge.exposeInMainWorld('electronAPI', {
  // Overlay control
  showOverlay: () => ipcRenderer.invoke('show-overlay'),
  hideOverlay: () => ipcRenderer.invoke('hide-overlay'),
  toggleOverlay: () => ipcRenderer.invoke('toggle-overlay'),
  
  // Send AI answers to overlay
  sendAIAnswer: (data) => ipcRenderer.invoke('send-ai-answer', data),
  
  // Overlay configuration
  toggleClickThrough: () => ipcRenderer.invoke('toggle-click-through'),
  setOverlayOpacity: (opacity) => ipcRenderer.invoke('set-overlay-opacity', opacity),
  
  // Overlay positioning
  getOverlayBounds: () => ipcRenderer.invoke('get-overlay-bounds'),
  setOverlayBounds: (bounds) => ipcRenderer.invoke('set-overlay-bounds', bounds),
  
  // Event listeners
  onToggleOverlayRequested: (callback) => {
    ipcRenderer.on('toggle-overlay-requested', callback);
  },
  
  // Remove listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  }
});

// Security: Remove access to Node.js APIs
delete window.require;
delete window.exports;
delete window.module;

console.log('Main window preload script loaded');