const { contextBridge, ipcRenderer } = require('electron');

// Expose secure API to overlay window renderer
contextBridge.exposeInMainWorld('overlayAPI', {
  // Window controls
  hideOverlay: () => ipcRenderer.invoke('hide-overlay'),
  toggleClickThrough: () => ipcRenderer.invoke('toggle-click-through'),
  setOpacity: (opacity) => ipcRenderer.invoke('set-overlay-opacity', opacity),
  
  // Positioning
  getBounds: () => ipcRenderer.invoke('get-overlay-bounds'),
  setBounds: (bounds) => ipcRenderer.invoke('set-overlay-bounds', bounds),
  
  // AI answer listener
  onAIAnswerReceived: (callback) => {
    ipcRenderer.on('ai-answer-received', (event, data) => {
      callback(data);
    });
  },
  
  // Remove listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  },
  
  // Utility functions for overlay
  minimize: () => {
    const currentBounds = ipcRenderer.invoke('get-overlay-bounds');
    currentBounds.then(bounds => {
      if (bounds) {
        ipcRenderer.invoke('set-overlay-bounds', {
          ...bounds,
          height: 50 // Minimize to title bar only
        });
      }
    });
  },
  
  restore: () => {
    const currentBounds = ipcRenderer.invoke('get-overlay-bounds');
    currentBounds.then(bounds => {
      if (bounds) {
        ipcRenderer.invoke('set-overlay-bounds', {
          ...bounds,
          height: 400 // Restore to full height
        });
      }
    });
  }
});

// Security: Remove access to Node.js APIs
delete window.require;
delete window.exports;
delete window.module;

console.log('Overlay window preload script loaded');