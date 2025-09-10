const { app, BrowserWindow, ipcMain, globalShortcut, screen } = require("electron");
const path = require("path");

let mainWindow;
let overlayWindow;

const isDev = !app.isPackaged; // safer than NODE_ENV check

// Get the development port from environment variable
function getDevPort() {
  const envPort = process.env.VITE_DEV_PORT;
  if (envPort) {
    const port = parseInt(envPort, 10);
    if (!isNaN(port) && port > 0) {
      console.log(`Using port from environment: ${port}`);
      return port;
    }
  }
  
  // Fallback to default port
  console.log('No valid port found in environment, using fallback: 5174');
  return 5174;
}

function createMainWindow() {
  const devPort = getDevPort();
  
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload-main.js"),
    },
    show: false,
  });

  if (isDev) {
    const devURL = `http://localhost:${devPort}`;
    console.log(`Loading dev URL: ${devURL}`);
    
    // Add error handling for URL loading
    mainWindow.loadURL(devURL).catch((error) => {
      console.error('Failed to load dev URL:', error);
      // Try alternative approach - wait and retry
      setTimeout(() => {
        mainWindow.loadURL(devURL).catch((retryError) => {
          console.error('Retry failed:', retryError);
        });
      }, 2000);
    });
  } else {
    mainWindow.loadFile(path.join(__dirname, "dist/index.html"));
  }

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
    console.log('Main window is ready and visible');
  });

  // Add additional error handling
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    console.error('Failed to load:', errorCode, errorDescription, validatedURL);
    if (isDev) {
      // Retry loading after a delay
      setTimeout(() => {
        const devPort = getDevPort();
        const devURL = `http://localhost:${devPort}`;
        console.log('Retrying to load:', devURL);
        mainWindow.loadURL(devURL);
      }, 3000);
    }
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
    if (overlayWindow) overlayWindow.close();
  });
}

function createOverlayWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width } = primaryDisplay.workAreaSize;

  overlayWindow = new BrowserWindow({
    width: 300,
    height: 400,
    x: width - 320,
    y: 50,
    frame: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: true,
    transparent: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "preload-overlay.js"),
    },
    show: false,
  });

  overlayWindow.loadFile(path.join(__dirname, "overlay.html"));
  overlayWindow.setIgnoreMouseEvents(false);

  overlayWindow.on("closed", () => {
    overlayWindow = null;
  });

  return overlayWindow;
}

// IPC handlers
ipcMain.handle("show-overlay", () => {
  if (!overlayWindow) createOverlayWindow();
  overlayWindow.show();
  overlayWindow.focus();
});

ipcMain.handle("hide-overlay", () => {
  if (overlayWindow) overlayWindow.hide();
});

ipcMain.handle("toggle-overlay", () => {
  if (!overlayWindow) {
    createOverlayWindow();
    overlayWindow.show();
  } else if (overlayWindow.isVisible()) {
    overlayWindow.hide();
  } else {
    overlayWindow.show();
    overlayWindow.focus();
  }
});

ipcMain.handle("send-ai-answer", (_, data) => {
  if (overlayWindow && overlayWindow.webContents) {
    overlayWindow.webContents.send("ai-answer-received", data);
  }
});

ipcMain.handle("toggle-click-through", () => {
  if (overlayWindow) {
    const isClickThrough = overlayWindow.isIgnoringMouseEvents();
    overlayWindow.setIgnoreMouseEvents(!isClickThrough);
    return !isClickThrough;
  }
  return false;
});

ipcMain.handle("set-overlay-opacity", (_, opacity) => {
  if (overlayWindow) {
    overlayWindow.setOpacity(Math.max(0.1, Math.min(1.0, opacity)));
  }
});

ipcMain.handle("get-overlay-bounds", () => {
  if (overlayWindow) return overlayWindow.getBounds();
  return null;
});

ipcMain.handle("set-overlay-bounds", (_, bounds) => {
  if (overlayWindow) overlayWindow.setBounds(bounds);
});

// App event handlers
app.whenReady().then(() => {
  console.log('Electron app is ready');
  console.log(`Development mode: ${isDev}`);
  console.log(`Environment port: ${process.env.VITE_DEV_PORT}`);
  
  createMainWindow();

  // Global shortcuts
  globalShortcut.register("CommandOrControl+Shift+H", () => {
    if (mainWindow) {
      mainWindow.webContents.send("toggle-overlay-requested");
    }
  });

  globalShortcut.register("CommandOrControl+Shift+T", () => {
    if (overlayWindow) {
      const isClickThrough = overlayWindow.isIgnoringMouseEvents();
      overlayWindow.setIgnoreMouseEvents(!isClickThrough);
    }
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on("window-all-closed", () => {
  globalShortcut.unregisterAll();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", () => {
  globalShortcut.unregisterAll();
});

// Prevent new window creation
app.on("web-contents-created", (event, contents) => {
  contents.on("new-window", (event) => {
    event.preventDefault();
  });
});

// Add some debugging info
console.log('Electron main process started');
console.log('Current working directory:', process.cwd());
console.log('__dirname:', __dirname);