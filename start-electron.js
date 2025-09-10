const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

let viteProcess;
let electronProcess;
let assignedPort = null;

// Function to start Vite dev server
function startViteServer() {
  return new Promise((resolve, reject) => {
    console.log('🚀 Starting Vite development server...');
    
    const viteProcess = spawn('npm', ['run', 'dev'], {
      cwd: path.join(__dirname, 'agenda frontend'),
      stdio: 'pipe',
      shell: true
    });

    let portFound = false;

    viteProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log(output);

      // Look for the port in Vite's output
      const portMatch = output.match(/Local:\s+http:\/\/localhost:(\d+)\//);
      if (portMatch && !portFound) {
        assignedPort = parseInt(portMatch[1]);
        portFound = true;
        console.log(`✅ Vite server started on port: ${assignedPort}`);
        resolve(assignedPort);
      }
    });

    viteProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.error('Vite Error:', output);
      
      // Also check stderr for port information (sometimes Vite logs there)
      const portMatch = output.match(/Local:\s+http:\/\/localhost:(\d+)\//);
      if (portMatch && !portFound) {
        assignedPort = parseInt(portMatch[1]);
        portFound = true;
        console.log(`✅ Vite server started on port: ${assignedPort}`);
        resolve(assignedPort);
      }
    });

    viteProcess.on('error', (error) => {
      console.error('Failed to start Vite:', error);
      reject(error);
    });

    viteProcess.on('exit', (code) => {
      if (code !== 0 && !portFound) {
        reject(new Error(`Vite process exited with code ${code}`));
      }
    });

    // Store the process for cleanup
    global.viteProcess = viteProcess;

    // Timeout after 30 seconds
    setTimeout(() => {
      if (!portFound) {
        reject(new Error('Timeout: Could not detect Vite server port'));
      }
    }, 30000);
  });
}

// Function to start Electron with the assigned port
function startElectron(port) {
  return new Promise((resolve, reject) => {
    console.log(`🔧 Starting Electron with port: ${port}`);
    
    // Set environment variable for the port
    process.env.VITE_DEV_PORT = port.toString();
    
    const electronProcess = spawn('npx', ['electron', 'electron-main.js'], {
      stdio: 'inherit',
      shell: true,
      env: {
        ...process.env,
        VITE_DEV_PORT: port.toString(),
        NODE_ENV: 'development'
      }
    });

    electronProcess.on('error', (error) => {
      console.error('Failed to start Electron:', error);
      reject(error);
    });

    electronProcess.on('exit', (code) => {
      console.log(`Electron process exited with code: ${code}`);
      // Clean up Vite process when Electron exits
      if (global.viteProcess) {
        global.viteProcess.kill();
      }
      resolve(code);
    });

    // Store the process for cleanup
    global.electronProcess = electronProcess;
  });
}

// Main function to orchestrate the startup
async function main() {
  try {
    console.log('🎯 Starting Interviewer AI with dynamic port allocation...\n');
    
    // Start Vite and wait for port assignment
    const port = await startViteServer();
    
    // Wait a moment for Vite to fully initialize
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Start Electron with the assigned port
    await startElectron(port);
    
  } catch (error) {
    console.error('❌ Startup failed:', error.message);
    
    // Cleanup processes on error
    if (global.viteProcess) {
      global.viteProcess.kill();
    }
    if (global.electronProcess) {
      global.electronProcess.kill();
    }
    
    process.exit(1);
  }
}

// Handle process cleanup
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down...');
  if (global.viteProcess) {
    global.viteProcess.kill();
  }
  if (global.electronProcess) {
    global.electronProcess.kill();
  }
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Shutting down...');
  if (global.viteProcess) {
    global.viteProcess.kill();
  }
  if (global.electronProcess) {
    global.electronProcess.kill();
  }
  process.exit(0);
});

// Start the application
main();