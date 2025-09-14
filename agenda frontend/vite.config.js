import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 61863, // Use port 0 for dynamic port assignment
    host: 'localhost',
    strictPort: false, // Allow Vite to find an alternative port if needed
    open: false, // Don't automatically open browser since Electron will handle this
  },
  // Ensure proper handling in development
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development')
  }
})