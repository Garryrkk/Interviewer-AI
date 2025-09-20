import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 61863, // Use port 0 for dynamic port assignment
    host: 'localhost',
    strictPort: false,
    open: false,

    // 👇 Add this proxy section
    proxy: {
      "/handsfree": {
        target: "http://127.0.0.1:8000",   // Your FastAPI backend
        changeOrigin: true,
      },
    },
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV || 'development')
  }
})