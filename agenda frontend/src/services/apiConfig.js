
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "https://5456cb9f09f8.ngrok-free.app/api",
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Interceptor for adding auth tokens (if needed later)
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export default api;
