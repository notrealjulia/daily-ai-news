import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
  // The data and the narration MP3s come from the read-only Python server (server.py).
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/audio": "http://localhost:8000",
    },
  },
})
