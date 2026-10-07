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
  // Locally, the data and the narration MP3s come from the read-only Python server
  // (server.py). The deployed build has static copies of both instead (see server.py).
  server: {
    proxy: {
      "/dashboard.json": "http://localhost:8000",
      "/audio": "http://localhost:8000",
    },
  },
})
