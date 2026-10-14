import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    watch: {
      // Don't watch the huge data / backend folders (hits the inotify limit)
      ignored: ["**/database/**", "**/backend/**", "**/notebooks/**", "**/scripts/**", "**/node_modules/**", "**/dist/**"],
    },
  },
})
