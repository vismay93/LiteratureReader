import { defineConfig } from "vite";

// Dev server stays on loopback; /api is proxied to the FastAPI backend.
export default defineConfig({
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
