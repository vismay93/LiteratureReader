import { defineConfig } from "vite";

// Dev server stays on loopback; /api is proxied to the FastAPI backend.
export default defineConfig({
  // Relative asset URLs so the UI also works under a proxy prefix (e.g. Jupyter /proxy/8000/).
  base: "./",
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } },
  },
});
