import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/agents": "http://localhost:8000",
      "/discover": "http://localhost:8000",
      "/traces": "http://localhost:8000",
      "/trust": "http://localhost:8000",
    },
  },
});
