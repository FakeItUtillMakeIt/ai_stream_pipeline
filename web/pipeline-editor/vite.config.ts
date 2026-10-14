import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base:'./' → 产物用相对路径，便于被 http_server 挂在任意路径(如 /editor/)
export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5173,
    // dev 时把 API 代理到 http_server，避免跨域（生产同源无需代理）
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/health': 'http://127.0.0.1:8080',
    },
  },
});
