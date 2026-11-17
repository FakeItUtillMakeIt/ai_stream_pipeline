import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// base:'./' → 产物用相对路径，便于被 http_server 挂在任意路径(如 /editor/)
// dev 代理目标可用环境变量覆盖：
//   VITE_API_TARGET=http://127.0.0.1:18888 npm run dev
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  const apiTarget = env.VITE_API_TARGET || 'http://127.0.0.1:8080';
  const liveTarget = env.VITE_LIVE_TARGET || 'http://127.0.0.1:8888';
  return {
    plugins: [react()],
    base: './',
    build: { outDir: 'dist', emptyOutDir: true },
    server: {
      port: 5173,
      // dev 时把 API 代理到 http_server，避免跨域（生产同源无需代理）
      proxy: {
        '/api': apiTarget,
        '/health': apiTarget,
        '/live': liveTarget,
      },
    },
  };
});
