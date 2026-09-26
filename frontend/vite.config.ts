import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
  },
  build: {
    // three.js 体积较大（约 1MB），懒加载后作为独立 chunk 按需加载；
    // 上调告警阈值避免对固有体积的误报。
    chunkSizeWarningLimit: 1200,
  },
})
