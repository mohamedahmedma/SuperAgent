import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]',
      },
    },
  },
  server: {
    port: 3000,
    proxy: {
      // Identity service. Authentication left the chat backend, so `/auth` no longer
      // exists on :8000 — login, registration and refresh are all under /v1/auth here.
      '/v1': 'http://localhost:8200',
      // Proxy backend API endpoints for local dev integration
      '/chat': 'http://localhost:8000',
      '/sessions': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
      '/media': 'http://localhost:8000',
    },
  },
});
