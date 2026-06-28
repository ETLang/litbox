import { defineConfig } from 'vite';

export default defineConfig({
  // Tells Vite your live site will be served from a subfolder named after your repo
  base: '/litbox/',
  
  server: {
    port: 5173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});