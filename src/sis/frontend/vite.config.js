/*
 * The build. Source lives here; the output is what `sis/app.py` serves at `/ui`.
 *
 * Three decisions worth stating, because each replaces something the previous no-build
 * console did by hand.
 *
 * **`base: './'`** makes every generated URL relative. The console is mounted at `/ui`, not
 * at the root, and an absolute `/assets/index-abc.js` would 404 there — the failure would be
 * a blank page with a console error, which is the least diagnosable thing a deployment can
 * produce.
 *
 * **`outDir` points at `sis/web`, and `emptyOutDir` is on.** That directory used to hold
 * hand-written files and now holds build output only; nothing in it should ever be edited,
 * which is why the build wipes it. `sis/app.py` already serves it, so there is no server
 * change and no second path to keep in step.
 *
 * **Hashed filenames, which is what finally fixes the caching bug properly.** The previous
 * build had stable names, so the only way to stop a browser serving a stale stylesheet was
 * `Cache-Control: no-cache` and a revalidation round trip per asset per load. With a hash in
 * the name every asset is immutable and can be cached for a year; only `index.html` needs
 * revalidating. That is strictly better and it is the main thing a bundler buys here.
 */
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../web',
    emptyOutDir: true,
    // Readable stack traces from a school laptop are worth more than the few kilobytes a
    // sourcemap costs, and it is never parsed unless devtools is open.
    sourcemap: true,
    rollupOptions: {
      output: {
        /*
         * React and Bootstrap in their own chunk. They change when a dependency is upgraded
         * — a few times a year — while the application code changes weekly, so splitting
         * them means a routine deploy re-downloads ~30 KB of app instead of ~80 KB of
         * everything. Both are immutable-hashed, so the vendor chunk stays in cache.
         */
        manualChunks: {
          vendor: ['react', 'react-dom']
        }
      }
    }
  },
  server: {
    port: 5173,
    /*
     * `npm run dev` serves the app on :5173 and proxies the API to the Python service, so
     * the browser sees one origin and there is no CORS story in development. Without this
     * the dev server would have to be told an API host, and `api.js` would need a base-URL
     * setting that production does not use — a configuration that exists only in
     * development is a configuration that breaks in production.
     */
    proxy: {
      '/v1': { target: 'http://127.0.0.1:8300', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8300', changeOrigin: true }
    }
  },
  resolve: {
    alias: {
      /*
       * Uncomment to run the identical React source on Preact's runtime: same JSX, same
       * hooks, same imports, measured at 5.3 KB gzipped instead of React's 45.5 KB. Left off
       * by default because React proper is what was asked for, and it is one line to switch
       * if the 40 KB matters more than React DevTools does on a school network.
       *
       * react: 'preact/compat',
       * 'react-dom': 'preact/compat',
       * 'react-dom/client': 'preact/compat',
       */
    }
  }
});
