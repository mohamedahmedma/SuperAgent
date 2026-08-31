import { defineConfig, loadEnv } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path';

/**
 * The repo root, and the single place this project's configuration lives.
 *
 * There used to be a `frontend/.env` as well, which made the deployment's addresses two
 * files that had to agree. They did not have to agree loudly: nothing reads both, so a UI
 * pointed at the wrong identity service builds, deploys, and fails only when a parent
 * tries to sign in.
 */
const REPO_ROOT = resolve(__dirname, '..');

/** Where the dev server proxies to when the root `.env` says nothing. The ports each
 *  service documents for itself — backend 8000, identity 8200. */
const DEV_BACKEND_DEFAULT = 'http://localhost:8000';
const DEV_IDENTITY_DEFAULT = 'http://localhost:8200';

/** The backend paths the dev server forwards. One list, because they all go to one place
 *  and writing the address four times is how three of them stay right and one drifts. */
const BACKEND_PATHS = ['/chat', '/sessions', '/documents', '/media'];

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  /* Every name, not just VITE_ ones. This runs in node at dev/build time and configures
     the proxy, which is server-side and never reaches the browser — `envPrefix` still
     governs what gets inlined into the bundle, and it still defaults to VITE_. Reading a
     file full of secrets here exposes none of them; `src/config.spec.ts` asserts it. */
  const env = loadEnv(mode, REPO_ROOT, '');

  const backend = env.DEV_PROXY_BACKEND || DEV_BACKEND_DEFAULT;
  const identity = env.DEV_PROXY_IDENTITY || DEV_IDENTITY_DEFAULT;

  return {
    plugins: [vue()],

    // Read the repo-root `.env` — the same file the four Python services read.
    //
    // Vite exposes ONLY names beginning with `VITE_` to the browser bundle, so reading a
    // file that also holds ARK_API_KEY, IDENTITY_WHATSAPP_TOKEN and RECORDS_API_KEY leaks
    // none of them. That guarantee is the whole basis of this setting, so it is asserted
    // rather than assumed — see `src/config.spec.ts`, which fails if a secret is ever
    // given a VITE_ name, including by having its value copied onto one.
    envDir: REPO_ROOT,

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

      /* DEVELOPMENT ONLY. `vite build` does not emit any of this, which is worth stating
         because it is precisely the trap: with the paths below proxied, a local run
         reaches four services through one origin and needs no VITE_*_BASE_URL at all — so
         the app appears to work while being configured for nothing, and the first
         deployment to a separate domain 404s on every call. VITE_API_BASE_URL and
         VITE_IDENTITY_BASE_URL are what express the deployed case; see the root `.env`. */
      proxy: {
        // Identity. Authentication left the chat backend, so `/auth` no longer exists on
        // the backend's port — login, registration and refresh are all under /v1/auth.
        '/v1': identity,
        ...Object.fromEntries(BACKEND_PATHS.map((path) => [path, backend])),
      },
    },
  };
});
