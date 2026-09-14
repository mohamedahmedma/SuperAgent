import axios from 'axios';
import { getActivePinia } from 'pinia';
import { useAuthStore } from '@/stores/auth';
import { readStoredTokens } from '@/utils/session';

/**
 * The chat backend's origin.
 *
 * EMPTY by default, and that is not a placeholder — empty means every call goes to the
 * app's own origin, which is what the vite dev proxy rewrites to localhost:8000 and what a
 * reverse proxy in front of a co-hosted deployment already handles. Development behaviour
 * is therefore exactly what it has always been.
 *
 * It exists for the deployment that does NOT co-host them: a UI on
 * superagent.example.com and an API on api.example.com. Until this setting existed there
 * was no way to express that at all — `axios.create` was called with a timeout and nothing
 * else, so `/sessions` and `/documents` resolved against the UI's own origin and 404ed
 * against the static server. There was no variable to set and no error that named one.
 *
 * Set it alongside CORS_ALLOW_ORIGINS on the backend: a cross-origin call needs both ends
 * to agree, and setting only this one turns a 404 into a CORS failure.
 */
export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
});

/**
 * `path` against the chat backend, for the callers axios cannot serve.
 *
 * Two of them exist and both are unavoidable: the chat stream needs a `ReadableStream`,
 * which axios does not expose in the browser, and an authenticated `<img>` has to be
 * fetched as a blob. Both used to pass a bare "/chat/stream" or "/media/..." to `fetch`,
 * which resolves against the PAGE's origin — so both silently ignored the base URL that
 * every axios call respects, and a split deployment broke in two places that looked
 * nothing alike.
 *
 * An absolute URL is returned untouched. The backend is free to start emitting absolute
 * asset URLs — a CDN, a signed storage link — and prefixing one would corrupt it.
 */
export function apiUrl(path: string): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(path) || path.startsWith('//')) {
    return path;
  }
  return `${API_BASE_URL}${path}`;
}

/**
 * The auth store, which owns the live session — or nothing, for a module imported on its
 * own with no application around it (a spec re-importing this file to read its base URL).
 */
function sessionStore() {
  return getActivePinia() ? useAuthStore() : null;
}

// Request interceptor: attach a bearer token that will still be valid when it arrives.
//
// The store renews the token BEFORE it expires (see `utils/session.ts`), so a request is
// never sent with one that is about to be refused. Without a store the request carries
// whatever storage holds, which is what this did before the store learned to renew.
api.interceptors.request.use(
  async (config) => {
    const store = sessionStore();
    const token = store ? await store.ensureFreshToken() : readStoredTokens().accessToken;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor: a 401 means "renew once and retry", not "log out".
//
// A token the clock still called fresh can be refused — the signing key rotated, or the
// session was revoked from the admin side — so the store is asked to renew regardless of
// what it thinks of the token. A renewal identity refuses ends the session inside the
// store, which is what tells the app; a retry that still fails ends it here.
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;
    const store = sessionStore();

    if (error.response?.status === 401 && original && !original._retriedAfterRefresh && store) {
      original._retriedAfterRefresh = true;

      const token = await store.refreshSession();
      if (token) {
        original.headers.Authorization = `Bearer ${token}`;
        return api(original);
      }
      // The store has already ended the session and said so.
      return Promise.reject(error);
    }

    if (error.response?.status === 401) {
      if (store) {
        store.endSession();
      } else {
        window.dispatchEvent(new CustomEvent('unauthorized'));
      }
    }

    return Promise.reject(error);
  }
);

export default api;
