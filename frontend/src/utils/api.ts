import axios from 'axios';
import { ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY, refreshAccessToken } from '@/utils/identityApi';

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

// Request interceptor to attach Bearer token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem(ACCESS_TOKEN_KEY);
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Serialises concurrent refreshes. Without it, a page that fires four requests at once
// after the token expires performs four refreshes, and three of them race to overwrite
// the token with an older value.
let inFlightRefresh: Promise<string | null> | null = null;

function refreshOnce(): Promise<string | null> {
  if (!inFlightRefresh) {
    inFlightRefresh = refreshAccessToken().finally(() => {
      inFlightRefresh = null;
    });
  }
  return inFlightRefresh;
}

// Response interceptor: a 401 now means "try refreshing once", not "log out".
//
// Access tokens are short-lived by design — that is what bounds the window in which a
// revoked session keeps working. Without this, a parent would be thrown back to the
// login screen every thirty minutes, which is a worse experience than the old
// twenty-four-hour token and would get the TTL raised back up for the wrong reason.
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;

    if (error.response?.status === 401 && original && !original._retriedAfterRefresh) {
      original._retriedAfterRefresh = true;

      const token = await refreshOnce();
      if (token) {
        original.headers.Authorization = `Bearer ${token}`;
        return api(original);
      }
    }

    if (error.response?.status === 401) {
      localStorage.removeItem(ACCESS_TOKEN_KEY);
      localStorage.removeItem(REFRESH_TOKEN_KEY);
      window.dispatchEvent(new CustomEvent('unauthorized'));
    }

    return Promise.reject(error);
  }
);

export default api;
