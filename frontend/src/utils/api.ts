import axios from 'axios';
import { ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY, refreshAccessToken } from '@/utils/identityApi';

const api = axios.create({
  timeout: 60000,
});

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
