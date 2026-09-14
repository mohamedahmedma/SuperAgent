import axios from 'axios';

/**
 * Client for the identity service.
 *
 * Separate from `utils/api.ts` on purpose. That instance talks to the chat backend and
 * attaches a bearer token to everything; this one talks to a different service, and its
 * two most important calls — login and refresh — must NOT carry the token being
 * replaced. Sharing one instance is how a refresh ends up authenticated by the expired
 * credential it exists to renew.
 *
 * `VITE_IDENTITY_BASE_URL` is empty in development, where vite proxies `/v1` to
 * localhost:8200. In production it is the identity service's origin.
 */
const identityApi = axios.create({
  baseURL: import.meta.env.VITE_IDENTITY_BASE_URL || '',
  timeout: 30000,
});

export const ACCESS_TOKEN_KEY = 'accessToken';
export const REFRESH_TOKEN_KEY = 'refreshToken';

export interface IdentitySession {
  access_token: string;
  refresh_token?: string | null;
}

/** Store the complete browser-owned part of an identity session atomically.
 *
 * A login response is allowed to omit a refresh token.  In that case an older token must
 * not survive in localStorage: it belongs to a previous session and can otherwise be
 * presented by a later refresh or logout request.
 */
export function persistSession({ access_token, refresh_token }: IdentitySession): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, access_token);
  if (refresh_token) {
    localStorage.setItem(REFRESH_TOKEN_KEY, refresh_token);
  } else {
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  }
}

/** Clear both credentials together wherever an identity session ends. */
export function clearStoredSession(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

/**
 * Exchange the stored refresh token for a fresh access token.
 *
 * Returns the new access token, or null when there is nothing to refresh with or the
 * refresh itself was rejected. Callers treat null as "sign in again".
 */
export async function refreshAccessToken(): Promise<string | null> {
  const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);
  if (!refreshToken) return null;

  try {
    const response = await identityApi.post('/v1/auth/refresh', { refresh_token: refreshToken });
    const accessToken = response.data.access_token as string;
    persistSession({ access_token: accessToken, refresh_token: refreshToken });
    return accessToken;
  } catch {
    // The refresh token is expired, revoked, or its binding was removed — all of which
    // mean the session is genuinely over. Clear both so the app cannot loop retrying.
    clearStoredSession();
    return null;
  }
}

export default identityApi;
