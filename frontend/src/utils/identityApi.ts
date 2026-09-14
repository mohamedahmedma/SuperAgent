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
 *
 * Renewing the session lives in `utils/session.ts`, and the live session in
 * `stores/auth.ts`; this module only knows where identity is.
 */
const identityApi = axios.create({
  baseURL: import.meta.env.VITE_IDENTITY_BASE_URL || '',
  timeout: 30000,
});

export const ACCESS_TOKEN_KEY = 'accessToken';
export const REFRESH_TOKEN_KEY = 'refreshToken';

export default identityApi;
