/**
 * The session's tokens: where they are kept, when one is stale, and how it is renewed.
 *
 * Pure helpers over storage and the identity API. The auth store (`stores/auth.ts`) is the
 * authority on the live session and composes these; nothing here holds state of its own,
 * which is what lets every rule be tested with a clock and a fake response.
 *
 * ## Why the token is renewed before it expires, not after
 *
 * Access tokens live thirty minutes. The old arrangement renewed one only when a request
 * came back 401 — and only for axios calls. The two callers that use `fetch` directly (the
 * chat stream and the image loader) read the token the page had started with, so half an
 * hour into a session every picture read "Image unavailable" and the next message signed
 * the parent out. Judging the token by its own `exp` claim, a minute early, means no
 * caller ever sends one that is about to be refused.
 *
 * ## Rotation
 *
 * Identity spends the refresh token on every exchange and returns a new one
 * (`identity/application/services/sessions.py`). Both tokens are therefore replaced
 * together, and a server that predates rotation — which returns no `refresh_token` —
 * leaves the old one in place.
 */
import identityApi, { ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY } from '@/utils/identityApi';

export interface SessionTokens {
  accessToken: string;
  refreshToken: string;
}

/** How long before an access token's expiry it counts as stale. Generous against clock
 *  skew between the phone and the server, and against a request that takes a while to
 *  arrive: a token good for another fifty seconds is not one to start a chat stream with. */
export const REFRESH_AHEAD_MS = 60_000;

export type RefreshOutcome =
  /** New tokens. The refresh token presented is spent; keep these. */
  | { status: 'refreshed'; tokens: SessionTokens }
  /** Identity refused the refresh token: revoked, replayed, or its window has closed. The
   *  session is over and nothing short of signing in again reopens it. */
  | { status: 'rejected' }
  /** Identity could not be reached or failed. The session is NOT over — the token in hand
   *  may still work, and the next attempt may succeed. */
  | { status: 'unavailable'; error: unknown };

export function readStoredTokens(): SessionTokens {
  try {
    return {
      accessToken: localStorage.getItem(ACCESS_TOKEN_KEY) || '',
      refreshToken: localStorage.getItem(REFRESH_TOKEN_KEY) || '',
    };
  } catch {
    // A private window that refuses storage, or a browser that cleared it. The session then
    // lives only in memory, which is a working session for as long as the tab is open.
    return { accessToken: '', refreshToken: '' };
  }
}

export function writeStoredTokens(tokens: SessionTokens): void {
  try {
    localStorage.setItem(ACCESS_TOKEN_KEY, tokens.accessToken);
    if (tokens.refreshToken) {
      localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refreshToken);
    } else {
      localStorage.removeItem(REFRESH_TOKEN_KEY);
    }
  } catch {
    /* see readStoredTokens */
  }
}

export function clearStoredTokens(): void {
  try {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  } catch {
    /* see readStoredTokens */
  }
}

/**
 * When the token expires, in milliseconds since the epoch — or null when it does not say.
 *
 * Read without verifying: the browser holds no key and needs none, because it is deciding
 * when to ask for a new token, not whether to trust this one. Anything that is not a JWT
 * with a numeric `exp` is "does not say", and is left for the server to judge.
 */
export function tokenExpiresAt(token: string): number | null {
  const parts = token.split('.');
  if (parts.length !== 3) return null;
  try {
    const payload = JSON.parse(decodeBase64Url(parts[1]));
    return typeof payload.exp === 'number' ? payload.exp * 1000 : null;
  } catch {
    return null;
  }
}

function decodeBase64Url(segment: string): string {
  const base64 = segment.replace(/-/g, '+').replace(/_/g, '/');
  const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4);
  const binary = atob(padded);
  // `atob` yields one char per byte; claims may carry a parent's name, so decode as UTF-8.
  try {
    return decodeURIComponent(
      Array.from(binary, (char) => `%${char.charCodeAt(0).toString(16).padStart(2, '0')}`).join(''),
    );
  } catch {
    return binary;
  }
}

/** Whether a token should be renewed before being sent. Missing: yes. Not a JWT: no —
 *  there is nothing to judge it by, and the server will say. Otherwise: when it expires
 *  within `REFRESH_AHEAD_MS`. */
export function isStale(token: string, now: number = Date.now()): boolean {
  if (!token) return true;
  const expiresAt = tokenExpiresAt(token);
  if (expiresAt === null) return false;
  return expiresAt - now <= REFRESH_AHEAD_MS;
}

/** Exchange the refresh token with identity. Never throws: the three outcomes a caller
 *  must tell apart are the return value. */
export async function exchangeRefreshToken(refreshToken: string): Promise<RefreshOutcome> {
  if (!refreshToken) return { status: 'rejected' };
  try {
    const { data } = await identityApi.post('/v1/auth/refresh', { refresh_token: refreshToken });
    return {
      status: 'refreshed',
      tokens: {
        accessToken: String(data.access_token || ''),
        // Rotation: the new one replaces the one just spent. An older identity that
        // rotates nothing sends none, and the old one stays valid.
        refreshToken: String(data.refresh_token || refreshToken),
      },
    };
  } catch (error: any) {
    const status = Number(error?.response?.status);
    // Identity answered, and said no. Anything in the 4xx range is its verdict on the
    // token; a 5xx or no response at all is identity's problem, not the session's.
    if (status >= 400 && status < 500 && status !== 429) {
      return { status: 'rejected' };
    }
    return { status: 'unavailable', error };
  }
}

const REFRESH_LOCK = 'aurexis-session-refresh';

/**
 * Run `work` while holding the browser-wide refresh lock.
 *
 * Two tabs that both find the token stale would both spend the same refresh token. Identity
 * forgives that inside a short grace window, but there is no reason to lean on it: where
 * the browser offers Web Locks, the second tab waits for the first and then finds fresh
 * tokens already in storage. Where it does not, the work simply runs.
 */
export async function withRefreshLock<T>(work: () => Promise<T>): Promise<T> {
  const locks = (globalThis as any).navigator?.locks;
  if (locks && typeof locks.request === 'function') {
    return locks.request(REFRESH_LOCK, work);
  }
  return work();
}
