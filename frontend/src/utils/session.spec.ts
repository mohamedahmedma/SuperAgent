/**
 * The token helpers the auth store composes: when a token is stale, and what a refresh
 * exchange means. `stores/auth.spec.ts` covers what the store does with the answers.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

import identityApi from '@/utils/identityApi';
import {
  REFRESH_AHEAD_MS,
  exchangeRefreshToken,
  isStale,
  readStoredTokens,
  tokenExpiresAt,
  writeStoredTokens,
} from './session';

vi.mock('@/utils/identityApi', () => ({
  default: { get: vi.fn(), post: vi.fn() },
  ACCESS_TOKEN_KEY: 'accessToken',
  REFRESH_TOKEN_KEY: 'refreshToken',
}));

const post = identityApi.post as unknown as ReturnType<typeof vi.fn>;

/** An unsigned JWT with these claims. The browser reads `exp`; it never verifies. */
export function jwt(claims: Record<string, unknown>): string {
  const encode = (value: object) => Buffer.from(JSON.stringify(value)).toString('base64url');
  return `${encode({ alg: 'RS256', kid: 'k1' })}.${encode(claims)}.signature`;
}

const NOW = Date.parse('2026-09-14T12:00:00Z');
const seconds = (n: number) => n * 1000;

describe('reading when a token expires', () => {
  it('reads exp, in milliseconds', () => {
    expect(tokenExpiresAt(jwt({ exp: NOW / 1000 + 1800 }))).toBe(NOW + seconds(1800));
  });

  it('does not say for anything that is not a JWT with an exp', () => {
    expect(tokenExpiresAt('test-token')).toBeNull();
    expect(tokenExpiresAt('')).toBeNull();
    expect(tokenExpiresAt(jwt({ sub: 'no-exp' }))).toBeNull();
    expect(tokenExpiresAt('a.b.c')).toBeNull();
  });

  it('survives a claim set that carries a name in Arabic', () => {
    expect(tokenExpiresAt(jwt({ exp: 1_800_000_000, name: 'فاطمة علي' }))).toBe(1_800_000_000_000);
  });
});

describe('deciding whether to renew first', () => {
  it('a missing token is stale', () => {
    expect(isStale('', NOW)).toBe(true);
  });

  it('a token with a minute or less left is stale', () => {
    expect(isStale(jwt({ exp: (NOW + REFRESH_AHEAD_MS) / 1000 }), NOW)).toBe(true);
    expect(isStale(jwt({ exp: (NOW + seconds(30)) / 1000 }), NOW)).toBe(true);
    expect(isStale(jwt({ exp: (NOW - seconds(5)) / 1000 }), NOW)).toBe(true);
  });

  it('a token with more than a minute left is not', () => {
    expect(isStale(jwt({ exp: (NOW + seconds(300)) / 1000 }), NOW)).toBe(false);
  });

  it('a token that does not say is left for the server to judge', () => {
    expect(isStale('test-token', NOW)).toBe(false);
  });
});

describe('exchanging the refresh token', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('returns the rotated pair', async () => {
    post.mockResolvedValueOnce({ data: { access_token: 'a2', refresh_token: 'r2' } });

    const outcome = await exchangeRefreshToken('r1');

    expect(post).toHaveBeenCalledWith('/v1/auth/refresh', { refresh_token: 'r1' });
    expect(outcome).toEqual({ status: 'refreshed', tokens: { accessToken: 'a2', refreshToken: 'r2' } });
  });

  it('keeps the old refresh token when identity rotates none', async () => {
    // An identity service from before rotation returns only the access token.
    post.mockResolvedValueOnce({ data: { access_token: 'a2' } });

    const outcome = await exchangeRefreshToken('r1');

    expect(outcome).toEqual({ status: 'refreshed', tokens: { accessToken: 'a2', refreshToken: 'r1' } });
  });

  it('is rejected when identity refuses the token', async () => {
    post.mockRejectedValueOnce({ response: { status: 401, data: { detail: { code: 'not_authorized' } } } });
    expect(await exchangeRefreshToken('r1')).toEqual({ status: 'rejected' });
  });

  it('is unavailable — not rejected — when identity cannot answer', async () => {
    // Offline, or identity down. The session is not over; a network blip must not sign a
    // parent out.
    post.mockRejectedValueOnce(new Error('Network Error'));
    expect((await exchangeRefreshToken('r1')).status).toBe('unavailable');

    post.mockRejectedValueOnce({ response: { status: 503 } });
    expect((await exchangeRefreshToken('r1')).status).toBe('unavailable');
  });

  it('is rejected without asking when there is nothing to refresh with', async () => {
    expect(await exchangeRefreshToken('')).toEqual({ status: 'rejected' });
    expect(post).not.toHaveBeenCalled();
  });
});

describe('storage', () => {
  beforeEach(() => {
    const store = new Map<string, string>();
    vi.stubGlobal('localStorage', {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => void store.set(key, String(value)),
      removeItem: (key: string) => void store.delete(key),
      clear: () => store.clear(),
    });
  });

  it('round-trips both tokens', () => {
    writeStoredTokens({ accessToken: 'a1', refreshToken: 'r1' });
    expect(readStoredTokens()).toEqual({ accessToken: 'a1', refreshToken: 'r1' });
  });

  it('removes a refresh token that is written empty', () => {
    writeStoredTokens({ accessToken: 'a1', refreshToken: 'r1' });
    writeStoredTokens({ accessToken: 'a2', refreshToken: '' });
    expect(readStoredTokens()).toEqual({ accessToken: 'a2', refreshToken: '' });
  });

  it('reads empty tokens when storage refuses', () => {
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('SecurityError');
      },
    });
    expect(readStoredTokens()).toEqual({ accessToken: '', refreshToken: '' });
  });
});
