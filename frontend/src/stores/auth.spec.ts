import { createPinia, setActivePinia } from 'pinia';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useAuthStore } from './auth';
import identityApi from '@/utils/identityApi';

/**
 * The WhatsApp sign-in, from the browser's side.
 *
 * The flow is mostly *waiting* — for a parent to switch apps, tap send, and come back —
 * so the interesting behaviour is all in the polling: that it stops when it should, that
 * it survives a dropped request, and that it does not outlive the panel. None of that is
 * visible in a screenshot, and all of it is visible here with a fake clock.
 */
/**
 * A localStorage the store can write to.
 *
 * vitest runs these in node, with no DOM — and the auth store reads localStorage while
 * building its initial state, so this has to exist before the store is ever created.
 * jsdom would do it too, but pulling in a DOM for one API this file uses four times is a
 * dependency the other specs manage without.
 */
const store = new Map<string, string>();
(globalThis as any).localStorage = {
  getItem: (key: string) => store.get(key) ?? null,
  setItem: (key: string, value: string) => void store.set(key, String(value)),
  removeItem: (key: string) => void store.delete(key),
  clear: () => store.clear(),
};

vi.mock('@/utils/identityApi', () => ({
  default: { get: vi.fn(), post: vi.fn() },
  ACCESS_TOKEN_KEY: 'accessToken',
  REFRESH_TOKEN_KEY: 'refreshToken',
}));

const post = identityApi.post as unknown as ReturnType<typeof vi.fn>;
const get = identityApi.get as unknown as ReturnType<typeof vi.fn>;

/** An unsigned JWT expiring `secondsFromNow` from now. The store reads `exp`; it never verifies. */
function jwtExpiringIn(secondsFromNow: number): string {
  const encode = (value: object) => Buffer.from(JSON.stringify(value)).toString('base64url');
  const exp = Math.floor(Date.now() / 1000) + secondsFromNow;
  return `${encode({ alg: 'RS256' })}.${encode({ sub: 'guardian:abc', exp })}.sig`;
}

const STARTED = {
  poll_secret: 'the-browsers-half',
  link: 'https://wa.me/201288339613?text=SCHOOL%20VERIFY%3A%20ABC12345',
  message: 'SCHOOL VERIFY: ABC12345',
  business_number: '+201288339613',
  expires_at: new Date(Date.now() + 10 * 60 * 1000).toISOString(),
};

const TOKENS = {
  access_token: 'access',
  refresh_token: 'refresh',
  username: 'guardian:abc',
  role: 'parent',
  guardian_id: 'abc',
  display_name: 'فاطمة علي',
};

/** Let the promises inside one interval tick settle. */
const settle = async () => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

const refusal = (code: string, message = 'no') => ({
  response: { data: { detail: { code, message } } },
});

describe('signing in through WhatsApp', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.useFakeTimers();
    vi.clearAllMocks();
    localStorage.clear();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('asks for a challenge and shows the parent a link', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();

    await auth.startWhatsAppLogin();

    expect(post).toHaveBeenCalledWith('/v1/auth/whatsapp/start');
    expect(auth.whatsapp.status).toBe('waiting');
    expect(auth.whatsapp.link).toContain('wa.me/201288339613');
    // The number and the text, for the parent whose in-app browser eats the link.
    expect(auth.whatsapp.businessNumber).toBe('+201288339613');
    expect(auth.whatsapp.message).toBe('SCHOOL VERIFY: ABC12345');
  });

  it('sends no phone number, because there is none to send', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();

    await auth.startWhatsAppLogin();

    // One argument: the path. A body carrying a number would make this endpoint a way to
    // ask whether a given number belongs to a parent.
    expect(post).toHaveBeenCalledTimes(1);
    expect(post.mock.calls[0]).toHaveLength(1);
  });

  it('stops polling once the code has been sent', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();

    post.mockResolvedValue({ data: { status: 'code_sent', display_name: 'فاطمة علي' } });
    await vi.advanceTimersByTimeAsync(2000);
    await settle();

    expect(auth.whatsapp.status).toBe('code_sent');
    expect(auth.whatsapp.displayName).toBe('فاطمة علي');

    // And it really has stopped — a poller left running hits identity every two seconds
    // for as long as the tab is open.
    const callsSoFar = post.mock.calls.length;
    await vi.advanceTimersByTimeAsync(10000);
    expect(post.mock.calls.length).toBe(callsSoFar);
  });

  it('keeps polling through a dropped request', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();

    post.mockRejectedValueOnce(new Error('network blip'));
    await vi.advanceTimersByTimeAsync(2000);
    await settle();
    // Still waiting: the challenge is alive on the server and the parent is mid-flow, so
    // one failed poll must not tear their screen down.
    expect(auth.whatsapp.status).toBe('waiting');

    post.mockResolvedValue({ data: { status: 'code_sent', display_name: '' } });
    await vi.advanceTimersByTimeAsync(2000);
    await settle();
    expect(auth.whatsapp.status).toBe('code_sent');
  });

  it('gives up on its own once the challenge has expired', async () => {
    post.mockResolvedValueOnce({
      data: { ...STARTED, expires_at: new Date(Date.now() - 1000).toISOString() },
    });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();

    await vi.advanceTimersByTimeAsync(2000);
    await settle();

    expect(auth.whatsapp.status).toBe('failed');
    expect(auth.whatsapp.errorCode).toBe('expired');
    // Nothing was asked of identity: the browser knew the window had closed.
    expect(post).toHaveBeenCalledTimes(1);
  });

  it('reports a number the school does not hold', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();

    post.mockResolvedValue({ data: { status: 'rejected', display_name: '' } });
    await vi.advanceTimersByTimeAsync(2000);
    await settle();

    expect(auth.whatsapp.status).toBe('failed');
    expect(auth.whatsapp.errorCode).toBe('rejected');
  });

  it('signs the parent in when the code is right', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();
    auth.whatsapp.status = 'code_sent';
    auth.whatsapp.code = '482103';

    post.mockResolvedValueOnce({ data: TOKENS });
    await auth.submitWhatsAppCode();

    expect(auth.token).toBe('access');
    expect(auth.currentUser?.guardianId).toBe('abc');
    expect(localStorage.getItem('accessToken')).toBe('access');
    // The challenge is spent and its state cleared, so a reload cannot resubmit it.
    expect(auth.whatsapp.status).toBe('idle');
    expect(auth.whatsapp.pollSecret).toBe('');
  });

  it('removes a refresh token left by a previous session when the new session has none', () => {
    const auth = useAuthStore();
    localStorage.setItem('refreshToken', 'stale-token');

    auth.applySession({ ...TOKENS, refresh_token: undefined });

    expect(auth.token).toBe('access');
    expect(auth.refreshToken).toBe('');
    expect(localStorage.getItem('refreshToken')).toBeNull();
  });

  it('keeps a parent on the same screen after one wrong code', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();
    auth.whatsapp.status = 'code_sent';
    auth.whatsapp.code = '000000';

    post.mockRejectedValueOnce(refusal('bad_code', 'That code is not correct.'));
    await auth.submitWhatsAppCode();

    // They have four more tries; sending them back to the start would waste a code that
    // still works.
    expect(auth.whatsapp.status).toBe('code_sent');
    expect(auth.whatsapp.errorCode).toBe('bad_code');
    expect(auth.whatsapp.code).toBe('');
    expect(auth.token).toBe('');
  });

  it('sends a parent back to the start when the challenge is dead', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();
    auth.whatsapp.status = 'code_sent';
    auth.whatsapp.code = '482103';

    post.mockRejectedValueOnce(refusal('too_many_attempts'));
    await auth.submitWhatsAppCode();

    // Anything but a wrong code has killed the challenge, so leaving them typing into it
    // would be leaving them typing into something that cannot ever succeed.
    expect(auth.whatsapp.status).toBe('failed');
    expect(auth.whatsapp.errorCode).toBe('too_many_attempts');
  });

  it('keeps identity\'s refusal code so the page can say something specific', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();
    auth.whatsapp.status = 'code_sent';
    auth.whatsapp.code = '482103';

    post.mockRejectedValueOnce(refusal('expired', 'That verification has expired.'));
    await auth.submitWhatsAppCode();

    // The old unwrapper kept only `message`, which is right for a password login where
    // every failure reads the same and wrong here, where each needs its own screen.
    expect(auth.whatsapp.errorCode).toBe('expired');
    expect(auth.whatsapp.error).toBe('That verification has expired.');
  });

  it('stops polling when the attempt is abandoned', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();

    auth.resetWhatsApp();
    const callsSoFar = post.mock.calls.length;
    await vi.advanceTimersByTimeAsync(10000);

    expect(post.mock.calls.length).toBe(callsSoFar);
    expect(auth.whatsapp.status).toBe('idle');
  });

  it('stops polling on logout', async () => {
    post.mockResolvedValueOnce({ data: STARTED });
    const auth = useAuthStore();
    await auth.startWhatsAppLogin();

    post.mockResolvedValue({ data: {} });
    await auth.handleLogout();
    const callsSoFar = post.mock.calls.length;
    await vi.advanceTimersByTimeAsync(10000);

    expect(post.mock.calls.length).toBe(callsSoFar);
  });
});

/**
 * Staying signed in.
 *
 * Access tokens live thirty minutes. The store renews one before it expires and hands
 * every caller — axios, the chat stream, an image — a token that will still be valid when
 * the request lands; identity rotates the refresh token on each renewal and the session
 * runs on until the parent signs out. The failures these pin were all real: pictures
 * reading "Image unavailable" half an hour in, and the next message signing the parent out.
 */
describe('staying signed in', () => {
  const windowEvents: Event[] = [];

  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
    localStorage.clear();
    windowEvents.length = 0;
    // The store announces an ended session on `window`; node has none.
    vi.stubGlobal('window', {
      dispatchEvent: (event: Event) => {
        windowEvents.push(event);
        return true;
      },
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
  });

  const signedIn = (accessToken: string, refreshToken = 'r1') => {
    const auth = useAuthStore();
    auth.applyTokens({ accessToken, refreshToken });
    auth.currentUser = { username: 'guardian:abc', role: 'parent' };
    return auth;
  };

  it('hands out a token that is still good without asking identity', async () => {
    const token = jwtExpiringIn(25 * 60);
    const auth = signedIn(token);

    expect(await auth.ensureFreshToken()).toBe(token);
    expect(post).not.toHaveBeenCalled();
  });

  it('renews a token about to expire before handing it out, and keeps the rotated pair', async () => {
    const auth = signedIn(jwtExpiringIn(30));
    const fresh = jwtExpiringIn(30 * 60);
    post.mockResolvedValueOnce({ data: { access_token: fresh, refresh_token: 'r2' } });

    expect(await auth.ensureFreshToken()).toBe(fresh);

    expect(post).toHaveBeenCalledWith('/v1/auth/refresh', { refresh_token: 'r1' });
    expect(auth.refreshToken).toBe('r2');
    // Both tokens land in storage, so a reload — and the other tabs — carry on with them.
    expect(localStorage.getItem('accessToken')).toBe(fresh);
    expect(localStorage.getItem('refreshToken')).toBe('r2');
    expect(auth.isAuthenticated).toBe(true);
  });

  it('shares one exchange between callers that find the token stale at the same moment', async () => {
    const auth = signedIn(jwtExpiringIn(10));
    post.mockResolvedValue({ data: { access_token: jwtExpiringIn(1800), refresh_token: 'r2' } });

    await Promise.all([auth.ensureFreshToken(), auth.ensureFreshToken(), auth.ensureFreshToken()]);

    expect(post).toHaveBeenCalledTimes(1);
  });

  it('leaves a token that does not say when it expires alone', async () => {
    // Not a JWT: nothing to judge it by, and the server will say. This is also what every
    // other spec's `auth.token = 'test-token'` relies on.
    const auth = signedIn('test-token');
    expect(await auth.ensureFreshToken()).toBe('test-token');
    expect(post).not.toHaveBeenCalled();
  });

  it('ends the session when identity refuses the refresh token', async () => {
    const auth = signedIn(jwtExpiringIn(10));
    // A 401 with identity's envelope: the verdict is in the status, and it is final.
    post.mockRejectedValueOnce({
      response: { status: 401, data: { detail: { code: 'not_authorized', message: 'Invalid or expired refresh token.' } } },
    });

    expect(await auth.ensureFreshToken()).toBe('');

    expect(auth.token).toBe('');
    expect(auth.refreshToken).toBe('');
    expect(auth.isAuthenticated).toBe(false);
    expect(localStorage.getItem('refreshToken')).toBeNull();
    expect(windowEvents.map((event) => event.type)).toEqual(['unauthorized']);
    // Nothing to revoke: identity has already said the token is dead.
    expect(post).toHaveBeenCalledTimes(1);
  });

  it('keeps the parent signed in when identity cannot be reached', async () => {
    const token = jwtExpiringIn(10);
    const auth = signedIn(token);
    post.mockRejectedValueOnce(new Error('Network Error'));

    expect(await auth.ensureFreshToken()).toBe(token);
    expect(auth.isAuthenticated).toBe(true);
    expect(windowEvents).toEqual([]);
  });

  it('adopts tokens another tab renewed instead of spending the refresh token again', async () => {
    const auth = signedIn(jwtExpiringIn(10));
    const theirs = jwtExpiringIn(1800);
    localStorage.setItem('accessToken', theirs);
    localStorage.setItem('refreshToken', 'r-theirs');

    expect(await auth.ensureFreshToken()).toBe(theirs);
    expect(auth.refreshToken).toBe('r-theirs');
    expect(post).not.toHaveBeenCalled();
  });

  it('authorizedFetch sends a fresh token, and renews once when the server refuses it anyway', async () => {
    const auth = signedIn(jwtExpiringIn(1800));
    const renewed = jwtExpiringIn(1800 + 60);
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ status: 401, ok: false })
      .mockResolvedValueOnce({ status: 200, ok: true });
    vi.stubGlobal('fetch', fetchMock);
    post.mockResolvedValueOnce({ data: { access_token: renewed, refresh_token: 'r2' } });

    const response = await auth.authorizedFetch('/media/x', { headers: { 'X-Thread-ID': 's1' } });

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const [, firstInit] = fetchMock.mock.calls[0];
    const [, secondInit] = fetchMock.mock.calls[1];
    expect(firstInit.headers['X-Thread-ID']).toBe('s1');
    expect(secondInit.headers.Authorization).toBe(`Bearer ${renewed}`);
  });

  it('restores a session from storage, renewing a stale token before asking who the parent is', async () => {
    // Written before the store exists: this is a page load, not a sign-in.
    localStorage.setItem('accessToken', jwtExpiringIn(-600));
    localStorage.setItem('refreshToken', 'r1');
    const fresh = jwtExpiringIn(1800);
    post.mockResolvedValueOnce({ data: { access_token: fresh, refresh_token: 'r2' } });
    get.mockResolvedValueOnce({
      data: { username: 'guardian:abc', role: 'parent', guardian_id: 'abc', display_name: 'فاطمة علي' },
    });

    const auth = useAuthStore();
    await auth.restoreSession();

    expect(auth.isAuthenticated).toBe(true);
    expect(auth.currentUser?.guardianId).toBe('abc');
    expect(get).toHaveBeenCalledWith('/v1/auth/me', { headers: { Authorization: `Bearer ${fresh}` } });
  });

  it('does nothing on a page load with no session stored', async () => {
    const auth = useAuthStore();
    await auth.restoreSession();
    expect(post).not.toHaveBeenCalled();
    expect(get).not.toHaveBeenCalled();
    expect(auth.isAuthenticated).toBe(false);
  });
});
