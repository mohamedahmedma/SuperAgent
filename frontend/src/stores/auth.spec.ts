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
