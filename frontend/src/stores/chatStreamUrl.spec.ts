/**
 * Where the chat stream is POSTed.
 *
 * Every other call the store makes goes through axios and inherits its `baseURL`. This one
 * cannot: it reads the response as a `ReadableStream`, which axios does not expose in the
 * browser, so it calls `fetch` directly — and for as long as it passed the literal
 * '/chat/stream', it resolved that against the PAGE's origin and ignored the base URL
 * every other call obeyed.
 *
 * That is a nasty shape of bug: on a co-hosted deployment nothing is wrong, and on a split
 * one the app loads, the history sidebar populates over axios, and only the actual
 * conversation 404s. So this file asserts the routing specifically, with a stubbed
 * `apiUrl` that prefixes — which is exactly what a configured VITE_API_BASE_URL does.
 *
 * `chat.spec.ts` covers the same store with `apiUrl` stubbed as identity, which is the
 * unconfigured default; between them both deployments are covered.
 */
import { createPinia, setActivePinia } from 'pinia';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { useAuthStore } from './auth';
import { useChatStore } from './chat';

const CONFIGURED_ORIGIN = 'https://api.aurexis.cc';

// The origin is repeated as a literal here rather than referenced: `vi.mock` is hoisted
// above every const in the file, so naming CONFIGURED_ORIGIN inside it is a TDZ error.
vi.mock('@/utils/api', () => ({
  default: { get: vi.fn(), delete: vi.fn(), post: vi.fn() },
  // Stands in for a build with VITE_API_BASE_URL set.
  apiUrl: (path: string) => `https://api.aurexis.cc${path}`,
  API_BASE_URL: 'https://api.aurexis.cc',
}));

/** A `fetch` that answers with an immediately-finished SSE body and records its arguments. */
function recordingFetch() {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const mock = vi.fn((url: string, init: RequestInit) => {
    calls.push({ url, init });
    return Promise.resolve({
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: () => Promise.resolve({ done: true, value: undefined }),
          releaseLock: () => {},
          cancel: () => Promise.resolve(),
        }),
      },
    } as unknown as Response);
  });
  return { mock, calls };
}

function localStorageMock() {
  const store = new Map<string, string>();
  return {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, String(value)),
    removeItem: (key: string) => void store.delete(key),
    clear: () => store.clear(),
  };
}

describe('the chat stream honours the configured backend origin', () => {
  let fetches: ReturnType<typeof recordingFetch>;

  beforeEach(async () => {
    vi.restoreAllMocks();
    vi.stubGlobal('localStorage', localStorageMock());
    vi.stubGlobal('alert', vi.fn());
    fetches = recordingFetch();
    vi.stubGlobal('fetch', fetches.mock);

    setActivePinia(createPinia());
    const auth = useAuthStore();
    auth.token = 'test-token';
    auth.currentUser = { username: 'tester', role: 'user' };

    const chat = useChatStore();
    chat.setViewedSession('session_current', []);
    chat.userInput = 'Where is my daughter up to in maths?';
    await chat.handleSend();
  });

  it('posts to the configured origin, not the page it is served from', () => {
    expect(fetches.calls).toHaveLength(1);
    expect(fetches.calls[0].url).toBe(`${CONFIGURED_ORIGIN}/chat/stream`);
  });

  it('does not send a bare path any more', () => {
    // The literal regression. Stated separately so the failure message names the bug.
    expect(fetches.calls[0].url).not.toBe('/chat/stream');
  });

  it('still sends the credentials and the thread header it always did', () => {
    /* Routing changed; nothing else was allowed to. The backend reads the bearer token to
       identify the parent and X-Thread-ID to know which conversation this belongs to, and
       a cross-origin request that drops either fails in a way that looks like a routing
       problem — which is exactly the confusion this fix exists to end. */
    const headers = fetches.calls[0].init.headers as Record<string, string>;
    expect(headers.Authorization).toBe('Bearer test-token');
    expect(headers['X-Thread-ID']).toBe('session_current');
    expect(headers['Content-Type']).toBe('application/json');
    expect(fetches.calls[0].init.method).toBe('POST');
  });
});
