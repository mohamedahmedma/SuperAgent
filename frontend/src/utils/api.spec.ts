/**
 * Where the chat backend's requests are actually sent.
 *
 * `utils/api.ts` used to create its axios client with a timeout and nothing else. Every
 * call — `/sessions`, `/documents`, `/documents/upload/async` — therefore resolved against
 * whatever origin served the page. Co-hosted, that is right and invisible. Deployed to its
 * own domain, every one of them 404s against the static server, and there was no setting
 * to correct it and no error that named one.
 *
 * So these tests are about a default and an override, in that order: the default must be
 * byte-identical to the old no-baseURL behaviour, or this fix has broken development to
 * repair production.
 */
import axios from 'axios';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

/** Every setting this module can read, so a case controls all of them.
 *
 *  Listed rather than left implicit because the alternative is a test that passes on a
 *  developer's checkout and fails the day somebody fills in the real `.env` — vitest loads
 *  the repo-root file through `envDir`, so an unstubbed name arrives carrying the
 *  DEPLOYMENT's value. That is exactly how this file broke once. */
const SETTINGS = ['VITE_API_BASE_URL', 'VITE_IDENTITY_BASE_URL', 'VITE_SCHOOL_CODE'];

/** Re-import `utils/api` with the environment as `env` describes it.
 *
 *  The module reads `import.meta.env` once, at import — which is the whole point, since
 *  Vite inlines the value at build time — so the registry has to be reset per case.
 *
 *  Anything `env` does not name is stubbed EMPTY rather than left alone, so "not
 *  configured" means not configured no matter what the real `.env` holds. */
async function loadApi(env: Record<string, string>) {
  vi.resetModules();
  for (const key of SETTINGS) {
    vi.stubEnv(key, env[key] ?? '');
  }
  for (const [key, value] of Object.entries(env)) {
    vi.stubEnv(key, value);
  }
  return import('./api');
}

describe('the chat backend base URL', () => {
  beforeEach(() => {
    // localStorage is read by the request interceptor; the module must import without one.
    const store = new Map<string, string>();
    (globalThis as any).localStorage = {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => void store.set(key, String(value)),
      removeItem: (key: string) => void store.delete(key),
      clear: () => store.clear(),
    };
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it('is empty when nothing is configured, so calls stay same-origin', async () => {
    /* The behaviour-preservation case, and the one that matters most: development and any
       co-hosted deployment must be exactly as they were. An empty baseURL makes axios
       resolve '/sessions' against the page's origin — identical to having no baseURL. */
    const { API_BASE_URL, default: api } = await loadApi({ VITE_API_BASE_URL: '' });
    expect(API_BASE_URL).toBe('');
    expect(api.defaults.baseURL).toBe('');
  });

  it('is empty when the variable is absent entirely', async () => {
    /* A checkout with no `.env` at all. `loadApi` stubs every known setting to empty, so
       this asserts the module's own fallback rather than whatever the repo happens to be
       configured with today. */
    const { API_BASE_URL } = await loadApi({});
    expect(API_BASE_URL).toBe('');
  });

  it('sends requests to the configured origin when one is set', async () => {
    const { API_BASE_URL, default: api } = await loadApi({
      VITE_API_BASE_URL: 'https://api.aurexis.cc',
    });
    expect(API_BASE_URL).toBe('https://api.aurexis.cc');
    expect(api.defaults.baseURL).toBe('https://api.aurexis.cc');
  });

  it('builds the absolute URL a cross-origin deployment needs', async () => {
    /* Asserted through axios's own resolution rather than by string-concatenating in the
       test, so this still holds if the client is ever built differently. */
    const { default: api } = await loadApi({
      VITE_API_BASE_URL: 'https://api.aurexis.cc',
    });
    const resolved = axios.getUri({
      url: '/sessions',
      baseURL: api.defaults.baseURL,
    });
    expect(resolved).toBe('https://api.aurexis.cc/sessions');
  });

  it('leaves an already-absolute request URL alone', async () => {
    // Asset URLs may arrive absolute from the backend; a baseURL must not corrupt them.
    const { default: api } = await loadApi({
      VITE_API_BASE_URL: 'https://api.aurexis.cc',
    });
    const resolved = axios.getUri({
      url: 'https://cdn.example.com/x.png',
      baseURL: api.defaults.baseURL,
    });
    expect(resolved).toBe('https://cdn.example.com/x.png');
  });

  it('keeps the timeout it always had', async () => {
    const { default: api } = await loadApi({ VITE_API_BASE_URL: '' });
    expect(api.defaults.timeout).toBe(60000);
  });
});

describe('apiUrl, for the callers axios cannot serve', () => {
  /* Two callers need this: the chat stream, which reads a ReadableStream, and an
     authenticated <img>, which is fetched as a blob. Both used to pass a bare path to
     `fetch`, which resolves against the page's origin — so both quietly ignored the base
     URL every axios call respected. */
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it('returns the path unchanged when nothing is configured', async () => {
    // The behaviour-preservation case: identical to passing the literal to `fetch`.
    const { apiUrl } = await loadApi({ VITE_API_BASE_URL: '' });
    expect(apiUrl('/chat/stream')).toBe('/chat/stream');
    expect(apiUrl('/media/abc123')).toBe('/media/abc123');
  });

  it('prefixes the configured origin', async () => {
    const { apiUrl } = await loadApi({ VITE_API_BASE_URL: 'https://api.aurexis.cc' });
    expect(apiUrl('/chat/stream')).toBe('https://api.aurexis.cc/chat/stream');
    expect(apiUrl('/media/abc123')).toBe('https://api.aurexis.cc/media/abc123');
  });

  it('leaves an absolute URL alone', async () => {
    /* The backend may start emitting absolute asset URLs — a CDN, a signed storage link —
       and prefixing one produces a URL that is wrong in a way no error explains. */
    const { apiUrl } = await loadApi({ VITE_API_BASE_URL: 'https://api.aurexis.cc' });
    expect(apiUrl('https://cdn.example.com/x.png')).toBe('https://cdn.example.com/x.png');
    expect(apiUrl('http://cdn.example.com/x.png')).toBe('http://cdn.example.com/x.png');
  });

  it('leaves a data: URI alone', async () => {
    // Inline assets arrive as complete `data:` URIs — see AssetRenditionMode.INLINE.
    const { apiUrl } = await loadApi({ VITE_API_BASE_URL: 'https://api.aurexis.cc' });
    const inline = 'data:image/png;base64,iVBORw0KGgo=';
    expect(apiUrl(inline)).toBe(inline);
  });

  it('leaves a blob: URL and a protocol-relative URL alone', async () => {
    const { apiUrl } = await loadApi({ VITE_API_BASE_URL: 'https://api.aurexis.cc' });
    expect(apiUrl('blob:https://x/y')).toBe('blob:https://x/y');
    expect(apiUrl('//cdn.example.com/x.png')).toBe('//cdn.example.com/x.png');
  });
});
