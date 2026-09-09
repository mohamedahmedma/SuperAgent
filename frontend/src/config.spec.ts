/**
 * Where the Vue app's configuration comes from, and what must never come with it.
 *
 * `vite.config.ts` sets `envDir` to the repo root so there is ONE `.env` in the project
 * rather than two that have to agree. Nothing reads both, which is what made the old
 * `frontend/.env` dangerous: a UI pointed at the wrong identity service built cleanly,
 * deployed cleanly, and failed at a parent's first sign-in.
 *
 * That root file also holds every secret the estate has. Vite only inlines names starting
 * with `VITE_`, so they stay out of the bundle — but that is a naming convention holding
 * back an API key, and a convention with nothing asserting it is a convention that gets
 * broken by someone in a hurry. These tests are the assertion.
 */
import { readFileSync, existsSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

import { loadEnv } from 'vite';
import { describe, expect, it } from 'vitest';

import viteConfig from '../vite.config';

/** The resolved config.
 *
 *  `defineConfig` accepts either an object or a function of the build context, and this
 *  project's config became a function when the dev proxy started reading the environment.
 *  Resolving it here rather than asserting on a shape keeps these tests about what the
 *  config SAYS instead of how it is written. */
const config: any = typeof viteConfig === 'function'
  ? (viteConfig as any)({ command: 'serve', mode: 'development' })
  : viteConfig;

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, '..', '..');

/** Names that must never be exposed to a browser, whatever they are called. */
const SECRET_NAMES = [
  'ARK_API_KEY',
  'VISION_API_KEY',
  'RECORDS_API_KEY',
  'IDENTITY_WHATSAPP_TOKEN',
  'IDENTITY_WHATSAPP_APP_SECRET',
  'IDENTITY_WHATSAPP_VERIFY_TOKEN',
  'JWT_SECRET_KEY',
  'LANGSMITH_API_KEY',
  'DATABASE_URL',
  'SIS_REGISTRAR_API_KEY',
  'ADMIN_INVITE_CODE',
];

/** A minimal `.env` reader. Deliberately not dotenv: this must see the file as written,
 *  including the keys dotenv would happily hand to `process.env` and hide among the rest. */
function parseEnvFile(path: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of readFileSync(path, 'utf-8').split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eq = trimmed.indexOf('=');
    if (eq <= 0) continue;
    out[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim();
  }
  return out;
}

/** The real `.env` where there is one, else the tracked example. `.env` is gitignored, so
 *  a checkout without one is normal and must still be able to run this file. */
function rootEnvPath(): string {
  const real = resolve(REPO_ROOT, '.env');
  return existsSync(real) ? real : resolve(REPO_ROOT, '.env.example');
}

describe('the frontend reads the repo-root .env', () => {
  it('points envDir at the repo root', () => {
    expect(config.envDir).toBe(REPO_ROOT);
  });

  it('no longer keeps a second env file beside the config', () => {
    // The specific regression. A leftover `frontend/.env` is now silently ignored, which
    // is worse than it was before: an operator edits it and nothing at all happens.
    expect(existsSync(resolve(REPO_ROOT, 'frontend', '.env'))).toBe(false);
  });

  it('finds the settings the app actually reads', () => {
    const env = parseEnvFile(rootEnvPath());
    // Declared, and allowed to be empty — empty is what makes the dev proxy the single
    // identity everything else talks to. Presence is the contract; the value is a choice.
    expect(env).toHaveProperty('VITE_IDENTITY_BASE_URL');
    expect(env).toHaveProperty('VITE_SCHOOL_CODE');
  });
});

describe('the dev server proxy', () => {
  /* Development only — `vite build` emits none of it — and that is exactly why it earns
     tests. With these paths proxied, a local run reaches four services through one origin
     and needs no VITE_*_BASE_URL at all, so the app looks configured while being
     configured for nothing. The first deploy to a separate domain then 404s on every call.
     These assertions pin what the proxy covers, so the deployed cases in
     `utils/api.spec.ts` and the dev case here stay in step. */
  const resolve_ = (env: Record<string, string> = {}) => {
    const saved: Record<string, string | undefined> = {};
    for (const [key, value] of Object.entries(env)) {
      saved[key] = process.env[key];
      process.env[key] = value;
    }
    try {
      return (typeof viteConfig === 'function'
        ? (viteConfig as any)({ command: 'serve', mode: 'development' })
        : viteConfig) as any;
    } finally {
      for (const [key, value] of Object.entries(saved)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
  };

  it('forwards every backend path the app actually calls', () => {
    /* The list must match what `utils/api.ts` and the stores request. A path missing here
       404s in development only, which is the most expensive place to find it: it looks
       like a backend bug. */
    const { proxy } = resolve_().server;
    for (const path of ['/chat', '/sessions', '/documents', '/media']) {
      expect(proxy, `${path} is not proxied`).toHaveProperty(path);
    }
  });

  it('sends them all to one place', () => {
    // The de-duplication. The address used to be written out four times.
    const { proxy } = resolve_().server;
    const targets = new Set(['/chat', '/sessions', '/documents', '/media'].map((p) => proxy[p]));
    expect(targets.size).toBe(1);
  });

  it('keeps the documented ports when nothing is configured', () => {
    // Behaviour preservation: an existing checkout must run exactly as it did.
    const { proxy } = resolve_().server;
    expect(proxy['/chat']).toBe('http://localhost:8000');
    expect(proxy['/v1']).toBe('http://localhost:8200');
  });

  it('proxies identity separately from the backend', () => {
    /* They are different services on different ports. Authentication left the chat
       backend, so `/v1/auth/login` on :8000 is a 404, and collapsing these two would be a
       plausible-looking simplification that breaks every sign-in. */
    const { proxy } = resolve_().server;
    expect(proxy['/v1']).not.toBe(proxy['/chat']);
  });

  it('can be repointed without editing this file', () => {
    const { proxy } = resolve_({
      DEV_PROXY_BACKEND: 'http://127.0.0.1:9000',
      DEV_PROXY_IDENTITY: 'http://127.0.0.1:9200',
    }).server;
    expect(proxy['/chat']).toBe('http://127.0.0.1:9000');
    expect(proxy['/media']).toBe('http://127.0.0.1:9000');
    expect(proxy['/v1']).toBe('http://127.0.0.1:9200');
  });
});

describe('no secret can reach the browser bundle', () => {
  const fileEnv = parseEnvFile(rootEnvPath());
  const exposed = loadEnv('production', config.envDir as string, 'VITE_');

  it('exposes only VITE_-prefixed names', () => {
    for (const key of Object.keys(exposed)) {
      expect(key.startsWith('VITE_')).toBe(true);
    }
  });

  it('exposes none of the estate\'s secrets by name', () => {
    for (const name of SECRET_NAMES) {
      expect(exposed).not.toHaveProperty(name);
      expect(exposed).not.toHaveProperty(`VITE_${name}`);
    }
  });

  it('never mirrors a non-VITE value into a VITE_ name', () => {
    /* The failure this is really guarding: someone needs a key in the browser, copies its
       VALUE onto a new `VITE_` name, and the prefix rule is satisfied while the secret
       ships anyway. Compared by value, so renaming does not evade it.

       Short and empty values are skipped — `VITE_SCHOOL_CODE=` and `LANGSMITH_TRACING=`
       are both empty-ish and matching them would mean nothing. */
    const secretValues = new Set(
      Object.entries(fileEnv)
        .filter(([key, value]) => !key.startsWith('VITE_') && value.length >= 12)
        .map(([, value]) => value),
    );

    for (const [key, value] of Object.entries(exposed)) {
      if (!value || value.length < 12) continue;
      expect(
        secretValues.has(value),
        `${key} carries the same value as a non-VITE variable — that is a secret in the bundle`,
      ).toBe(false);
    }
  });
});
