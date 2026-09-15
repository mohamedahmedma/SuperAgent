/**
 * How an answer's images are fetched.
 *
 * The backend hands the UI a PATH — `/media/{asset_id}`, built by `asset_url_path` in
 * `backend/assets/delivery.py` — and the image is fetched rather than given to `<img src>`
 * because the endpoint wants a bearer token and `<img>` cannot carry a header.
 *
 * A bare `fetch('/media/...')` resolves against the page's origin. Co-hosted that is
 * right; on a UI with its own domain it is a request to the static server, and the symptom
 * is peculiar enough to waste a day: the answer's text streams in perfectly and every
 * figure below it reads "Image unavailable".
 *
 * ## Why this reads the source
 *
 * These tests run in node with no DOM and the project has no `@vue/test-utils`, so the
 * component cannot be mounted to observe the call. `stores/documents.spec.ts` established
 * the same approach for the same reason. It is a weaker test than mounting — it proves the
 * call is *written* correctly, not that it *ran* — so the other half of the coverage is in
 * `utils/api.spec.ts`, which exercises `apiUrl` against real `/media/...` inputs.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const SOURCE = readFileSync(new URL('./MessageAssets.vue', import.meta.url), 'utf8');

describe('asset fetching is routed at the API, not at the page', () => {
  it('imports the URL builder', () => {
    expect(SOURCE).toMatch(/import\s*\{\s*apiUrl\s*\}\s*from\s*'\.\.\/\.\.\/utils\/api'/);
  });

  it('passes the asset URL through apiUrl', () => {
    expect(SOURCE).toContain('authorizedFetch(apiUrl(asset.url)');
  });

  it('no longer fetches the bare path', () => {
    // The literal regression, named so a failure here says what broke.
    expect(SOURCE).not.toContain('fetch(asset.url');
  });

  it('sends the bearer token through the auth store, which keeps it fresh', () => {
    /* Without the header the endpoint 401s, which renders as the same "Image unavailable"
       placeholder as a bad URL. Reading `authStore.token` here sent the token the page had
       started with, so half an hour into a session every picture failed the same way —
       `authorizedFetch` renews it first (see stores/auth.ts). */
    expect(SOURCE).toContain('authStore.authorizedFetch(');
    expect(SOURCE).not.toMatch(/Authorization:\s*`Bearer \$\{authStore\.token\}`/);
  });

  it('still short-circuits inline assets instead of fetching them', () => {
    // INLINE mode carries a complete `data:` URI. Fetching it would work and be waste.
    expect(SOURCE).toContain("asset.mode === 'inline' && asset.inline_data");
  });
});
