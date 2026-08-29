/*
 * The console boots, and every screen renders against a stubbed service.
 *
 * `node tests/smoke.mjs`
 *
 * What this is for: the Python contract suite reads the console as text and can tell you a
 * screen calls a route nobody wrote. It cannot tell you a screen throws on its first render —
 * a typo'd import, a `.map` on something that arrives undefined, a hook called conditionally.
 * That class of bug takes the whole screen out and shows a blank page, and it is invisible to
 * every other test in this repository.
 *
 * So this actually mounts the app. React renders into jsdom, the router is driven through each
 * route in turn, and the assertion is that identifying text for that screen is on the page and
 * that nothing was written to `console.error` — which is where React reports a render that
 * threw, a key warning and a bad prop type, all three of which are real faults.
 *
 * Three things about the harness are deliberate and easy to get wrong again:
 *
 * **It bundles rather than loading `../web`.** jsdom does not execute `<script type="module">`,
 * so the built entry point cannot be loaded the way a browser loads it. esbuild produces the
 * same source as one IIFE, which jsdom does run. The CSS is dropped — this test is about
 * whether a screen renders, and bundling Bootstrap's stylesheet into jsdom costs seconds and
 * proves nothing.
 *
 * **`pretendToBeVisual: true` is load-bearing.** Without it jsdom has no
 * `requestAnimationFrame`, React falls back to a timer for its scheduling, and effects land
 * hundreds of milliseconds late. The screens then look broken here while working perfectly in a
 * browser — a harness artefact that costs an hour to diagnose the first time.
 *
 * **Every response is a fixture, and the fixtures are the shapes the service really returns.**
 * A stub that answers `{}` to everything would pass while proving nothing: the interesting
 * failures are a screen that assumes a list is present, or that a nullable field is not null.
 * So `null` grades, an unmarked attendance day and an empty class are all in here on purpose.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');

/* -- The service, as this test pretends it behaves --------------------------------- */

const YEAR = '2025-2026';

/*
 * The stubs live in `fixtures.json`, not here, and that is load-bearing rather than tidy.
 *
 * A jsdom test whose stubs are written by hand beside the screens they test will eventually
 * agree with a bug: `GET /students/{n}/placements` answers
 * `{student_number, count, placements: [...]}`, the record screen read it as a bare list, and
 * the fixture written next to it was a bare list too. Green test, broken screen, found only by
 * walking the console against the real service.
 *
 * `sis/tests/test_ui_fixtures.py` now checks every fixture in that file against the response
 * model its route declares in the OpenAPI document — required keys present, no invented keys,
 * list where a list is declared, recursively. So the shapes here are checked by the service's
 * own contract rather than by whoever last edited them.
 *
 * Which means: to add a screen to the walk, add its routes to `fixtures.json`. If the shape is
 * wrong, pytest says so before this test gets the chance to pass for the wrong reason.
 */
const FIXTURES = JSON.parse(readFileSync(resolve(HERE, 'fixtures.json'), 'utf8'));

const seen = new Set();
const unstubbed = [];

function answer(method, url) {
  /* The client may build an absolute URL or a root-relative one; the fixtures are keyed by
     path. Getting this wrong is silent — every lookup misses and every screen receives the
     catch-all shape, which for a route that returns a bare list is an object, and the screen
     throws on `.filter`. */
  const path = new URL(url, 'http://localhost:8300').pathname;
  const exact = FIXTURES[`${method} ${path}`];
  if (exact !== undefined) return exact;

  /* Anything not named above answers with an empty list and says so once. An empty *object*
     was the earlier default and it is the wrong one: most of these routes return a bare array,
     a screen does `(value || []).filter(...)`, and an object sails through the guard and throws
     inside it. Announcing the miss is the other half — a silently-defaulted fixture is a screen
     tested against a shape the service never sends. */
  if (!seen.has(path)) {
    seen.add(path);
    unstubbed.push(`${method} ${path}`);
  }
  return [];
}

/* -- Harness ---------------------------------------------------------------------- */

async function bundle() {
  const result = await build({
    entryPoints: [resolve(ROOT, 'src/main.jsx')],
    bundle: true,
    format: 'iife',
    write: false,
    /* jsdom is not a browser and does not run modules; one IIFE is the shape it can run. */
    platform: 'browser',
    target: 'es2020',
    /* The automatic runtime, matching @vitejs/plugin-react. The classic transform needs
       `React` in scope and none of these files import it — the build would succeed and every
       screen would throw `React is not defined` on its first render. */
    jsx: 'automatic',
    define: { 'process.env.NODE_ENV': '"development"' },
    /* The stylesheets are irrelevant to whether a screen renders, and bundling Bootstrap's
       into jsdom costs seconds. `empty` drops them without touching the imports. */
    loader: { '.css': 'empty' },
    logLevel: 'silent'
  });
  return result.outputFiles[0].text;
}

function newWindow(script, language = 'en') {
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on('jsdomError', (error) => errors.push(String(error)));

  const dom = new JSDOM(
    '<!doctype html><html data-bs-theme="light"><body><div id="app"></div></body></html>',
    {
      url: 'http://localhost:8300/ui/',
      runScripts: 'outside-only',
      /* Without this there is no requestAnimationFrame, React schedules on timers, and every
         effect lands late enough to look like a broken screen. */
      pretendToBeVisual: true,
      virtualConsole
    }
  );

  const { window } = dom;
  const requests = [];

  window.fetch = (input, init = {}) => {
    const url = String(typeof input === 'string' ? input : input.url);
    const method = (init.method || 'GET').toUpperCase();
    requests.push(`${method} ${url}`);
    const body = JSON.stringify(answer(method, url));
    return Promise.resolve({
      ok: true,
      status: 200,
      headers: { get: (name) => (name.toLowerCase() === 'content-type' ? 'application/json' : null) },
      json: () => Promise.resolve(JSON.parse(body)),
      text: () => Promise.resolve(body)
    });
  };
  window.FormData = class FormData {
    append() {}
  };
  /* jsdom has no layout, so scrolling is unimplemented and throws through the virtual console.
     The router scrolls to the top on every route change, which would otherwise report thirteen
     errors for correct behaviour. */
  window.scrollTo = () => {};

  /* React reports a render that threw, and several real prop faults, through console.error.
     Collected rather than printed so a screen that renders *something* while logging an error
     still fails the test. */
  window.console.error = (...args) => errors.push(args.map(String).join(' '));
  window.console.warn = () => {};

  window.localStorage.setItem('sis.lang', language);
  window.eval(script);
  return { dom, window, errors, requests };
}

const settle = (window, ms = 60) =>
  new Promise((done) => window.setTimeout(done, ms));

/* -- The walk --------------------------------------------------------------------- */

const SCREENS = [
  /* With a school already chosen, `#/school` is that school rather than the index — the
     school strip in the header is how you get back to the others. */
  { hash: '#/school', expect: ['Main School', 'academic years'] },
  { hash: `#/year?code=${YEAR}`, expect: ['Terms', 'Subjects'] },
  { hash: `#/level?code=Y3&year=${YEAR}`, expect: ['Classes', '3A'] },
  { hash: `#/class?code=3A&year=${YEAR}`, expect: ['Register', 'Layla Hassan'] },
  { hash: `#/class?code=3A&year=${YEAR}&tab=attendance`, expect: ['not yet marked'] },
  { hash: `#/class?code=3B&year=${YEAR}`, expect: ['Nobody is in 3B yet'] },
  { hash: '#/student?number=10432', expect: ['Layla Hassan', 'Insights', '10432'] },
  { hash: '#/student', expect: ['Find a child'] },
  { hash: '#/roster', expect: ['roster'] },
  { hash: '#/guardians', expect: ['Guardians'] },
  { hash: '#/marks', expect: ['Marks'] },
  { hash: '#/batches', expect: ['Batches'] }
];

async function main() {
  process.stdout.write('bundling… ');
  const script = await bundle();
  process.stdout.write(`${(script.length / 1024).toFixed(0)} KB\n`);

  const { window, errors, requests } = newWindow(script);
  await settle(window, 200);

  const mount = window.document.getElementById('app');
  if (!mount.childElementCount) {
    /* Printed before the assertion, because the reason the app did not mount is always in
       here and an assertion message that hides it costs a debugging session. */
    errors.slice(0, 5).forEach((line) => console.log(`  ${line.slice(0, 400)}`));
  }
  assert.ok(mount.childElementCount > 0, 'the app mounted nothing into #app');
  assert.ok(
    window.document.querySelector('.sis-app'),
    'the shell did not render — check main.jsx and App.jsx'
  );

  let failures = 0;
  for (const screen of SCREENS) {
    window.location.hash = screen.hash;
    await settle(window, 120);
    const text = window.document.body.textContent || '';
    const missing = screen.expect.filter((needle) => !text.includes(needle));
    if (missing.length) {
      failures += 1;
      console.log(`  FAIL ${screen.hash}`);
      console.log(`       missing ${JSON.stringify(missing)}`);
      console.log(`       rendered: ${text.slice(0, 200).replace(/\s+/g, ' ')}`);
    } else {
      console.log(`  ok   ${screen.hash}`);
    }
  }

  /* The invariants worth checking once, on the whole walk, rather than per screen. */
  const marks = (() => {
    window.location.hash = '#/student?number=10432';
    return null;
  })();
  await settle(window, 150);
  const record = window.document.body.textContent || '';
  assert.ok(
    record.includes('0%'),
    'a stated zero must render as 0% — `percentage || DASH` would hide it'
  );
  assert.ok(
    !/Science\s*0%/.test(record),
    'an unmarked subject must not render as 0% — `percentage ?? 0` would invent it'
  );

  assert.ok(
    requests.some((line) => line.includes(' /v1/')),
    `no request went to /v1 — the client base path may be wrong. Saw: ${requests.slice(0, 3)}`
  );

  /* Boot a second shell from the persisted Arabic preference. This catches the failure where
     labels translate but the document remains LTR, or direction flips but static chrome falls
     back to English. The route fixtures are shared; only browser-owned language differs. */
  const arabic = newWindow(script, 'ar');
  await settle(arabic.window, 200);
  assert.equal(arabic.window.document.documentElement.lang, 'ar');
  assert.equal(arabic.window.document.documentElement.dir, 'rtl');
  const arabicText = arabic.window.document.body.textContent || '';
  assert.ok(arabicText.includes('نظام معلومات الطلاب'), 'Arabic shell title did not render');
  assert.ok(arabicText.includes('المدرسة'), 'Arabic navigation did not render');
  errors.push(...arabic.errors);

  if (unstubbed.length) {
    console.log(`
${unstubbed.length} route(s) answered from the empty default:`);
    unstubbed.forEach((line) => console.log(`  ${line}`));
  }

  if (errors.length) {
    console.log(`\n${errors.length} console error(s) during the walk:`);
    errors.slice(0, 8).forEach((line) => console.log(`  ${line.slice(0, 300)}`));
  }

  const total = failures + errors.length;
  console.log(
    total
      ? `\nFAILED — ${failures} screen(s), ${errors.length} error(s)`
      : `\nok — ${SCREENS.length} screens rendered, ${requests.length} requests, no errors`
  );
  process.exit(total ? 1 : 0);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
