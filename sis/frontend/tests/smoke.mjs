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

function newWindow(script, language = 'en', session = '') {
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
  const bodies = [];

  window.fetch = (input, init = {}) => {
    const url = String(typeof input === 'string' ? input : input.url);
    const method = (init.method || 'GET').toUpperCase();
    requests.push(`${method} ${url}`);
    /* The body as well as the line. `PATCH /v1/students/10432` looks identical whether the
       console sent `{full_name_ar: "…"}` or the confirmation dialog's own array of
       `{label, was, now}` — and only one of those is a request the service accepts. */
    if (init.body !== undefined && init.body !== null) {
      bodies.push({ line: `${method} ${url}`, body: String(init.body) });
    }
    let payload = answer(method, url);
    if (session === 'smoke-admin' && method === 'GET' && url.includes('/v1/auth/me')) {
      payload = JSON.parse(JSON.stringify(payload));
      const permissions = [
        'structure.read', 'students.read', 'students.create', 'students.write', 'guardians.read',
        'grades.read', 'grades.write', 'imports.run', 'roles.assign', 'users.read',
        'teachers.read', 'teachers.assign_subjects', 'teachers.assign_classes',
        'attendance.read', 'attendance.write', 'timetable.read', 'timetable.write',
        'chat.read', 'chat.write'
      ];
      payload.profile.is_system_admin = true;
      payload.profile.roles = [
        { role_code: 'admin', scope_type: 'global', scope_id: null },
        { role_code: 'year_supervisor', scope_type: 'global', scope_id: null }
      ];
      payload.profile.permissions = permissions;
      payload.profile.grants = permissions.map((permission) => ({
        permission, scope_type: 'global', scope_id: null, scope_code: null
      }));
    }
    if (session === 'smoke-manager-chat' && method === 'GET' && url.includes('/v1/auth/me')) {
      payload = JSON.parse(JSON.stringify(payload));
      payload.profile.is_system_admin = false;
      payload.profile.roles = [
        { role_code: 'school_manager', scope_type: 'school', scope_id: 1 }
      ];
      /* Simulate a still-valid session issued before chat.write was added to auth/me.
         Chat keeps the composer visible for the manager; the API remains authoritative. */
      payload.profile.permissions = ['structure.read', 'chat.read'];
      payload.profile.grants = [
        { permission: 'structure.read', scope_type: 'school', scope_id: 1, scope_code: 'MAIN' },
        { permission: 'chat.read', scope_type: 'school', scope_id: 1, scope_code: 'MAIN' }
      ];
      payload.profile.overrides = [];
    }
    if (session === 'smoke-supervisor' && method === 'GET' && url.includes('/v1/auth/me')) {
      payload = JSON.parse(JSON.stringify(payload));
      payload.profile.roles = [
        { role_code: 'year_supervisor', scope_type: 'year_level', scope_id: 3 }
      ];
      payload.profile.permissions = ['structure.read', 'students.read'];
      payload.profile.grants = [
        { permission: 'structure.read', scope_type: 'year_level', scope_id: 3, scope_code: 'Y3' },
        { permission: 'students.read', scope_type: 'year_level', scope_id: 3, scope_code: 'Y3' }
      ];
    }
    if (session === 'smoke-viewer' && method === 'GET' && url.includes('/v1/auth/me')) {
      payload = JSON.parse(JSON.stringify(payload));
      payload.profile.roles = [
        { role_code: 'school_owner', scope_type: 'school', scope_id: 1 }
      ];
      payload.profile.permissions = ['structure.read', 'timetable.read'];
      payload.profile.grants = [
        { permission: 'structure.read', scope_type: 'school', scope_id: 1, scope_code: 'MAIN' },
        { permission: 'timetable.read', scope_type: 'school', scope_id: 1, scope_code: 'MAIN' }
      ];
    }
    const body = JSON.stringify(payload);
    return Promise.resolve({
      ok: true,
      status: 200,
      headers: { get: (name) => (name.toLowerCase() === 'content-type' ? 'application/json' : null) },
      blob: () => Promise.resolve(new window.Blob([body], { type: 'application/pdf' })),
      json: () => Promise.resolve(JSON.parse(body)),
      text: () => Promise.resolve(body)
    });
  };
  window.FormData = class FormData {
    append() {}
  };
  window.URL.createObjectURL = () => 'blob:http://localhost/mock-attachment';
  window.URL.revokeObjectURL = () => {};
  /* jsdom has no layout, so scrolling is unimplemented and throws through the virtual console.
     The router scrolls to the top on every route change, which would otherwise report thirteen
     errors for correct behaviour. `scrollIntoView` is worse than unimplemented — it is absent
     from jsdom's Element entirely, so the register's "scroll to the panel you just opened"
     effect throws a TypeError and React unmounts the screen. A browser has it; the harness
     must too, or this file cannot test any control that opens a panel. */
  window.scrollTo = () => {};
  window.Element.prototype.scrollIntoView = () => {};

  /* React reports a render that threw, and several real prop faults, through console.error.
     Collected rather than printed so a screen that renders *something* while logging an error
     still fails the test. */
  window.console.error = (...args) => errors.push(args.map(String).join(' '));
  window.console.warn = () => {};

  window.localStorage.setItem('sis.lang', language);
  /* A stored session token makes the console redeem it through `/v1/auth/me` on boot and
     come up as that person, which is the only way to reach the permission-dependent
     branches from here. The value is never checked — the fixture answers regardless — so
     any non-empty string does. */
  if (session) window.sessionStorage.setItem('sis.session_token.v2', session);
  window.eval(script);
  return { dom, window, errors, requests, bodies };
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
  { hash: '#/studentSetup', expect: ['Create student', 'Create student and guardian'] },
  { hash: '#/guardians', expect: ['Guardians'] },
  { hash: '#/marks', expect: ['Marks'] },
  { hash: '#/batches', expect: ['Batches'] },
  /* A multi-role teacher/principal configuring eligibility before the grade supervisor
     assigns concrete rooms. This covers the Stage 15 manager handoff in the real router. */
  { hash: '#/teacherSetup', expect: ['Create teacher', 'Subject, grade, and track eligibility'] },
  /* The grade supervisor's screen, walked as somebody holding no year-level grant: it
     renders its own empty state rather than throwing, which is the branch every other
     visitor to this route takes. The populated flow needs a profile with a `year_level`
     grant, and `GET /v1/auth/me` has one fixture for the whole walk, so it is not
     reachable from here — the Python suite covers that half against the real service. */
  { hash: '#/gradeAssignments', expect: ['Class assignments', 'No managed grades'] },
  /* The register workflow, from its own first step: the day/grade/class pickers and the
     panel underneath, driven by the classes fixture rather than by navigating a structure
     an attendance supervisor cannot read. */
  { hash: '#/attendance', expect: ['Take attendance', 'Year 3', '3A'] },
  { hash: '#/chat', expect: ['Chats', 'All school staff', 'Ahmed Hassan', 'Grade groups', 'Class groups'] },
  { hash: '#/timetable', expect: ['Timetable', 'Weekly timetable', 'Mathematics'] }
];

async function main() {
  process.stdout.write('bundling… ');
  const script = await bundle();
  process.stdout.write(`${(script.length / 1024).toFixed(0)} KB\n`);

  const signedOut = newWindow(script);
  await settle(signedOut.window, 100);
  assert.ok(signedOut.window.document.querySelector('.sis-login-page'), 'signed-out visitors must see the sign-in page');
  assert.ok(!signedOut.window.document.querySelector('.sis-app'), 'the SIS shell must stay hidden before sign-in');

  const { window, errors, requests, bodies } = newWindow(script, 'en', 'smoke-admin');
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

  window.location.hash = '#/school';
  await settle(window, 140);
  const schoolRungs = [...window.document.querySelectorAll('.sis-rung')];
  assert.ok(schoolRungs.length > 0, 'the school ladder rendered no grade rows');
  schoolRungs.forEach((rung) => {
    assert.ok(!rung.querySelector('.sis-row-actions'), 'an existing grade still exposes structure-changing actions');
    assert.ok(!rung.textContent.includes('Remove'), 'an existing grade still offers the unsupported remove action');
  });

  const settingsButton = window.document.querySelector('button[title="Settings — appearance and language"]');
  assert.ok(settingsButton, 'the shell has no settings trigger');
  settingsButton.click();
  await settle(window, 60);
  const settingsCard = window.document.querySelector('.sis-settings-card');
  assert.ok(settingsCard, 'the settings trigger did not open the preferences card');
  assert.ok(settingsCard.textContent.includes('Preferences'), 'the mobile settings card has no preferences heading');
  assert.equal(window.document.body.style.overflow, 'hidden', 'the page can scroll behind the settings dialog');
  settingsCard.querySelector('.sis-settings-close').click();
  await settle(window, 40);
  assert.ok(!window.document.querySelector('.sis-settings-modal'), 'closing settings left the dialog open');
  assert.equal(window.document.body.style.overflow, '', 'closing settings left page scrolling locked');

  assert.ok(!window.document.querySelector('.sis-mobile-dock'), 'the removed mobile bottom bar is still rendered');
  assert.ok(
    !window.document.querySelector('.sis-nav-categories'),
    'the category strip is shown to an admin instead of only the school manager'
  );
  const menuButton = window.document.querySelector('.sis-header button[title="Menu"]');
  assert.ok(menuButton, 'the mobile menu trigger is missing from the header');
  menuButton.click();
  await settle(window, 60);
  const mobileIdentity = window.document.querySelector('.sis-mobile-drawer-identity');
  assert.ok(mobileIdentity, 'the mobile navigation drawer did not open');
  assert.ok(mobileIdentity.querySelector('small').textContent.includes('Admin'), 'the drawer does not show the account role below the name');
  assert.ok(!mobileIdentity.querySelector('small').textContent.includes('Student Information Service'), 'the drawer still shows product copy below the account name');
  assert.ok(!window.document.querySelector('.sis-mobile-message-item'), 'the drawer still duplicates the floating chats trigger');
  assert.ok(!window.document.querySelector('.sis-mobile-section-title'), 'a non-manager still sees category sections in the mobile menu');
  assert.ok(window.document.querySelector('.sis-mobile-nav-list-flat'), 'a non-manager is missing the direct mobile navigation list');
  window.document.querySelector('.sis-mobile-drawer-close').click();
  await settle(window, 40);

  window.location.hash = '#/chat';
  await settle(window, 140);
  assert.ok(window.document.querySelector('.sis-app-chat'), 'chat did not activate its viewport-bound app shell');
  assert.ok(window.document.querySelector('.sis-main-chat'), 'chat main region is not internally constrained');
  assert.ok(!window.document.querySelector('.sis-chat-shell.has-active-chat'), 'opening chats selected a conversation without user input');
  assert.ok(!window.document.querySelector('.sis-chat-compose textarea'), 'opening chats jumped straight into a conversation');
  window.document.querySelector('.sis-chat-conversation').click();
  await settle(window, 100);
  assert.ok(window.document.querySelector('.sis-chat-compose textarea'), 'the active chat has no visible message composer');
  assert.ok(window.document.querySelector('.sis-chat-messages'), 'the active chat has no internal message scroller');
  const attachedFile = window.document.querySelector('.sis-chat-file');
  assert.ok(attachedFile, 'the chat fixture has no attached file preview trigger');
  attachedFile.click();
  await settle(window, 80);
  const attachmentPreview = window.document.querySelector('.sis-attachment-preview');
  assert.ok(attachmentPreview, 'clicking an attached file downloaded it instead of opening the preview');
  assert.ok(attachmentPreview.querySelector('iframe'), 'a PDF attachment did not open inside the preview');
  assert.ok(attachmentPreview.querySelector('.sis-preview-download'), 'the attachment preview has no separate download action');
  attachmentPreview.querySelector('.sis-preview-close').click();
  await settle(window, 40);
  const attachedImage = window.document.querySelector('.sis-chat-image');
  assert.ok(attachedImage, 'the chat fixture has no image attachment');
  attachedImage.click();
  await settle(window, 80);
  const imagePreview = window.document.querySelector('.sis-attachment-preview-backdrop');
  assert.ok(imagePreview?.querySelector('.sis-attachment-preview-body img'), 'clicking an image did not open it in the preview');
  /* The route and every message run transform animations, which make them the containing
     block for position: fixed. Rendered inside a bubble, the overlay shrank to the bubble. */
  assert.equal(imagePreview.parentElement, window.document.body, 'the image preview is trapped inside the message instead of covering the page');
  imagePreview.querySelector('.sis-preview-close').click();
  await settle(window, 40);
  assert.ok(!window.document.querySelector('.sis-attachment-preview'), 'closing the image preview left it open');
  const groupReceipt = window.document.querySelector('.sis-chat-receipts.has-count');
  assert.ok(groupReceipt, 'a sent group message has no discoverable member receipt control');
  assert.ok(groupReceipt.textContent.includes('2/4'), 'the group receipt control does not show read members out of total members');
  groupReceipt.click();
  await settle(window, 80);
  const receiptPanel = window.document.querySelector('.sis-chat-receipt-panel');
  assert.ok(receiptPanel, 'clicking the group receipt control did not open message info');
  assert.ok(receiptPanel.textContent.includes('Read by'), 'message info does not separate members who read the message');
  assert.ok(receiptPanel.textContent.includes('Delivered to'), 'message info does not separate delivered unread members');
  assert.ok(receiptPanel.textContent.includes('Not delivered yet'), 'message info does not separate undelivered members');
  receiptPanel.querySelector('header button').click();
  await settle(window, 40);
  assert.ok(window.document.querySelector('.sis-chat-thread-head .sis-chat-typing'), 'a typing group member is not shown in the conversation header');
  const onlineSender = [...window.document.querySelectorAll('.sis-chat-sender')].find((button) => !button.disabled);
  assert.ok(onlineSender?.querySelector('.sis-chat-presence-dot.is-online'), 'an online sender has no green presence dot');
  onlineSender.click();
  await settle(window, 60);
  const staffProfile = window.document.querySelector('.sis-chat-profile');
  assert.ok(staffProfile?.textContent.includes('Mathematics teacher · Grade 3'), 'clicking a sender did not open their staff profile');
  assert.equal(staffProfile.parentElement.parentElement, window.document.body, 'the staff profile is trapped inside the chat pane');
  staffProfile.querySelector('.sis-chat-profile-close').click();
  await settle(window, 40);
  const composerTool = window.document.querySelector('.sis-chat-compose .sis-chat-tool');
  composerTool.focus();
  composerTool.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true }));
  await settle(window, 30);
  assert.equal(window.document.activeElement, window.document.querySelector('.sis-chat-compose textarea'), 'Enter repeated the last composer action instead of returning to message sending');
  const conversationButtons = [...window.document.querySelectorAll('.sis-chat-conversation')];
  assert.equal(conversationButtons.length, 6, 'the chat fixture must cover every conversation category');
  for (const conversationButton of conversationButtons) {
    conversationButton.click();
    await settle(window, 80);
    assert.ok(
      window.document.querySelector('.sis-chat-compose textarea'),
      `the composer disappeared after switching to ${conversationButton.textContent.trim()}`
    );
  }

  const managerChat = newWindow(script, 'ar', 'smoke-manager-chat');
  managerChat.window.location.hash = '#/chat';
  await settle(managerChat.window, 180);
  assert.ok(
    managerChat.window.document.querySelector('.sis-nav-categories'),
    'the school manager is missing the category strip'
  );
  assert.ok(managerChat.window.document.querySelector('.sis-chat-conversations'), 'the manager has no group list scroller');
  const managerConversations = [...managerChat.window.document.querySelectorAll('.sis-chat-conversation')];
  assert.equal(managerConversations.length, 6, 'the manager did not receive every chat category');
  for (const conversationButton of managerConversations) {
    conversationButton.click();
    await settle(managerChat.window, 80);
    assert.ok(
      managerChat.window.document.querySelector('.sis-chat-compose textarea'),
      `the manager composer disappeared after switching to ${conversationButton.textContent.trim()}`
    );
  }
  errors.push(...managerChat.errors);

  window.location.hash = '#/timetable';
  await settle(window, 140);

  const timetableDayNav = window.document.querySelector('.sis-timetable-day-nav');
  assert.ok(timetableDayNav, 'the timetable has no mobile day navigation');
  const initialMobileDay = window.document.querySelector('.sis-timetable-grid thead .is-mobile-day-active')?.dataset.day;
  assert.ok(initialMobileDay, 'the timetable did not select an initial mobile day');
  const nextDayButton = timetableDayNav.querySelector('button[aria-label="Next day"]');
  const previousDayButton = timetableDayNav.querySelector('button[aria-label="Previous day"]');
  assert.ok(nextDayButton && previousDayButton, 'the mobile timetable is missing previous/next day buttons');
  nextDayButton.click();
  await settle(window, 40);
  const advancedMobileDay = window.document.querySelector('.sis-timetable-grid thead .is-mobile-day-active')?.dataset.day;
  assert.notEqual(advancedMobileDay, initialMobileDay, 'the next-day button did not advance the mobile timetable');
  previousDayButton.click();
  await settle(window, 40);
  assert.equal(
    window.document.querySelector('.sis-timetable-grid thead .is-mobile-day-active')?.dataset.day,
    initialMobileDay,
    'the previous-day button did not return to the prior mobile timetable day'
  );

  /* Timetable edits stay in the browser until the authorised user presses Save. The save
     then sends additions and removals together to the transactional endpoint. */
  const saveTimetable = [...window.document.querySelectorAll('button')].find(
    (button) => button.textContent.trim() === 'Save timetable'
  );
  assert.ok(saveTimetable, 'an authorised timetable editor has no Save button');
  assert.ok(saveTimetable.disabled, 'Save timetable must start disabled with no changes');
  const subjectChip = [...window.document.querySelectorAll('.sis-subject-chip')].find(
    (button) => button.textContent.includes('Mathematics')
  );
  const emptySlot = [...window.document.querySelectorAll('.sis-timetable-slot')].find(
    (cell) => cell.querySelector('.sis-empty-slot')
  );
  assert.ok(subjectChip && emptySlot, 'the timetable fixture has no editable empty slot');
  const originalLesson = window.document.querySelector('.sis-timetable-slot .sis-lesson');
  const originalSlot = originalLesson?.closest('.sis-timetable-slot');
  assert.ok(originalLesson && originalSlot, 'the timetable fixture has no lesson to swap');
  originalLesson.click();
  await settle(window, 30);
  assert.ok(
    emptySlot.classList.contains('is-drop-ready'),
    'tapping an existing lesson did not select it for a touch-friendly swap'
  );
  emptySlot.click();
  await settle(window, 40);
  assert.ok(originalSlot.querySelector('.sis-empty-slot'), 'moving a lesson did not clear its old slot');
  assert.ok(emptySlot.querySelector('.sis-lesson'), 'moving a lesson did not fill its target slot');
  emptySlot.querySelector('.sis-lesson').click();
  await settle(window, 30);
  originalSlot.click();
  await settle(window, 40);
  assert.ok(originalSlot.querySelector('.sis-lesson'), 'swapping the lesson back did not restore its original slot');
  assert.ok(emptySlot.querySelector('.sis-empty-slot'), 'swapping the lesson back did not clear the temporary slot');
  assert.ok(saveTimetable.disabled, 'a round-trip lesson swap left a false unsaved change');
  const writesBeforeEdit = requests.filter((line) => line.startsWith('PUT /v1/timetable')).length;
  subjectChip.click();
  await settle(window, 40);
  emptySlot.click();
  await settle(window, 60);
  assert.equal(
    requests.filter((line) => line.startsWith('PUT /v1/timetable')).length,
    writesBeforeEdit,
    'editing a timetable cell wrote to the service before Save was pressed'
  );
  assert.ok(!saveTimetable.disabled, 'editing a timetable cell did not enable Save');
  saveTimetable.click();
  await settle(window, 100);
  const timetableWrite = bodies.find((entry) => entry.line === 'PUT /v1/timetable/week');
  assert.ok(timetableWrite, `Save timetable sent no database request. Saw: ${bodies.map((e) => e.line)}`);
  const timetableBody = JSON.parse(timetableWrite.body);
  assert.equal(timetableBody.academic_year_code, YEAR);
  assert.equal(timetableBody.entries.length, 1);
  assert.equal(timetableBody.entries[0].subject_code, 'MATH');
  assert.deepEqual(timetableBody.clear_slots, []);

  const timetableViewer = newWindow(script, 'en', 'smoke-viewer');
  await settle(timetableViewer.window, 180);
  timetableViewer.window.location.hash = '#/timetable';
  await settle(timetableViewer.window, 140);
  assert.ok(
    timetableViewer.window.document.body.textContent.includes('View only'),
    'a read-only timetable user was not shown the read-only state'
  );
  assert.ok(
    ![...timetableViewer.window.document.querySelectorAll('button')].some(
      (button) => button.textContent.trim() === 'Save timetable'
    ),
    'a read-only timetable user was shown the Save button'
  );
  errors.push(...timetableViewer.errors);

  window.location.hash = '#/teacherSetup';
  await settle(window, 120);
  const roleSelect = [...window.document.querySelectorAll('[aria-haspopup="listbox"]')].find((button) =>
    button.textContent.trim() === 'Teacher');
  assert.ok(roleSelect, 'staff creation must offer supervisor roles');
  for (const role of ['Floor supervisor', 'Attendance supervisor']) {
    roleSelect.click();
    await settle(window, 40);
    const option = [...window.document.querySelectorAll('[role="option"]')].find((item) => item.textContent.trim() === role);
    assert.ok(option, `missing role ${role}`);
    option.click();
    await settle(window, 80);
    assert.ok(window.document.body.textContent.includes('Supervision grade'));
    assert.ok(!window.document.body.textContent.includes('Subject, grade, and track eligibility'));
    assert.ok(window.document.body.textContent.includes('Create supervisor account'));
  }

  /* The finder must do more than render. Submit a real value and pin the request's
     academic-year scope; a screen-only smoke check missed the regression where every
     scoped account received 403 from an otherwise healthy search endpoint. */
  window.location.hash = '#/student';
  await settle(window, 100);
  const searchInput = window.document.querySelector('.sis-field-search input');
  assert.ok(searchInput, 'the student finder has no search input');
  const valueSetter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype,
    'value'
  ).set;
  valueSetter.call(searchInput, 'Layla');
  searchInput.dispatchEvent(new window.Event('input', { bubbles: true }));
  searchInput.closest('form').dispatchEvent(
    new window.Event('submit', { bubbles: true, cancelable: true })
  );
  await settle(window, 120);
  assert.ok(
    requests.some((line) =>
      line.includes(`/v1/students?q=Layla&academic_year=${encodeURIComponent(YEAR)}`)
    ),
    `student search did not carry its year scope. Saw: ${requests.filter((line) => line.includes('/v1/students'))}`
  );
  assert.ok(
    (window.document.body.textContent || '').includes('Layla Hassan'),
    'submitting the student finder did not render its result'
  );

  /*
   * The register's own three buttons, driven the way a registrar drives them.
   *
   * Rendering the screen is not enough here and never was. Edit, Move and Remove all reported
   * success and changed nothing: the editor sent the confirmation dialog's diff — a JSON
   * *array* of `{label, was, now}` — as the PATCH body, which the service answers 422 to, and
   * rendered the same objects where a value belongs, which React refuses to draw at all. The
   * screen went blank, the toast said "updated", and the name was unchanged. So this walks the
   * form: open it, type, confirm, and read what actually went over the wire.
   */
  window.location.hash = `#/class?code=3A&year=${YEAR}`;
  await settle(window, 150);

  const openRows = [...window.document.querySelectorAll('tbody tr')];
  const editButton = openRows
    .flatMap((row) => [...row.querySelectorAll('.sis-row-actions button')])
    .find((button) => button.textContent.trim() === 'Edit');
  assert.ok(editButton, 'the register has no Edit button');
  editButton.click();
  await settle(window, 150);

  const nameInput = [...window.document.querySelectorAll('.card input')].find(
    (input) => input.value === 'Layla Hassan'
  );
  assert.ok(nameInput, 'the Edit panel did not load her record into the form');
  valueSetter.call(nameInput, 'Layla Hassan Ali');
  nameInput.dispatchEvent(new window.Event('input', { bubbles: true }));
  await settle(window, 60);
  nameInput.closest('form').dispatchEvent(
    new window.Event('submit', { bubbles: true, cancelable: true })
  );
  await settle(window, 100);

  const modal = window.document.querySelector('.modal .sis-diff');
  assert.ok(modal, 'submitting the edit did not open its confirmation');
  const shown = modal.textContent || '';
  assert.ok(
    shown.includes('Name (English)') && shown.includes('Layla Hassan Ali'),
    `the diff must name the field and both values. Read: ${shown.replace(/\s+/g, ' ')}`
  );
  assert.ok(
    !shown.includes('object Object') && !shown.includes('"label"'),
    `the diff rendered its own descriptor objects instead of the values: ${shown}`
  );

  const confirmButton = [...window.document.querySelectorAll('.modal-footer button')].find(
    (button) => button.textContent.includes('Save the change')
  );
  assert.ok(confirmButton, 'the confirmation has no confirm button');
  confirmButton.click();
  await settle(window, 150);

  const patch = bodies.find((entry) => /PATCH .*\/v1\/students\/10432/.test(entry.line));
  assert.ok(patch, `no PATCH reached the service. Saw: ${bodies.map((e) => e.line)}`);
  const sent = JSON.parse(patch.body);
  assert.ok(
    sent && !Array.isArray(sent) && typeof sent === 'object',
    `the PATCH body must be the changed fields, not the dialog's diff. Sent: ${patch.body}`
  );
  assert.deepEqual(
    sent,
    { full_name_en: 'Layla Hassan Ali' },
    `only the field that changed may be sent. Sent: ${patch.body}`
  );

  /*
   * And the other half of the same complaint: Remove used to say "removed" and leave the
   * child on the register anyway, correctly — a placement ends on her LAST day, and that
   * day's attendance is taken against that day's register. Ending a placement on today's
   * date now files that attendance itself, in the same call, so there is nothing left for
   * this screen to hold her for: `10434` is closed in the fixture and must not be drawn at
   * all, not even with an explanation.
   */
  const registerText = window.document.body.textContent || '';
  assert.ok(
    !registerText.includes('Nour Adel'),
    'a child whose placement has closed is still drawn on the register'
  );
  assert.ok(
    !registerText.includes('off the register after it') && !registerText.includes('on their last day'),
    'the retired "last day" copy is still on the register screen'
  );
  const removeButtons = [...window.document.querySelectorAll('.sis-row-actions button')].filter(
    (button) => button.textContent.trim() === 'Remove'
  );
  assert.equal(
    removeButtons.length,
    2,
    'Remove must be offered on exactly the two open placements, and on no others'
  );

  /*
   * Move, driven the same way, because "moved" and "still in the old class" was the third
   * report and the one that could only be checked from the console. The service's half is
   * proven in `test_registrar_edits_api.py`; what could not be proven anywhere is that the
   * panel this button opens reaches it at all — with the class the registrar picked, in the
   * year the screen is on.
   */
  const moveButton = [...window.document.querySelectorAll('.sis-row-actions button')].find(
    (button) => button.textContent.trim() === 'Move'
  );
  assert.ok(moveButton, 'the register has no Move button');
  moveButton.click();
  await settle(window, 150);

  const movePanel = [...window.document.querySelectorAll('.card')].find((card) => {
    const text = card.textContent || '';
    return text.includes('Move') && text.includes('Transfer date');
  });
  assert.ok(movePanel, 'clicking Move opened no panel');
  const picker = movePanel.querySelector('.sis-field-trigger');
  assert.ok(picker, 'the Move panel has no class picker');
  picker.click();
  await settle(window, 60);

  /* Same grade only, so 3A's own row is absent and 3B is the whole list. A picker that
     offered 3A back would be offering a transfer the service refuses. */
  const options = [...movePanel.querySelectorAll('[role="option"]')].map((node) =>
    node.textContent.trim()
  );
  assert.ok(
    options.length === 1 && options[0].startsWith('3B'),
    `the Move picker must offer the other section of this grade and nothing else. Saw: ${JSON.stringify(options)}`
  );
  [...movePanel.querySelectorAll('[role="option"]')][0].click();
  await settle(window, 80);

  movePanel.querySelector('form').dispatchEvent(
    new window.Event('submit', { bubbles: true, cancelable: true })
  );
  await settle(window, 100);
  const moveConfirm = [...movePanel.querySelectorAll('button')].find(
    (button) => button.textContent.trim() === 'Confirm transfer'
  );
  assert.ok(moveConfirm, 'submitting the move did not open its confirmation');
  moveConfirm.click();
  await settle(window, 150);

  const transfer = bodies.find((entry) => entry.line.includes('/v1/students/10432/transfer'));
  assert.ok(transfer, `the move sent no transfer. Saw: ${bodies.map((e) => e.line)}`);
  const moved = JSON.parse(transfer.body);
  assert.equal(moved.to_class_code, '3B', `the picked class did not reach the service: ${transfer.body}`);
  assert.equal(moved.academic_year_code, YEAR, `the move left its year behind: ${transfer.body}`);
  assert.ok(moved.on_date, `a transfer must carry the day it happens on: ${transfer.body}`);

  /* And Remove, which used to report success while leaving the child on screen "for one
     more day, correctly" — a delay a registrar reads as a broken button. It closes her
     placement and files today's absence in the same call now, so what is asserted here is
     that the confirmation says so plainly rather than explaining a wait that no longer
     happens. */
  const removeButton = [...window.document.querySelectorAll('.sis-row-actions button')].find(
    (button) => button.textContent.trim() === 'Remove'
  );
  assert.ok(removeButton, 'the register has no Remove button');
  removeButton.click();
  await settle(window, 100);
  const removeConfirm = [...window.document.querySelectorAll('.modal-footer button')].find(
    (button) => button.textContent.includes('Remove from the class')
  );
  assert.ok(removeConfirm, 'Remove opened no confirmation');
  assert.ok(
    (window.document.querySelector('.modal-body').textContent || '').includes('recorded absent'),
    'the confirmation no longer says today is filed as an absence automatically'
  );
  removeConfirm.click();
  await settle(window, 150);

  const ended = bodies.find((entry) => entry.line.includes('/placements/current'));
  assert.ok(ended, `Remove sent nothing. Saw: ${bodies.map((e) => e.line)}`);
  assert.ok(
    /^\d{4}-\d{2}-\d{2}$/.test(JSON.parse(ended.body).ends_on),
    `a placement is ended on a date, and this one was not: ${ended.body}`
  );

  const supervisor = newWindow(script, 'en', 'smoke-supervisor');
  await settle(supervisor.window, 180);
  supervisor.window.location.hash = '#/student';
  await settle(supervisor.window, 100);
  const supervisorInput = supervisor.window.document.querySelector('.sis-field-search input');
  assert.ok(supervisorInput, 'a grade supervisor was sent to the teacher class picker');
  valueSetter.call(supervisorInput, 'Layla');
  supervisorInput.dispatchEvent(new supervisor.window.Event('input', { bubbles: true }));
  supervisorInput.closest('form').dispatchEvent(
    new supervisor.window.Event('submit', { bubbles: true, cancelable: true })
  );
  await settle(supervisor.window, 120);
  assert.ok(
    supervisor.requests.some((line) =>
      line.includes(`academic_year=${encodeURIComponent(YEAR)}&year_level=Y3`)
    ),
    `grade-supervisor search did not carry its grade scope. Saw: ${supervisor.requests}`
  );
  errors.push(...supervisor.errors);

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
  const arabic = newWindow(script, 'ar', 'smoke-admin');
  await settle(arabic.window, 200);
  assert.equal(arabic.window.document.documentElement.lang, 'ar');
  assert.equal(arabic.window.document.documentElement.dir, 'rtl');
  const arabicText = arabic.window.document.body.textContent || '';
  assert.ok(arabicText.includes('نظام معلومات الطلاب'), 'Arabic shell title did not render');
  assert.ok(arabicText.includes('المدرسة'), 'Arabic navigation did not render');
  errors.push(...arabic.errors);

  /*
   * A third shell, signed in as somebody whose grants reach exactly one classroom.
   *
   * `GET /v1/auth/me` in the fixtures answers as one person holding **two** roles at once —
   * Teacher and Attendance Supervisor — both bounded to class `3A`. That shape is the
   * point. It is what the console has to render without picking one role to believe, and
   * it is what makes the two halves of the check separable:
   *
   *   the permission union   decides which *screens* exist. This person holds neither
   *                          `students.write` nor `imports.run`, so Roster and Batches
   *                          must be gone from the nav.
   *   the scope on a grant   decides which *controls* are live. `attendance.write` is held
   *                          on 3A and nowhere else, so the register is writable in 3A and
   *                          read-only in 3B — a distinction a permission list alone
   *                          cannot make, and the one every naive console gets wrong.
   *
   * Asserted here because this branch is invisible everywhere else: the Python suite proves
   * the *service* refuses, and nothing but this proves the console stops asking.
   */
  const scoped = newWindow(script, 'en', 'a-session-token');
  await settle(scoped.window, 220);
  const chrome = scoped.window.document.body.textContent || '';
  assert.ok(chrome.includes('Nadia Kamal'), 'the signed-in person is not named in the header');
  /* Both, not the first or the last. A header that printed one title would be the first
     place in the product to suggest that a second role replaces the first. */
  assert.ok(
    chrome.includes('Teacher') && chrome.includes('Attendance Supervisor'),
    `both held roles must be shown. Header read: ${chrome.slice(0, 200).replace(/\s+/g, ' ')}`
  );
  const scopedBrand = scoped.window.document.querySelector('.sis-header .sis-brand');
  assert.equal(
    scopedBrand?.getAttribute('href'),
    '#/attendance',
    'the attendance supervisor brand link does not lead to their permitted home screen'
  );
  scopedBrand.click();
  await settle(scoped.window, 150);
  assert.equal(scoped.window.location.hash, '#/attendance');
  assert.ok(
    !(scoped.window.document.body.textContent || '').includes('Not your screen'),
    'the role-aware brand shortcut still opened an unauthorized screen'
  );
  const class3A = [...scoped.window.document.querySelectorAll('button')].find(
    (button) => button.textContent.trim().startsWith('3A —')
  );
  assert.ok(class3A, 'the attendance supervisor cannot open the 3A register from their home screen');
  class3A.click();
  await settle(scoped.window, 150);
  const unmarkedAttendanceRow = [...scoped.window.document.querySelectorAll('.sis-attendance-register tbody tr')]
    .find((row) => row.textContent.includes('10433'));
  assert.ok(unmarkedAttendanceRow, 'the writable attendance row did not render');
  const excusedButton = unmarkedAttendanceRow.querySelector('button[title="Excused"]');
  assert.ok(excusedButton && !excusedButton.disabled, 'the excused attendance control is not writable');
  for (const stateName of ['present', 'absent', 'late', 'excused']) {
    assert.ok(
      unmarkedAttendanceRow.querySelector(`.sis-attendance-choice.is-${stateName}`),
      `the ${stateName} attendance choice lost its semantic colour class`
    );
  }
  excusedButton.click();
  await settle(scoped.window, 60);
  assert.ok(excusedButton.classList.contains('is-selected'), 'the selected attendance colour is not activated');
  const reasonField = unmarkedAttendanceRow.querySelector('.sis-attendance-note-cell input');
  assert.ok(reasonField, 'choosing Excused did not reveal the full-width reason field');
  reasonField.value = 'Medical appointment';
  reasonField.dispatchEvent(new scoped.window.Event('input', { bubbles: true }));
  await settle(scoped.window, 40);
  assert.equal(reasonField.value, 'Medical appointment', 'the attendance reason field loses typed text');

  const navLabels = [...scoped.window.document.querySelectorAll('nav[aria-label] .nav-link')]
    .map((node) => node.textContent);
  assert.ok(
    navLabels.some((label) => label.includes('Marks')),
    `a screen this teacher may reach was hidden. Saw: ${JSON.stringify(navLabels)}`
  );
  assert.ok(
    !navLabels.some((label) => label.includes('Batches')),
    'Batches needs imports.run, which this teacher does not hold, and was still offered'
  );
  assert.ok(
    !navLabels.some((label) => label.includes('Roster')),
    'Roster needs students.write, which this teacher does not hold, and was still offered'
  );

  scoped.window.location.hash = `#/class?code=3A&year=${YEAR}&tab=attendance`;
  await settle(scoped.window, 150);
  const own = scoped.window.document.body.textContent || '';
  assert.ok(
    !own.includes('You can read this register but not record it'),
    'the teacher of 3A was told 3A is read-only'
  );

  scoped.window.location.hash = `#/class?code=3B&year=${YEAR}&tab=attendance`;
  await settle(scoped.window, 150);
  const other = scoped.window.document.body.textContent || '';
  assert.ok(
    other.includes('Read-only attendance view. Recording controls are hidden for this account.'),
    `the teacher of 3A was offered attendance write controls on 3B. Rendered: ${other.slice(0, 400).replace(/\s+/g, ' ')}`
  );
  errors.push(...scoped.errors);

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
