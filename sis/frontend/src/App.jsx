/*
 * The shell: header, school strip, nav, and view outlet.
 *
 * Built for a phone first, because that is where most of this console is read. The chrome is
 * three shallow rows rather than one wide one, and each row solves a different problem at
 * 360px:
 *
 *   Row 1  brand, the year picker and the settings button. The year picker is *in* the header
 *          rather than on each screen because every screen is scoped to a year, and an
 *          off-screen year selector means a registrar cannot see which year the four hundred
 *          rows below belong to. It grows to fill the row on a phone and shrinks to its
 *          content from `sm` up. Appearance and language live behind one labelled settings
 *          control, keeping the header compact without hiding the current year.
 *
 *   Row 2  the school strip, and only when there is more than one school. A tab strip with
 *          one tab is chrome that teaches nothing.
 *
 *   Row 3  the nav, as a horizontally scrollable underline nav rather than a collapsed
 *          hamburger. Six destinations behind a toggle is two taps to reach anything and no
 *          sense of where you are; a scrolling strip is one tap and always shows the current
 *          screen. Bootstrap's `flex-nowrap overflow-auto` does the whole thing.
 *
 * The header is deliberately **not** sticky on a phone. A sticky three-row header eats a third
 * of a 360×640 screen before the register starts, and the thing a registrar needs on screen
 * while scrolling a register is the register.
 */
import { useEffect, useState } from 'react';
import { api, getSessionToken } from './api.js';
import { Router } from './router.js';
import { Store } from './store.js';
import { pickName, useResource, useStore } from './hooks.js';
import { Icon, Select, Toasts, cx } from './components/Ui.jsx';
import { Settings } from './components/Settings.jsx';
import { t } from './i18n.js';

/*
 * What each screen needs before it is worth drawing.
 *
 * Keyed by route, including the drill-down screens that are not in the nav, and named after
 * the permission the screen's **own** requests carry. That last part is the rule worth
 * stating: the School screen lists schools through `/v1/schools`, which asks for
 * `structure.read`, so that is what gates it — gating it on `schools.read` would hide a
 * screen the service would happily have answered.
 *
 * The server checks all of this again. What this table decides is whether a person is shown
 * a door that opens.
 */
const ROUTE_PERMISSION = {
  school: 'structure.read',
  year: 'structure.read',
  level: 'structure.read',
  class: 'structure.read',
  student: 'students.read',
  roster: 'students.write',
  studentSetup: 'students.create',
  guardians: 'guardians.read',
  marks: 'grades.read',
  batches: 'imports.run',
  roles: 'roles.assign',
  teacherSetup: 'teachers.assign_subjects',
  gradeAssignments: 'teachers.assign_classes',
  /* Read, not write: a supervisor who may read a register and not record it still has
     somewhere to read it, and the panel decides which of the two they get. */
  attendance: 'attendance.read',
  timetable: 'timetable.read'
};

/* Order is the order of the work: see the school, find a child, put children in classes,
   record who may ask about them, record what they scored, audit what was written. */
const NAV = [
  { name: 'school', label: 'School', icon: 'dashboard', roles: ['system_admin', 'school_owner', 'principal'] },
  { name: 'student', label: 'Find a child', icon: 'search' },
  { name: 'studentSetup', label: 'Student setup', icon: 'people' },
  { name: 'roster', label: 'Roster', icon: 'upload' },
  { name: 'guardians', label: 'Guardians', icon: 'people' },
  { name: 'marks', label: 'Marks', icon: 'marks' },
  { name: 'batches', label: 'Batches', icon: 'batches' },
  { name: 'roles', label: 'Staff roles', icon: 'people', roles: ['system_admin', 'school_owner'] },
  { name: 'teacherSetup', label: 'Teacher setup', icon: 'people' },
  { name: 'gradeAssignments', label: 'Class assignments', icon: 'people' },
  { name: 'attendance', label: 'Take attendance', icon: 'calendar' },
  { name: 'timetable', label: 'Timetable', icon: 'calendar', roles: ['year_supervisor', 'teacher'] }
];

/* Which nav item is lit for a route that is not in the nav. The drill-down screens are
   reached from School, so School stays underlined all the way down — losing the highlight
   four levels deep reads as having left the section.
   Highlighting only: a route's *permission* comes from `ROUTE_PERMISSION` and is its own.
   Reading a requirement off this table instead would refuse a teacher their own class
   screen because they cannot see the school dashboard it is reached from. */
const NAV_PARENT = { year: 'school', level: 'school', class: 'school' };

/*
 * English labels for the role codes the service ships, for the header chip.
 *
 * A fallback rather than a source of truth: `/v1/rbac/roles` serves the real names in both
 * languages, and a screen that lists or assigns roles reads them from there. This table
 * exists so the header can print "Teacher · Attendance Supervisor" on first paint without
 * a second request, and an unknown code falls through to the code itself rather than to a
 * blank — a role added next term shows up as `subject_coordinator` and not as nothing.
 */
const ROLE_LABELS = {
  system_admin: 'System Administrator',
  school_owner: 'School Owner',
  principal: 'School Manager',
  year_supervisor: 'Class Supervisor',
  attendance_supervisor: 'Attendance Supervisor',
  teacher: 'Teacher'
};

/**
 * Sign in with a username and password, and pick up the roles that come with it.
 *
 * **Offered, not imposed**, and that is a decision worth defending. The service has two
 * doors: an integration door that still answers a caller carrying no credential at all,
 * and this one. A console that refused to draw anything until somebody signed in would be
 * stricter than the service behind it — it would hide screens the server would have
 * answered, and the person staring at the sign-in form would have no account to type
 * because none is required. So the shell renders either way, and signing in is what
 * *narrows* the console to one person's roles rather than what unlocks it.
 *
 * Shown as a dialog over the shell rather than as a screen replacing it, for the same
 * reason: the console you were reading is still there behind it.
 */
function SignIn() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  return (
    <main className="sis-login-page">
      <div className="sis-login-card">
        <div className="modal-content border-0">
          <form
            className="modal-body p-4"
            onSubmit={(event) => {
              event.preventDefault();
              setError('');
              setBusy(true);
              api.login(username, password).then(
                (result) => { Store.setAccount(result); setBusy(false); },
                (reason) => { setError(reason.message || t('Sign in failed')); setBusy(false); }
              );
            }}
          >
            <div className="sis-login-brand mb-4">SIS</div>
            <h1 className="h4 mb-1">{t('Sign in')}</h1>
            <p className="small text-body-tertiary">
              {t('Signing in shows you the classes and screens your roles cover.')}
            </p>
            <label className="form-label w-100 mt-3">{t('Username')}
              <input className="form-control" autoComplete="username" value={username}
                onChange={(event) => setUsername(event.target.value)} required />
            </label>
            <label className="form-label w-100 mt-3">{t('Password')}
              <span className="sis-password-field mt-1">
                <input className="form-control sis-password-input" type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password" value={password}
                  onChange={(event) => setPassword(event.target.value)} required />
                <button className="sis-password-toggle" type="button"
                  aria-label={t(showPassword ? 'Hide password' : 'Show password')}
                  title={t(showPassword ? 'Hide password' : 'Show password')}
                  aria-pressed={showPassword}
                  onClick={() => setShowPassword((visible) => !visible)}>
                  <Icon name={showPassword ? 'eyeOff' : 'eye'} size={18} />
                </button>
              </span>
            </label>
            {/* One message for every way a sign-in can fail, because the service answers
                with one — a form that told a wrong password from an unknown username
                would be a way to read a school's staff list. */}
            {error ? (
              <p className="sis-inline-message mt-3 mb-0 small" role="status">
                {t('We could not sign you in. Please check your details and try again.')}
              </p>
            ) : null}
            <div className="d-grid mt-4">
              <button className="btn btn-primary" type="submit" disabled={busy}>
                {busy ? t('Signing in…') : t('Sign in')}
              </button>
            </div>
          </form>
        </div>
      </div>
    </main>
  );
}

/* -- School strip ---------------------------------------------------------------- */

function SchoolTabs() {
  const state = useStore();
  const admin = Store.roles().indexOf('system_admin') >= 0;
  const schools = useResource(Store.keys.schools(false), () => api.schools(false));
  const list = schools.value || [];

  /*
   * Settle on a school as soon as the list arrives: the remembered one if it is still real,
   * else the first. Checked against the list rather than trusted, because a branch can be
   * closed between sessions and a console pinned to it would show empty screens with nothing
   * saying why.
   */
  useEffect(() => {
    if (!list.length) return;
    if (!list.some((school) => school.code === state.school)) {
      Store.setSchool(list[0].code);
    }
  }, [list.length, state.school]);

  if (!admin || list.length < 2) return null;

  return (
    <nav
      className="d-flex flex-nowrap overflow-auto px-3 px-sm-4 py-2 border-bottom"
      style={{ background: 'var(--canvas)', scrollbarWidth: 'none' }}
      aria-label={t('Schools')}
    >
      {/* The track is its own element so the segments have something to sit in: a segmented
          control is a filled strip with one segment lit, and without the wrapper the lit one
          is a white box floating on the page. */}
      <div className="sis-school-strip">
        {list.map((school) => (
          <button
            key={school.code}
            className={cx('sis-school-tab', school.code === state.school && 'active')}
            aria-current={school.code === state.school ? 'true' : undefined}
            onClick={() => {
              Store.setSchool(school.code);
              /* Back to the top of the hierarchy. Staying on a class screen would leave a
                 class code from the previous school in the URL, and the screen would render
                 an error rather than the school just chosen. */
              Router.go('school', { code: school.code });
            }}
          >
            <span className="sis-school-tab-code">{school.code}</span>
            <span className="sis-school-tab-name">{pickName(school, state.lang)}</span>
          </button>
        ))}
      </div>
    </nav>
  );
}

/* -- Year picker ------------------------------------------------------------------ */

function YearPicker() {
  const state = useStore();
  /* This school's years only. Year codes are globally unique, so an unscoped list would offer
     another branch's years in this branch's picker. */
  const years = useResource(
    Store.keys.years(state.school),
    () => api.years(state.school),
    !!state.school
  );
  const list = (years.value && years.value.academic_years) || [];

  useEffect(() => {
    if (!list.length) return;
    if (list.some((year) => year.code === state.year)) return;
    const current = list.find((year) => year.is_current);
    Store.setYear((current || list[list.length - 1]).code);
  }, [list.length, state.year]);

  if (!years.ready && years.loading) {
    return <span className="sis-skel" style={{ width: '9rem', height: '1.9rem' }} />;
  }
  if (!list.length) return null;

  return (
    <label className="d-flex align-items-center gap-2 flex-grow-1 flex-sm-grow-0">
      <span className="small text-body-tertiary text-nowrap d-none d-sm-inline">{t('Year')}</span>
      <Select
        className="sis-code"
        size="sm"
        value={state.year || ''}
        options={list.map((year) => ({
          value: year.code,
          label: year.code + (year.is_current ? ' (current)' : '')
        }))}
        onChange={Store.setYear}
      />
    </label>
  );
}

/* -- Who is signed in ------------------------------------------------------------- */

/**
 * The signed-in person, the roles they hold, and the way out.
 *
 * The roles are shown rather than a single title, and that is the point of the control: a
 * person here is a Teacher *and* an Attendance Supervisor, and a header that picked one to
 * print would be the first place in the product to suggest that roles replace each other.
 * Several badges is the honest rendering of an additive model.
 *
 * Hidden entirely when nobody is signed in, because the service still answers an
 * unauthenticated console and an empty account chip would read as a broken one.
 */
function Account({ onSignIn }) {
  const state = useStore();

  if (!state.profile) {
    return (
      <button className="btn btn-sm btn-outline-secondary text-nowrap" onClick={onSignIn}>
        {t('Sign in')}
      </button>
    );
  }

  const account = state.account || {};
  /* `pickName` already reads `full_name_*`, so the account goes in as it arrived. Falls
     back to the username: an account with no name filled in is still somebody, and a
     blank chip would read as a broken header rather than as a missing field. */
  const name = pickName(account, state.lang) || state.profile.username;
  const held = Store.roles();

  return (
    <div className="d-flex align-items-center gap-2">
      <span className="lh-sm d-none d-lg-block text-end">
        <span className="d-block small fw-semibold text-nowrap">{name}</span>
        <span className="d-block text-body-tertiary sis-role-line">
          {held.length ? held.map((code) => t(ROLE_LABELS[code] || code)).join(' · ') : t('No role')}
        </span>
      </span>
      <button
        className="btn btn-sm btn-quiet"
        title={t('Sign out')}
        onClick={() => {
          api.logout().then(
            () => Store.setAccount(null),
            /* The token is gone from this tab either way — `api.logout` clears it before
               it can fail. Dropping the profile regardless keeps the console from showing
               a signed-in header over a session it can no longer use. */
            () => Store.setAccount(null)
          );
        }}
      >
        <Icon name="signout" />
        <span className="visually-hidden">{t('Sign out')}</span>
      </button>
    </div>
  );
}

/* -- Header ---------------------------------------------------------------------- */

function Header({ onOpenSettings, onSignIn }) {
  const state = useStore();

  return (
    <>
      {state.inflight > 0 ? <div className="sis-progress" aria-hidden="true" /> : null}
      <header
        className="sis-header d-flex flex-wrap align-items-center gap-2 gap-sm-3 px-3 px-sm-4 py-2 border-bottom"
        style={{ background: 'var(--canvas)' }}
      >
        <a
          className="sis-brand text-decoration-none text-body"
          href={Router.href('school')}
        >
          <img
            className="sis-company-mark"
            src="./brand/aurexis-mark.svg"
            width="30"
            height="30"
            alt="Aurexis"
            draggable="false"
          />
          <span className="sis-brand-divider" aria-hidden="true" />
          <span className="sis-brand-mark">SIS</span>
          {/* The product name is the first thing to go on a phone: the badge already says
              which application this is, and the space is worth more than the words. */}
          <span className="d-none d-md-block lh-sm">
            <span className="fw-semibold d-block text-nowrap">{t('Student Information Service')}</span>
            <span className="small text-body-tertiary d-block text-nowrap">
              {t('Registrar console')}
            </span>
          </span>
        </a>

        <div className="d-flex align-items-center gap-2 sis-push">
          <YearPicker />
          <Account onSignIn={onSignIn} />

          {/*
            * One button where there were two.
            *
            * Appearance and language stay in the dialog so both choices have visible labels
            * and the year selector keeps enough room on a phone.
            */}
          <button
            className="btn btn-sm btn-quiet"
            onClick={onOpenSettings}
            title={t('Settings — appearance and language')}
          >
            <Icon name="settings" />
            <span className="visually-hidden">{t('Open settings')}</span>
          </button>
        </div>
      </header>
    </>
  );
}

/* -- Nav ------------------------------------------------------------------------- */

function Nav({ active }) {
  const here = NAV_PARENT[active] || active;
  return (
    <nav
      className="sis-nav border-bottom px-3 px-sm-4"
      aria-label={t('Screens')}
    >
      <ul
        className="nav nav-underline flex-nowrap overflow-auto"
        style={{ scrollbarWidth: 'none' }}
      >
        {NAV.filter((item) => Store.can(ROUTE_PERMISSION[item.name]) &&
          (!item.roles || item.roles.some((role) => Store.roles().indexOf(role) >= 0))).map((item) => {
          const current = here === item.name;
          return (
            <li className="nav-item" key={item.name}>
              <a
                className={cx('nav-link d-flex align-items-center gap-2 text-nowrap', current && 'active')}
                href={Router.href(item.name)}
                aria-current={current ? 'page' : undefined}
              >
                <Icon name={item.icon} />
                {/* The label hides on the narrowest screens and the icon carries it, which is
                    what keeps six destinations reachable without scrolling on a 360px phone. */}
                <span className="d-none d-sm-inline">{t(item.label)}</span>
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

/* -- Footer ---------------------------------------------------------------------- */

function Footer() {
  return (
    <footer className="sis-footer sis-no-print px-3 px-sm-4 py-2 border-top">
      <div className="sis-footer-center">
        <span>Powered by </span>
        <a
          href="https://aurexis.cc/"
          target="_blank"
          rel="noopener noreferrer"
          className="sis-footer-link"
        >
          AUREXIS
        </a>
      </div>
    </footer>
  );
}

/* -- Root ------------------------------------------------------------------------ */

/**
 * The application root. Subscribes to the router once and renders the matched view.
 *
 * The view is keyed by route name so React unmounts the old screen rather than reconciling it
 * with the new one. Two screens with a table in the same position would otherwise reuse those
 * rows and, for one frame, show the marks table filled with roster data — and the entrance
 * animation would not replay, so the transition would only ever work on a first visit.
 */
export function App() {
  const [route, setRoute] = useState(Router.current);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [authReady, setAuthReady] = useState(!getSessionToken());
  /*
   * The root subscribes to the store as well as to the router, and the language is why.
   * `t()` reads a module-level table rather than a hook, which is what keeps it cheap enough to
   * call once per cell; the cost of that is that changing the language does not by itself
   * invalidate anything. Re-rendering from the root does, and nothing below here is memoised,
   * so one subscription re-translates the whole tree.
   */
  useStore();

  useEffect(() => Router.subscribe(setRoute), []);
  /*
   * A reload with a live token restores the session rather than asking again. `/auth/me`
   * answers in the same shape as `/auth/login`, so one setter takes either and the header
   * after a refresh is the header before it. A failure means the token is dead, and the
   * console falls back to the unauthenticated view — never to a half-signed-in one, where
   * the header would name somebody whose permissions had already been forgotten.
   */
  useEffect(() => {
    if (!getSessionToken()) return;
    api.me().then((account) => { Store.setAccount(account); setAuthReady(true); },
      () => { Store.setAccount(null); setAuthReady(true); });
  }, []);

  if (!route) return null;
  /* Only while a stored token is being redeemed. Painting the shell first and correcting
     it a moment later would flash the whole nav and then hide half of it, which reads as
     the console losing screens rather than as it working out who you are. */
  if (!authReady) return null;
  if (!Store.state.profile) return <SignIn />;

  const View = route.route.view;
  /* A screen whose permission this person does not hold is refused here as well as by the
     server. Reachable by typing the URL even when the nav item is hidden, so the check has
     to live on the view and not only on the link. A route with no entry in the table is
     one nothing gates — the sign-in screen, say — and is drawn. */
  const needed = ROUTE_PERMISSION[route.route.name];
  const routeItem = NAV.find((item) => item.name === route.route.name);
  const roleAllowed = !routeItem?.roles || routeItem.roles.some(
    (role) => Store.roles().indexOf(role) >= 0
  );
  const allowed = (!needed || Store.can(needed)) && roleAllowed;

  return (
    <div className="sis-app">
      <Header
        onOpenSettings={() => setSettingsOpen(true)}
        onSignIn={() => {}}
      />
      <SchoolTabs />
      <Nav active={route.route.name} />
      <main className="flex-grow-1 w-100 mx-auto p-3 p-sm-4" style={{ maxWidth: '96rem' }}>
        <div className="sis-rise" key={route.route.name}>
          {allowed ? (
            <View params={route.params} />
          ) : (
            <div className="alert alert-warning">
              <strong>{t('Not your screen')}</strong>
              <div className="small mt-1">
                {t('Your roles do not cover this part of the console. Ask whoever manages roles at your school if you need it.')}
              </div>
            </div>
          )}
        </div>
      </main>
      <Footer />
      <Toasts />
      {/* Rendered last so its backdrop lies over the whole shell — including the header the
          button that opened it sits in. */}
      {settingsOpen ? <Settings onClose={() => setSettingsOpen(false)} /> : null}
    </div>
  );
}
