/*
 * The shell: header, school strip, categorized modern navigation, side chat drawer popup,
 * mobile menu, and view outlet.
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
 * of a 360Ã—640 screen before the register starts, and the thing a registrar needs on screen
 * while scrolling a register is the register.
 */
import { useEffect, useState } from 'react';
import { api, getSessionToken } from './api.js';
import { Router } from './router.js';
import { Store } from './store.js';
import { pickName, useResource, useStore } from './hooks.js';
import { Icon, Select, Toasts, cx } from './components/Ui.jsx';
import { Settings } from './components/Settings.jsx';
import { Chat } from './views/Chat.jsx';
import { t } from './i18n.js';

/*
 * What each screen needs before it is worth drawing.
 *
 * Keyed by route, including the drill-down screens that are not in the nav, and named after
 * the permission the screen's **own** requests carry. That last part is the rule worth
 * stating: the School screen lists schools through `/v1/schools`, which asks for
 * `structure.read`, so that is what gates it â€” gating it on `schools.read` would hide a
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
  promotions: 'students.write',
  studentSetup: 'students.create',
  guardians: 'guardians.read',
  homework: 'grades.read',
  marks: 'grades.read',
  batches: 'imports.run',
  roles: 'roles.assign',
  teacherSetup: 'teachers.assign_subjects',
  teachingStaff: 'teachers.read',
  gradeAssignments: 'teachers.assign_classes',
  chat: 'chat.read',
  /* Read, not write: a supervisor who may read a register and not record it still has
     somewhere to read it, and the panel decides which of the two they get. */
  attendance: 'attendance.read',
  timetable: 'timetable.read',
  auditLog: 'audit.read'
};

function principalExperience() {
  const held = Store.roles();
  return held.indexOf('school_manager') >= 0 && held.indexOf('admin') < 0 && held.indexOf('school_owner') < 0;
}
/* Order is the order of the work: see the school, find a child, put children in classes,
   record who may ask about them, record what they scored, audit what was written.
   Chats is not in this list: chat opens as a side drawer from the header or the floating
   button, so a separate navigation entry would duplicate those entry points. */
const NAV = [
  { name: 'school', label: 'School', icon: 'school', roles: ['admin', 'school_owner', 'school_manager'], category: 'school' },
  { name: 'student', label: 'Find a child', icon: 'search', category: 'students' },
  { name: 'studentSetup', label: 'Create student', icon: 'studentAdd', category: 'students' },
  { name: 'roster', label: 'Roster', icon: 'roster', category: 'students' },
  { name: 'promotions', label: 'Student promotion', icon: 'classAssign', roles: ['school_manager'], category: 'students' },
  { name: 'batches', label: 'Batches', icon: 'batches', roles: ['admin', 'school_owner'], category: 'students' },
  { name: 'roles', label: 'Staff roles', icon: 'roles', roles: ['admin', 'school_owner', 'school_manager'], category: 'school' },
  { name: 'auditLog', label: 'Audit Log', icon: 'roles', roles: ['admin', 'school_manager', 'floor_supervisor', 'attendance_supervisor'], category: 'school' },
  { name: 'teacherSetup', label: 'Create teacher', icon: 'teacher', roles: ['admin', 'school_manager'], category: 'school' },
  { name: 'teachingStaff', label: 'Teaching staff', icon: 'staff', roles: ['school_manager', 'school_owner'], category: 'school' },
  { name: 'gradeAssignments', label: 'Class assignments', icon: 'classAssign', roles: ['admin', 'school_manager', 'floor_supervisor'], category: 'school' },
  { name: 'homework', label: 'Homework', icon: 'upload', roles: ['teacher'], category: 'academic' },
  { name: 'marks', label: 'Marks', icon: 'marks', roles: ['teacher', 'floor_supervisor'], category: 'academic' },
  { name: 'attendance', label: 'Take attendance', icon: 'calendar', roles: ['teacher', 'attendance_supervisor', 'floor_supervisor'], category: 'academic' },
  { name: 'timetable', label: 'Timetable', icon: 'timetable',
    roles: ['admin', 'school_owner', 'school_manager', 'floor_supervisor', 'teacher'], category: 'academic' }
];

const CATEGORIES = [
  { id: 'all', label: 'All Screens', icon: 'layers' },
  { id: 'academic', label: 'Academics', icon: 'timetable' },
  { id: 'students', label: 'Students', icon: 'search' },
  { id: 'school', label: 'School & Staff', icon: 'school' }
];

/* Which nav item is lit for a route that is not in the nav. The drill-down screens are
   reached from School, so School stays underlined all the way down â€” losing the highlight
   four levels deep reads as having left the section.
   Highlighting only: a route's *permission* comes from `ROUTE_PERMISSION` and is its own.
   Reading a requirement off this table instead would refuse a teacher their own class
   screen because they cannot see the school dashboard it is reached from. */
const NAV_PARENT = { year: 'school', level: 'school', class: 'school' };

/* The nav items this person may reach: the permission the screen needs, one of the roles
   the item is offered to, and the principal-only flag. Shared by the desktop nav and the
   mobile menu so the two can never offer different screens. */
function permittedNav() {
  return NAV.filter((item) =>
    Store.can(ROUTE_PERMISSION[item.name]) &&
    (!item.roles || item.roles.some((role) => Store.roles().indexOf(role) >= 0)) &&
    (!item.principalOnly || principalExperience())
  );
}

/* The brand is a way home, but "home" is not the School dashboard for every account.
   Pick a role-specific working screen only when that exact destination is permitted, then
   fall back to the first visible navigation item. This prevents the logo itself from sending
   scoped staff into the access-denied screen. */
function accountHomeRoute() {
  const permitted = permittedNav();
  const allowedNames = new Set(permitted.map((item) => item.name));
  const roles = Store.roles();
  const preferences = [
    { roles: ['admin', 'school_owner', 'school_manager'], routes: ['school'] },
    { roles: ['attendance_supervisor'], routes: ['attendance'] },
    { roles: ['floor_supervisor'], routes: ['gradeAssignments', 'attendance', 'marks', 'timetable'] },
    { roles: ['teacher'], routes: ['timetable', 'homework', 'marks', 'attendance'] }
  ];

  for (const preference of preferences) {
    if (!preference.roles.some((role) => roles.includes(role))) continue;
    const destination = preference.routes.find((route) => allowedNames.has(route));
    if (destination) return destination;
  }

  if (permitted.length) return permitted[0].name;
  return Store.can(ROUTE_PERMISSION.chat) ? 'chat' : 'school';
}

/*
 * English labels for the role codes the service ships, for the header chip.
 *
 * A fallback rather than a source of truth: `/v1/rbac/roles` serves the real names in both
 * languages, and a screen that lists or assigns roles reads them from there. This table
 * exists so the header can print "Teacher Â· Attendance Supervisor" on first paint without
 * a second request, and an unknown code falls through to the code itself rather than to a
 * blank â€” a role added next term shows up as `subject_coordinator` and not as nothing.
 */
const ROLE_LABELS = {
  admin: 'Admin',
  school_owner: 'School Owner',
  school_manager: 'School Manager',
  floor_supervisor: 'Floor Supervisor',
  attendance_supervisor: 'Attendance Supervisor',
  teacher: 'Teacher'
};

/**
 * Sign in with a username and password, and pick up the roles that come with it.
 *
 * The shell stays hidden until somebody signs in: every screen is scoped to a person's
 * roles, and a console drawn before it knows who is asking would offer doors and then take
 * them away.
 */
function SignIn() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  return (
    <main className="sis-login-page d-flex align-items-center justify-content-center p-3" style={{ minHeight: '100vh', background: 'radial-gradient(ellipse at top, var(--surface-hover), var(--canvas))' }}>
      <div className="sis-login-card card shadow-lg border-0" style={{ maxWidth: '26rem', width: '100%', borderRadius: '1.25rem', overflow: 'hidden' }}>
        <form
          className="card-body p-4 p-sm-5"
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
          <div className="d-flex align-items-center gap-2 mb-4">
            <div className="sis-brand-mark sis-brand-mark-login">SIS</div>
            <div>
              <span className="fw-bold d-block lh-1" style={{ fontSize: '1.05rem' }}>{t('Student Information Service')}</span>
              <small className="text-body-tertiary">{t('Registrar console')}</small>
            </div>
          </div>
          <h1 className="h5 fw-bold mb-1">{t('Sign in')}</h1>
          <p className="small text-body-tertiary mb-3">
            {t('Signing in shows you the classes and screens your roles cover.')}
          </p>
          <div className="mb-3 text-start">
            <label className="form-label small fw-semibold" htmlFor="login-username">{t('Username')}</label>
            <input
              id="login-username"
              className="form-control"
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              required
            />
          </div>
          <div className="mb-3 text-start">
            <label className="form-label small fw-semibold" htmlFor="login-password">{t('Password')}</label>
            <span className="sis-password-field">
              <input
                id="login-password"
                className="form-control sis-password-input"
                type={showPassword ? 'text' : 'password'}
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
              <button
                className="sis-password-toggle"
                type="button"
                aria-label={t(showPassword ? 'Hide password' : 'Show password')}
                title={t(showPassword ? 'Hide password' : 'Show password')}
                aria-pressed={showPassword}
                onClick={() => setShowPassword((visible) => !visible)}
              >
                <Icon name={showPassword ? 'eyeOff' : 'eye'} size={18} />
              </button>
            </span>
          </div>
          {/* One message for every way a sign-in can fail, because the service answers
              with one â€” a form that told a wrong password from an unknown username
              would be a way to read a school's staff list. */}
          {error ? (
            <p className="alert alert-danger p-2 small mt-3 mb-0 text-start" role="status">
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
    </main>
  );
}

/* -- School strip ---------------------------------------------------------------- */

function SchoolTabs() {
  const state = useStore();
  const admin = Store.roles().indexOf('admin') >= 0;
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
    return <span className="sis-skel" style={{ width: '8.5rem', height: '1.9rem' }} />;
  }
  if (!list.length) return null;

  return (
    <label className="d-flex align-items-center gap-2 flex-grow-1 flex-sm-grow-0 mb-0">
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

/* -- Account chip ----------------------------------------------------------------- */

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
  const initial = (name || 'U').trim().charAt(0).toUpperCase();

  return (
    <div className="sis-user-chip d-flex align-items-center gap-2">
      <div className="sis-user-avatar" title={name}>{initial}</div>
      <div className="sis-user-info d-none d-lg-flex flex-column text-start">
        <span className="sis-user-name">{name}</span>
        <span className="sis-user-role">
          {held.length ? held.map((code) => t(ROLE_LABELS[code] || code)).join(' · ') : t('No role')}
        </span>
      </div>
      <button
        type="button"
        className="btn btn-sm btn-quiet p-1 sis-user-signout"
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
        <Icon name="signout" size={15} />
        <span className="visually-hidden">{t('Sign out')}</span>
      </button>
    </div>
  );
}

/* -- Header ---------------------------------------------------------------------- */

function Header({ onOpenSettings, onSignIn, onToggleChat, unreadChatCount, onToggleMenu }) {
  const state = useStore();
  const canChat = (state.profile?.permissions || []).includes('chat.read');
  const schools = useResource(Store.keys.schools(false), () => api.schools(false));
  const currentSchool = (schools.value || []).find((school) => school.code === state.school);
  const schoolName = pickName(currentSchool, state.lang) || state.school || '';
  const homeRoute = accountHomeRoute();

  return (
    <>
      {state.inflight > 0 ? <div className="sis-progress" aria-hidden="true" /> : null}
      <header className="sis-header d-flex align-items-center justify-content-between w-100 px-3 px-sm-4 py-2">
        <div className="sis-header-brand-group d-flex align-items-center">
          <div className="sis-mobile-header-controls">
            <button
              type="button"
              className="btn btn-sm btn-quiet d-lg-none p-1"
              onClick={onToggleMenu}
              aria-label={t('Menu')}
              title={t('Menu')}
            >
              <Icon name="menu" size={20} />
            </button>
            <AuditBell />
          </div>
          <a
            className="sis-brand text-decoration-none text-body"
            href={Router.href(homeRoute)}
            aria-label={t('Home')}
          >
            <img
              className="sis-company-mark"
              src="./brand/aurexis-mark.svg"
              width="28"
              height="28"
              alt="Aurexis"
              draggable="false"
            />
            <span className="sis-brand-divider" aria-hidden="true" />
            <span className="sis-brand-mark">SIS</span>
            <span className="sis-mobile-school-name d-md-none">{schoolName}</span>
            {/* The product name is the first thing to go on a phone: the badge already says
                which application this is, and the space is worth more than the words. */}
            <span className="d-none d-md-block lh-sm">
              <span className="fw-bold d-block text-nowrap" style={{ fontSize: '0.86rem' }}>{t('Student Information Service')}</span>
              <span className="small text-body-tertiary d-block text-nowrap" style={{ fontSize: '0.72rem' }}>
                {t('Registrar console')}
              </span>
            </span>
          </a>
        </div>

        <div className="sis-header-actions-group d-flex align-items-center sis-push">
          <YearPicker />

          {canChat ? (
            <button
              type="button"
              className={cx('sis-header-chat-btn', unreadChatCount > 0 && 'has-unread')}
              onClick={onToggleChat}
              title={t('Open chat')}
              aria-label={unreadChatCount ? t('{0} unread messages', [unreadChatCount]) : t('Open chat')}
            >
              <Icon name="chat" size={17} />
              <span className="d-none d-sm-inline">{t('Chats')}</span>
              {unreadChatCount > 0 ? (
                <span className="sis-header-chat-badge pulse">
                  {unreadChatCount > 99 ? '99+' : unreadChatCount}
                </span>
              ) : null}
            </button>
          ) : null}

          <Account onSignIn={onSignIn} />

          {/*
            * One button where there were two.
            *
            * Appearance and language stay in the dialog so both choices have visible labels
            * and the year selector keeps enough room on a phone.
            */}
          <button
            className="btn btn-sm btn-quiet p-2"
            onClick={onOpenSettings}
            title={t('Settings — appearance and language')}
          >
            <Icon name="settings" size={17} />
            <span className="visually-hidden">{t('Open settings')}</span>
          </button>
        </div>
      </header>
    </>
  );
}

/* -- Categorized nav ------------------------------------------------------------- */

function Nav({ active }) {
  const here = NAV_PARENT[active] || active;
  const allPermitted = permittedNav();
  const showCategories = Store.roles().includes('school_manager');
  const activeCategory = allPermitted.find((item) => item.name === here)?.category || 'all';
  const [selectedCategory, setSelectedCategory] = useState(activeCategory);
  const visibleItems = !showCategories || selectedCategory === 'all'
    ? allPermitted
    : allPermitted.filter((item) => item.category === selectedCategory);

  return (
    <nav className="sis-nav-wrapper border-bottom" aria-label={t('Screens')}>
      <div className="sis-nav-shell">
        {showCategories ? <div className="sis-nav-categories d-none d-md-flex" role="group" aria-label={t('Screen categories')}>
          {CATEGORIES.map((cat) => {
            const count = cat.id === 'all'
              ? allPermitted.length
              : allPermitted.filter((item) => item.category === cat.id).length;
            if (count === 0 && cat.id !== 'all') return null;
            const isActive = selectedCategory === cat.id;
            return (
              <button
                key={cat.id}
                type="button"
                className={cx('sis-nav-category-pill', 'sis-category-btn', isActive && 'is-active')}
                aria-pressed={isActive}
                onClick={() => setSelectedCategory(cat.id)}
              >
                <Icon name={cat.icon} size={16} />
                <span>{t(cat.label)}</span>
                <span className="sis-nav-category-count" aria-hidden="true">{count}</span>
              </button>
            );
          })}
        </div> : null}

        <ul className="nav sis-nav-track flex-nowrap overflow-auto" style={{ scrollbarWidth: 'none' }}>
          {visibleItems.map((item) => {
            const current = here === item.name;
            return (
              <li className="nav-item" key={item.name}>
                <a
                  className={cx('nav-link sis-nav-link d-flex align-items-center gap-2 text-nowrap', current && 'active')}
                  href={Router.href(item.name)}
                  aria-current={current ? 'page' : undefined}
                >
                  <Icon name={item.icon} size={18} weight={1.8} />
                  <span className="sis-nav-label">{t(item.label)}</span>
                </a>
              </li>
            );
          })}
        </ul>
      </div>
    </nav>
  );
}

/* -- Side chat drawer (popup) ---------------------------------------------------- */

function ChatDrawer({ open, onClose, isFullscreen, onToggleFullscreen }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <>
      <div className="sis-chat-drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <aside
        className={cx('sis-chat-drawer', isFullscreen && 'is-fullscreen')}
        role="dialog"
        aria-modal="true"
        aria-label={t('Staff Chats')}
      >
        <header className="sis-chat-drawer-header">
          <h2 className="sis-chat-drawer-title">
            <Icon name="chat" size={20} />
            <span>{t('Staff Chats')}</span>
            <span className="sis-presence-dot-pulse" title={t('Online')} />
          </h2>
          <div className="sis-chat-drawer-actions">
            <button
              type="button"
              className="sis-chat-drawer-btn is-expand"
              onClick={onToggleFullscreen}
              title={isFullscreen ? t('Exit fullscreen') : t('Fullscreen')}
              aria-label={isFullscreen ? t('Exit fullscreen') : t('Fullscreen')}
            >
              <Icon name={isFullscreen ? 'minimize' : 'expand'} size={17} />
            </button>
            <button
              type="button"
              className="sis-chat-drawer-btn is-close"
              onClick={onClose}
              title={t('Close')}
              aria-label={t('Close')}
            >
              <Icon name="close" size={17} />
            </button>
          </div>
        </header>
        <div className="sis-chat-drawer-body">
          <Chat />
        </div>
      </aside>
    </>
  );
}

/* -- Mobile navigation drawer (offcanvas) ----------------------------------------- */

function MobileNavDrawer({ open, onClose, active, onOpenSettings }) {
  const state = useStore();
  const here = NAV_PARENT[active] || active;

  useEffect(() => {
    if (!open) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [open, onClose]);

  if (!open) return null;

  const allPermitted = permittedNav();
  const account = state.account || {};
  const name = pickName(account, state.lang) || state.profile?.username;
  const heldRoles = Store.roles();
  const roleLabel = heldRoles.length
    ? heldRoles.map((code) => t(ROLE_LABELS[code] || code)).join(' · ')
    : t('No role');
  const homeRoute = accountHomeRoute();
  const showCategories = Store.roles().includes('school_manager');
  const mobileNavItem = (item) => (
    <a
      key={item.name}
      className={cx('sis-mobile-nav-item', here === item.name && 'is-active')}
      href={Router.href(item.name)}
      aria-current={here === item.name ? 'page' : undefined}
      onClick={onClose}
    >
      <Icon name={item.icon} size={19} />
      <span>{t(item.label)}</span>
    </a>
  );

  return (
    <>
      <div className="sis-mobile-drawer-backdrop" onClick={onClose} aria-hidden="true" />
      <aside className="sis-mobile-drawer" role="dialog" aria-modal="true" aria-label={t('Menu')}>
        <div className="sis-mobile-drawer-header">
          <div className="sis-mobile-drawer-identity">
            <a
              className="sis-brand-mark text-decoration-none"
              href={Router.href(homeRoute)}
              aria-label={t('Home')}
              onClick={onClose}
            >
              SIS
            </a>
            <div>
              <strong>{name || t('Student Information Service')}</strong>
              <small>{roleLabel}</small>
            </div>
          </div>
          <button
            type="button"
            className="sis-mobile-drawer-close"
            onClick={onClose}
            aria-label={t('Close menu')}
          >
            <Icon name="close" size={18} />
          </button>
        </div>

        <div className="sis-mobile-drawer-body">
          <div className="sis-mobile-year-card">
            <span>{t('Academic year')}</span>
            <YearPicker />
          </div>

          {showCategories ? CATEGORIES.filter((cat) => cat.id !== 'all').map((cat) => {
            const items = allPermitted.filter((item) => item.category === cat.id);
            if (!items.length) return null;
            return (
              <section key={cat.id} className="sis-mobile-nav-section">
                <h2 className="sis-mobile-section-title">{t(cat.label)}</h2>
                <div className="sis-mobile-nav-list">
                  {items.map(mobileNavItem)}
                </div>
              </section>
            );
          }) : (
            <div className="sis-mobile-nav-list sis-mobile-nav-list-flat">
              {allPermitted.map(mobileNavItem)}
            </div>
          )}
        </div>

        <div className="sis-mobile-drawer-footer">
          <button
            type="button"
            className="sis-mobile-nav-item"
            onClick={() => {
              onClose();
              onOpenSettings();
            }}
          >
            <Icon name="settings" size={19} />
            <span>{t('Settings')}</span>
          </button>
          {state.profile ? (
            <button
              type="button"
              className="sis-mobile-nav-item sis-mobile-signout"
              onClick={() => {
                onClose();
                api.logout().then(
                  () => Store.setAccount(null),
                  () => Store.setAccount(null)
                );
              }}
            >
              <Icon name="signout" size={19} />
              <span>{t('Sign out')}</span>
            </button>
          ) : null}
        </div>
      </aside>
    </>
  );
}

/* -- Floating chat trigger (desktop) --------------------------------------------- */

function FloatingChatButton({ onToggleChat, unreadCount }) {
  const [footerVisible, setFooterVisible] = useState(false);

  useEffect(() => {
    const footer = document.querySelector('.sis-footer');
    if (!footer) return undefined;

    if (typeof window.IntersectionObserver === 'function') {
      const observer = new window.IntersectionObserver(
        ([entry]) => setFooterVisible(entry.isIntersecting),
        { threshold: 0 }
      );
      observer.observe(footer);
      return () => observer.disconnect();
    }

    const update = () => {
      const bounds = footer.getBoundingClientRect();
      setFooterVisible(bounds.top < window.innerHeight && bounds.bottom > 0);
    };
    update();
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    return () => {
      window.removeEventListener('scroll', update);
      window.removeEventListener('resize', update);
    };
  }, []);

  return (
    <button
      type="button"
      className={cx('sis-fab-chat', footerVisible && 'is-footer-visible')}
      onClick={onToggleChat}
      title={t('Open chat')}
      aria-label={unreadCount ? t('{0} unread messages', [unreadCount]) : t('Open chat')}
    >
      <Icon name="chat" size={20} />
      <span>{t('Chats')}</span>
      {unreadCount > 0 ? (
        <span className="sis-fab-badge">{unreadCount > 99 ? '99+' : unreadCount}</span>
      ) : null}
    </button>
  );
}

const AUDIT_EVENT_LABELS = {
  homework_uploaded: 'رفع واجب', homework_deleted: 'حذف واجب', homework_restored: 'استعادة واجب',
  student_created: 'إضافة طالب', enrollment_created: 'تسجيل طالب', document_uploaded: 'رفع مستند',
  create: 'إضافة سجل', update: 'تعديل سجل', delete: 'حذف سجل', soft_delete: 'إلغاء سجل', restore: 'استعادة سجل'
};

function AuditBell() {
  const [open, setOpen] = useState(false);
  const [events, setEvents] = useState([]);
  const userId = Store.state.profile?.user_id || 'anonymous';
  const storageKey = `sis.audit.lastSeen.${userId}`;
  const [lastSeen, setLastSeen] = useState(() => Number(localStorage.getItem(storageKey) || 0));
  const canAudit = Store.can('audit.read');
  useEffect(() => setLastSeen(Number(localStorage.getItem(storageKey) || 0)), [storageKey]);
  useEffect(() => {
    if (!canAudit) return undefined;
    let alive = true;
    const refresh = () => api.auditLog({ limit: 500, offset: 0 }).then((rows) => {
      if (alive && Array.isArray(rows)) setEvents(rows);
    }).catch(() => {});
    refresh();
    const timer = window.setInterval(refresh, 15000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [canAudit]);
  if (!canAudit) return null;
  const unread = events.filter((entry) => entry.id > lastSeen).length;
  const toggle = () => {
    const next = !open; setOpen(next);
    if (next && events.length) { setLastSeen(events[0].id); localStorage.setItem(storageKey, String(events[0].id)); }
  };
  return <div className="sis-audit-bell-host">
    {open ? <section className="sis-audit-popover" aria-label="آخر أحداث النظام"><header><div><strong>آخر أحداث النظام</strong><small>تحديث تلقائي كل 15 ثانية</small></div><button type="button" onClick={() => setOpen(false)} aria-label="إغلاق"><Icon name="close" /></button></header>
      <div className="sis-audit-popover-list">{events.length ? events.slice(0, 8).map((entry) => <div className="sis-audit-popover-item" key={entry.id}><strong>{AUDIT_EVENT_LABELS[entry.action] || 'تحديث في النظام'}</strong><span>{entry.actor_name}{entry.actor_role ? ` · ${entry.actor_role}` : ''}</span>{entry.context?.title ? <small>{entry.context.title}</small> : null}</div>) : <p className="m-0 p-3 text-body-tertiary">لا توجد أحداث حديثة.</p>}</div>
      <a className="sis-audit-popover-more" href={Router.href('auditLog')}>عرض سجل التدقيق بالكامل</a></section> : null}
    <button type="button" className="sis-fab-audit" onClick={toggle} aria-label={unread ? `${unread} إجراءات جديدة` : 'تنبيهات سجل التدقيق'} aria-expanded={open}><Icon name="bell" size={21} />{unread ? <span className="sis-fab-badge" aria-hidden="true">{unread > 99 ? '99+' : unread}</span> : null}</button>
  </div>;
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

const CHAT_PRESENCE_PULSE_MS = 15000;
const CHAT_UNREAD_POLL_MS = 15000;

/* Presence belongs to the signed-in shell, not to the chat route. This keeps a member
   online and acknowledges delivery while they are working anywhere in SIS; opening a
   conversation remains the separate action that marks its messages as read. */
function ChatPresenceHeartbeat() {
  const state = useStore();
  const school = state.school;
  const userId = state.profile?.user_id;
  const canChat = (state.profile?.permissions || []).includes('chat.read');

  useEffect(() => {
    if (!school || !userId || !canChat) return undefined;
    let alive = true;
    const pulse = () => {
      if (!alive || document.visibilityState === 'hidden' || !navigator.onLine) return;
      api.chatPresence(school, {}).catch(() => {});
    };
    pulse();
    const timer = window.setInterval(pulse, CHAT_PRESENCE_PULSE_MS);
    window.addEventListener('online', pulse);
    window.addEventListener('focus', pulse);
    document.addEventListener('visibilitychange', pulse);
    return () => {
      alive = false;
      window.clearInterval(timer);
      window.removeEventListener('online', pulse);
      window.removeEventListener('focus', pulse);
      document.removeEventListener('visibilitychange', pulse);
    };
  }, [school, userId, canChat]);

  return null;
}

/* -- Root ------------------------------------------------------------------------ */

/**
 * The application root. Subscribes to the router once and renders the matched view.
 *
 * The view is keyed by route name so React unmounts the old screen rather than reconciling it
 * with the new one. Two screens with a table in the same position would otherwise reuse those
 * rows and, for one frame, show the marks table filled with roster data â€” and the entrance
 * animation would not replay, so the transition would only ever work on a first visit.
 */
export function App() {
  const [route, setRoute] = useState(Router.current);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [authReady, setAuthReady] = useState(!getSessionToken());
  const [chatDrawerOpen, setChatDrawerOpen] = useState(false);
  const [chatFullscreen, setChatFullscreen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [unreadChatCount, setUnreadChatCount] = useState(0);

  /*
   * The root subscribes to the store as well as to the router, and the language is why.
   * `t()` reads a module-level table rather than a hook, which is what keeps it cheap enough to
   * call once per cell; the cost of that is that changing the language does not by itself
   * invalidate anything. Re-rendering from the root does, and nothing below here is memoised,
   * so one subscription re-translates the whole tree.
   */
  const state = useStore();
  const school = state.school;
  const userId = state.profile?.user_id;
  const canChat = (state.profile?.permissions || []).includes('chat.read');

  /* The unread total for the header, dock and floating badges. Deferred on mount so it does
     not race the first screen's own requests, and skipped in a hidden or offline tab. */
  useEffect(() => {
    if (!school || !userId || !canChat) {
      setUnreadChatCount(0);
      return undefined;
    }
    let alive = true;
    const checkUnread = () => {
      if (!alive || document.visibilityState === 'hidden' || !navigator.onLine) return;
      api.chatConversations(school)
        .then((rows) => {
          if (!alive || !Array.isArray(rows)) return;
          const total = rows.reduce((sum, c) => sum + (c.is_muted ? 0 : (c.unread_count || 0)), 0);
          setUnreadChatCount(total);
        })
        .catch(() => {});
    };
    const initialTimer = window.setTimeout(checkUnread, 2000);
    const intervalTimer = window.setInterval(checkUnread, CHAT_UNREAD_POLL_MS);
    return () => {
      alive = false;
      window.clearTimeout(initialTimer);
      window.clearInterval(intervalTimer);
    };
  }, [school, userId, canChat]);

  useEffect(() => Router.subscribe(setRoute), []);

  /*
   * A reload with a live token restores the session rather than asking again. `/auth/me`
   * answers in the same shape as `/auth/login`, so one setter takes either and the header
   * after a refresh is the header before it. A failure means the token is dead, and the
   * console falls back to the unauthenticated view â€” never to a half-signed-in one, where
   * the header would name somebody whose permissions had already been forgotten.
   */
  useEffect(() => {
    if (!getSessionToken()) return;
    api.me().then(
      (account) => { Store.setAccount(account); setAuthReady(true); },
      () => { Store.setAccount(null); setAuthReady(true); }
    );
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
     one nothing gates â€” the sign-in screen, say â€” and is drawn. */
  const needed = ROUTE_PERMISSION[route.route.name];
  const routeItem = NAV.find((item) => item.name === route.route.name);
  // Admin is a universal role. Role-specific navigation is helpful for staff UX, but
  // must never turn into a frontend-only denial for the account the backend grants all
  // permissions to.
  const roleAllowed = Store.roles().indexOf('admin') >= 0 || !routeItem?.roles || routeItem.roles.some(
    (role) => Store.roles().indexOf(role) >= 0
  );
  const experienceAllowed = !routeItem?.principalOnly || principalExperience();
  const allowed = (!needed || Store.can(needed)) && roleAllowed && experienceAllowed;

  return (
    <div className={cx('sis-app', route.route.name === 'chat' && 'sis-app-chat')}>
      <ChatPresenceHeartbeat />
      <Header
        onOpenSettings={() => setSettingsOpen(true)}
        onSignIn={() => {}}
        onToggleChat={() => setChatDrawerOpen((prev) => !prev)}
        unreadChatCount={unreadChatCount}
        onToggleMenu={() => setMobileMenuOpen((prev) => !prev)}
      />
      <SchoolTabs />
      <Nav active={route.route.name} />

      <main
        className={cx('flex-grow-1 w-100 mx-auto p-3 p-sm-4', route.route.name === 'chat' && 'sis-main-chat')}
        style={{ maxWidth: '96rem' }}
      >
        <div className={cx('sis-rise', route.route.name === 'chat' && 'sis-chat-route')} key={route.route.name}>
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

      {canChat && !chatDrawerOpen && route.route.name !== 'chat' ? (
        <FloatingChatButton
          onToggleChat={() => setChatDrawerOpen(true)}
          unreadCount={unreadChatCount}
        />
      ) : null}

      {/* Not over the full-page chat route: two live copies of the same conversation list
          would each poll, and a message read in one would stay unread in the other. */}
      {canChat && chatDrawerOpen && route.route.name !== 'chat' ? (
        <ChatDrawer
          open={chatDrawerOpen}
          onClose={() => setChatDrawerOpen(false)}
          isFullscreen={chatFullscreen}
          onToggleFullscreen={() => setChatFullscreen((prev) => !prev)}
        />
      ) : null}

      <MobileNavDrawer
        open={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        active={route.route.name}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <Footer />
      <Toasts />
      {/* Rendered last so its backdrop lies over the whole shell â€” including the header the
          button that opened it sits in. */}
      {settingsOpen ? <Settings onClose={() => setSettingsOpen(false)} /> : null}
    </div>
  );
}
