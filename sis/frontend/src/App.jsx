/*
 * The shell: header, school strip, nav, view outlet, footer.
 *
 * Built for a phone first, because that is where most of this console is read. The chrome is
 * three shallow rows rather than one wide one, and each row solves a different problem at
 * 360px:
 *
 *   Row 1  brand, the year picker and the settings button. The year picker is *in* the header
 *          rather than on each screen because every screen is scoped to a year, and an
 *          off-screen year selector means a registrar cannot see which year the four hundred
 *          rows below belong to. It grows to fill the row on a phone and shrinks to its
 *          content from `sm` up. The appearance, the page colour and the name language used to
 *          be icon buttons beside it and are now behind the one gear: three preferences that
 *          each need a label do not fit in a row that also has to hold a year.
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
import { api } from './api.js';
import { Router } from './router.js';
import { Store } from './store.js';
import { pickName, useResource, useStore } from './hooks.js';
import { Icon, Select, Toasts, cx } from './components/Ui.jsx';
import { Settings } from './components/Settings.jsx';
import { t } from './i18n.js';

/* Order is the order of the work: see the school, find a child, put children in classes,
   record who may ask about them, record what they scored, audit what was written. */
const NAV = [
  { name: 'school', label: 'School', icon: 'dashboard' },
  { name: 'student', label: 'Find a child', icon: 'search' },
  { name: 'roster', label: 'Roster', icon: 'upload' },
  { name: 'guardians', label: 'Guardians', icon: 'people' },
  { name: 'marks', label: 'Marks', icon: 'marks' },
  { name: 'batches', label: 'Batches', icon: 'batches' }
];

/* Which nav item is lit for a route that is not in the nav. The drill-down screens are
   reached from School, so School stays underlined all the way down — losing the highlight
   four levels deep reads as having left the section. */
const NAV_PARENT = { year: 'school', level: 'school', class: 'school' };

/* -- School strip ---------------------------------------------------------------- */

function SchoolTabs() {
  const state = useStore();
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

  if (list.length < 2) return null;

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

/* -- Header ---------------------------------------------------------------------- */

function Header({ onOpenSettings }) {
  const state = useStore();

  return (
    <>
      {state.inflight > 0 ? <div className="sis-progress" aria-hidden="true" /> : null}
      <header
        className="d-flex flex-wrap align-items-center gap-2 gap-sm-3 px-3 px-sm-4 py-2 border-bottom"
        style={{ background: 'var(--canvas)' }}
      >
        <a
          className="d-flex align-items-center gap-2 text-decoration-none text-body"
          href={Router.href('school')}
        >
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

          {/*
            * One button where there were two.
            *
            * The theme toggle and the EN/ع pair used to live here, and the header was the wrong
            * room for both: no space for a label, a theme control that cycled through three
            * states while showing only one icon, and nowhere at all to put a third preference.
            * They are all in the dialog now, where each one can say what it is — and the page
            * colour, which needs a demonstration rather than a label, has somewhere to be
            * demonstrated.
            */}
          <button
            className="btn btn-sm btn-quiet"
            onClick={onOpenSettings}
            title={t('Settings — appearance, page colour, names')}
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
      className="border-bottom px-3 px-sm-4"
      style={{ background: 'var(--canvas)' }}
      aria-label={t('Screens')}
    >
      <ul
        className="nav nav-underline flex-nowrap overflow-auto"
        style={{ scrollbarWidth: 'none' }}
      >
        {NAV.map((item) => {
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

/**
 * Carries one piece of live information: whether the service is answering. Polled rather than
 * inferred from the last request, because the useful case is the registrar who has had a
 * screen open for an hour and is about to start typing. A failed poll never blanks anything —
 * it colours a dot, and the screens keep whatever they already loaded.
 */
function Footer() {
  const state = useStore();

  useEffect(() => {
    let stopped = false;
    const ping = () =>
      api.health().then(
        () => !stopped && Store.set({ online: true }),
        () => !stopped && Store.set({ online: false })
      );

    ping();
    const timer = setInterval(ping, 60000);
    /* Poll on return to the tab as well: a laptop closed at lunch and reopened has a stale dot
       for up to a minute otherwise, which is exactly when it is read. */
    window.addEventListener('focus', ping);
    return () => {
      stopped = true;
      clearInterval(timer);
      window.removeEventListener('focus', ping);
    };
  }, []);

  const colour =
    state.online === null ? 'var(--grey-400)' : state.online ? 'var(--ok-ink)' : 'var(--bad-ink)';
  const word =
    state.online === null ? 'checking…' : state.online ? 'service online' : 'service unreachable';

  return (
    <footer
      className="sis-footer sis-no-print d-flex flex-column flex-md-row align-items-md-center gap-2 gap-md-3 px-3 px-sm-4 py-3 border-top small text-body-tertiary"
      style={{ background: 'var(--canvas)' }}
    >
      <span className="d-flex align-items-center gap-2">
        <span
          style={{ width: '.5rem', height: '.5rem', borderRadius: '50%', background: colour }}
          aria-hidden="true"
        />
        {word}
      </span>
      <span className="flex-md-grow-1 text-md-end">
        {t('Marks are stated figures, reported exactly as the school recorded them. A blank is not a zero.')}
      </span>
      <a href="/docs" target="_blank" rel="noreferrer">
        {t('API reference')}
      </a>
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
  /*
   * The root subscribes to the store as well as to the router, and the language is why.
   * `t()` reads a module-level table rather than a hook, which is what keeps it cheap enough to
   * call once per cell; the cost of that is that changing the language does not by itself
   * invalidate anything. Re-rendering from the root does, and nothing below here is memoised,
   * so one subscription re-translates the whole tree.
   */
  useStore();

  useEffect(() => Router.subscribe(setRoute), []);

  if (!route) return null;
  const View = route.route.view;

  return (
    <div className="sis-app">
      <Header onOpenSettings={() => setSettingsOpen(true)} />
      <SchoolTabs />
      <Nav active={route.route.name} />
      <main className="flex-grow-1 w-100 mx-auto p-3 p-sm-4" style={{ maxWidth: '96rem' }}>
        <div className="sis-rise" key={route.route.name}>
          <View params={route.params} />
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
