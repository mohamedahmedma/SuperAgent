/*
 * The store: shared UI state, and one cache for the reference data every screen needs.
 *
 * This module is the reason the console became a single page rather than six of them. The
 * old build reloaded the document on every navigation, and each page independently
 * re-fetched the academic years, the terms and the subject catalogue before it could draw
 * anything — four requests and a stylesheet parse to move from the roster screen to the
 * marks screen, on a school laptop, over school wifi. Here that data is fetched once and
 * read from memory afterwards.
 *
 * Two things live in here and nothing else:
 *
 *   state       small, flat, UI-owned: the selected year, the theme, the toasts.
 *   resources   cached server reads, keyed by a string that includes their arguments.
 *
 * Screen data — a class register, a child's marks, an import report — is deliberately NOT
 * cached. It is the thing the registrar came to look at, it changes under them while they
 * work, and serving yesterday's register out of a cache is the one failure this design
 * must not introduce. Only the slow-moving skeleton of the school is held.
 *
 * The pattern is stale-while-revalidate: a cached value is returned immediately and a
 * refresh runs behind it, so a screen paints from memory and corrects itself a moment
 * later rather than showing a spinner over data it already has. A mutation calls
 * `invalidate` with a prefix, which drops every key that starts with it — creating a term
 * invalidates the terms of that year and leaves the subject catalogue alone.
 */
import { api } from './api.js';
import { setLocale } from './i18n.js';

var SCHOOL_KEY = 'sis.school';
var THEME_KEY = 'sis.theme';
var LANG_KEY = 'sis.lang';
var YEAR_KEY = 'sis.year';

function readTheme() {
  /* Only Light and Dark are valid. Legacy `system` and malformed stored values become Light. */
  return readLocal(THEME_KEY, 'light') === 'dark' ? 'dark' : 'light';
}

/*
 * localStorage, not sessionStorage, and the distinction matters. None of these three is
 * a credential: they are a colour scheme, a reading direction and which academic year a
 * clerk works in all day. Forgetting them on every tab close is the behaviour that made
 * the old console feel like a form rather than a tool. The API key is the opposite case
 * and stays in sessionStorage, in api.js, for exactly the reason stated there.
 */
function readLocal(key, fallback) {
  try {
    var raw = window.localStorage.getItem(key);
    return raw === null || raw === undefined || raw === '' ? fallback : raw;
  } catch (e) {
    return fallback; // Private mode: behave as "nothing stored", never as broken.
  }
}

function writeLocal(key, value) {
  try {
    if (value === null || value === undefined || value === '') {
      window.localStorage.removeItem(key);
    } else {
      window.localStorage.setItem(key, String(value));
    }
  } catch (e) {
    /* Storage disabled. The session still works; it just will not be remembered. */
  }
}

/* -- State ------------------------------------------------------------------------ */

var state = {
  /**
   * The school every screen is inside. Null until the school list loads.
   *
   * Remembered across sessions like the year, and for the same reason: a clerk works at
   * one branch all day, and making them re-pick it every morning is how the wrong branch
   * gets picked. It is stored rather than derived because "the only school" stops being
   * an answer the moment a second one exists.
   */
  school: readLocal(SCHOOL_KEY, null),
  /** The academic year every screen scopes itself to. Null until the years load. */
  year: readLocal(YEAR_KEY, null),
  theme: readTheme(),
  lang: readLocal(LANG_KEY, 'en'),
  /** Requests in flight, for the header progress bar. A count, not a boolean: two
      screens loading at once must not have the first one to finish hide the bar. */
  inflight: 0,
  online: null,
  toasts: [],
  /** The signed-in account — names, language, school — or null when nobody is. */
  account: null,
  /** Their roles, permissions and the scope each is held at. See `can` / `canIn`. */
  profile: null
};

var listeners = [];

function notify() {
  listeners.slice().forEach(function (fn) {
    fn(state);
  });
}

function subscribe(fn) {
  listeners.push(fn);
  return function () {
    listeners = listeners.filter(function (item) {
      return item !== fn;
    });
  };
}

/** Replace state with a shallow merge and tell everyone. The only writer. */
function set(patch) {
  var next = {};
  Object.keys(state).forEach(function (key) {
    next[key] = state[key];
  });
  Object.keys(patch).forEach(function (key) {
    next[key] = patch[key];
  });
  state = next;
  notify();
}

/* -- Preferences ------------------------------------------------------------------ */

/**
 * Move to another school, and drop the year with it.
 *
 * Clearing the year is the whole point of this being its own function. Year codes are
 * globally unique, so the remembered `2025-2026` belongs to the school just left — and a
 * screen that kept it would show one school's tabs above another school's classes, with
 * nothing on the page admitting it. The year picker settles on the new school's current
 * year a moment later.
 */
function setSchool(code) {
  if (code === state.school) return;
  writeLocal(SCHOOL_KEY, code);
  writeLocal(YEAR_KEY, null);
  set({ school: code || null, year: null });
}

function setYear(code) {
  if (code === state.year) return;
  writeLocal(YEAR_KEY, code);
  set({ year: code || null });
}

/* -- Who is signed in, and what they may do --------------------------------------- */

/*
 * Everything below decides what the console *draws*. None of it decides what the service
 * *allows* — every one of these permissions is checked again on the server, and the checks
 * there are the boundary. Hiding a control the caller may not use is a courtesy: it stops
 * a teacher clicking Save on a class that was never theirs and reading a 403 they cannot
 * act on. Treating it as security would mean an editor with dev tools is an administrator.
 *
 * Two questions, and the difference between them is the whole of the scope model:
 *
 *   can(p)         may this person do `p` anywhere? Decides whether a *screen* exists.
 *   canIn(p, at)   may they do it to *this* class, this grade? Decides whether the button
 *                  on the row is live.
 *
 * A console built on `can` alone shows a teacher of one classroom an edit control on every
 * classroom in the school, which is the exact failure the scopes were added to prevent.
 */

/**
 * The signed-in account: names, language, school, and the profile inside it.
 *
 * `/auth/login` and `/auth/me` answer in the same shape, so this one setter takes either
 * and the page after a reload is the page before it.
 */
function setAccount(account) {
  var schoolCode = account && account.school_code;
  if (schoolCode) writeLocal(SCHOOL_KEY, schoolCode);
  set({
    account: account || null,
    profile: (account && account.profile) || null,
    school: schoolCode || state.school
  });
}

/** Kept for callers holding a bare profile; the account fields go unset. */
function setProfile(profile) {
  set({ profile: profile || null, account: profile ? state.account : null });
}

function can(permission) {
  /* No human session means the legacy integration console remains available — the service
     still admits an unauthenticated caller, and a console that hid everything from one
     would be refusing what the server allows. */
  if (!state.profile) return false;
  var held = state.profile.permissions;
  return Array.isArray(held) && held.indexOf(permission) >= 0;
}

/* Which field of a `canIn` question each kind of grant is compared against. */
var SCOPE_FIELD = {
  school: 'school',
  track: 'track',
  year_level: 'yearLevel',
  class_section: 'classSection',
  subject: 'subject'
};

/*
 * Does one grant's scope answer for the place being asked about?
 *
 * The same rule as `Scope.covers` on the server, and deliberately the same shape so the
 * two can be read side by side: a wider scope covers a narrower one, but only when the
 * caller *says* which wider thing it is inside. A question naming nothing is answered by a
 * global grant alone — the conservative reading, and the one that errs towards hiding a
 * control rather than towards offering one the server will refuse.
 *
 * Compared on **codes**, not on the surrogate ids the grant also carries. A screen here
 * knows it is showing `P1A` in `AR-P1`; it has never seen class 47, and threading ids
 * through the whole UI so this one function could use them would be a second identity
 * system maintained for the sake of a check the server repeats anyway. The server sends
 * `scope_code` beside `scope_id` for exactly this.
 */
function scopeCovers(grant, at) {
  if (grant.scope_type === 'global') return true;
  var field = SCOPE_FIELD[grant.scope_type];
  if (!field) return false;
  var asked = (at || {})[field];
  /* A grant whose target has been deleted carries no code and covers nothing. Refusing
     here hides a control; the alternative — treating a missing code as a match — would
     show one on every class in the school. */
  if (asked == null || grant.scope_code == null) return false;
  return String(asked) === String(grant.scope_code);
}

/**
 * May this person do `permission` to the thing at `at`?
 *
 * `at` is a plain object of codes, each optional:
 * `{ school, track, yearLevel, classSection, subject }`. Name as many as the screen knows —
 * the more it names, the more grants can match, and a narrow grant is invisible to a
 * question that only says which school.
 */
function canIn(permission, at) {
  if (!state.profile) return false;
  var grants = state.profile.grants;
  /* A payload from a service too old to send scopes. Falling back to the unscoped answer
     keeps the console usable rather than blanking every control; the server is still the
     boundary, so the worst case is a button that reports a refusal. */
  if (!Array.isArray(grants)) return can(permission);
  for (var i = 0; i < grants.length; i += 1) {
    if (grants[i].permission === permission && scopeCovers(grants[i], at)) return true;
  }
  return false;
}

/** The distinct role codes this person holds, for the header and for a role badge. */
function roles() {
  if (!state.profile || !Array.isArray(state.profile.roles)) return [];
  var seen = [];
  state.profile.roles.forEach(function (row) {
    if (seen.indexOf(row.role_code) < 0) seen.push(row.role_code);
  });
  return seen;
}

function setTheme(theme) {
  var value = theme === 'dark' ? 'dark' : 'light';
  writeLocal(THEME_KEY, value);
  applyTheme(value);
  set({ theme: value });
}

/* -- Appearance ------------------------------------------------------------------- */

function applyTheme(theme) {
  var root = window.document.documentElement;
  /*
   * `data-bs-theme`, which is the attribute Bootstrap's own components already watch. Setting
   * a second custom attribute alongside it would mean the framework and the console could
   * disagree about which theme is on — and they would, the first time somebody styled
   * something against one and not the other.
   *
   * The value is normalised again at the DOM boundary so a stale caller cannot introduce a
   * third appearance.
   */
  root.setAttribute('data-bs-theme', theme === 'dark' ? 'dark' : 'light');
}

/*
 * The language toggle now changes three things, where it used to change two.
 *
 * It still flips `<html lang>` and `<html dir>`, and it still decides which of the two names
 * the service stores -- `name_ar` or `name_en` -- each screen shows first. What is new is that
 * it also switches the console's own wording, which this comment used to say it would never do:
 * "a half-finished translation table is worse than none" was the argument, and it was a fair
 * one right up until somebody asked for the Arabic. The table is in `locale/ar.js`, keyed by
 * the English sentence, and a line missing from it renders in English rather than as a key --
 * so the half-finished state degrades to the old behaviour rather than to something broken.
 */
function setLang(lang) {
  var value = lang === 'ar' ? 'ar' : 'en';
  writeLocal(LANG_KEY, value);
  applyLang(value);
  set({ lang: value });
}

function applyLang(lang) {
  var root = window.document.documentElement;
  root.setAttribute('lang', lang);
  root.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');
  /* The translation table reads a module-level value rather than the store, so that `t()` --
     which runs once per label in a four-hundred-row register -- is a dictionary lookup and not
     a subscription. Pushed here so the two can never be more than an instant apart. */
  setLocale(lang);
}

/* -- Toasts ---------------------------------------------------------------------- */

var toastSeq = 0;

/**
 * Announce something that happened. Used for outcomes a registrar must not miss and
 * must not have to look for: a batch committed, a term created, a guardian revoked.
 *
 * Failures are NOT toasted by default. An error belongs beside the control that caused
 * it, where the registrar is already looking — a toast in the corner about a rejected
 * form is a message they read after they have started retyping.
 */
function toast(tone, title, detail, ttl) {
  toastSeq += 1;
  var id = toastSeq;
  var next = state.toasts.concat([{ id: id, tone: tone, title: title, detail: detail || null }]);
  set({ toasts: next });

  var life = ttl === undefined ? (tone === 'bad' ? 9000 : 5000) : ttl;
  if (life > 0) {
    window.setTimeout(function () {
      dismiss(id);
    }, life);
  }
  return id;
}

function dismiss(id) {
  set({
    toasts: state.toasts.filter(function (item) {
      return item.id !== id;
    })
  });
}

/* -- Request counter ------------------------------------------------------------- */

/**
 * Wrap a promise so the header bar knows something is happening. Increment and
 * decrement rather than set-and-clear, so overlapping requests cannot leave the bar
 * stuck on after the slower of the two finishes.
 */
function track(promise) {
  set({ inflight: state.inflight + 1 });
  var done = function () {
    set({ inflight: Math.max(0, state.inflight - 1) });
  };
  return promise.then(
    function (value) {
      done();
      return value;
    },
    function (error) {
      done();
      throw error;
    }
  );
}

/* -- Resource cache -------------------------------------------------------------- */

/*
 * entry = {
 *   value:   the last successful result, or undefined
 *   error:   the last failure, or null
 *   loading: whether a request is in flight
 *   at:      when `value` was fetched (ms), for the staleness check
 *   promise: the in-flight promise, so two components mounting together share one request
 *   watchers: subscriber callbacks
 * }
 */
var cache = {};

/* Long enough that moving between screens never refetches; short enough that a
   registrar who creates a class in another tab sees it within a minute of coming back. */
var FRESH_MS = 60000;

function entryFor(key) {
  if (!cache[key]) {
    cache[key] = { value: undefined, error: null, loading: false, at: 0, promise: null, watchers: [] };
  }
  return cache[key];
}

function announce(key) {
  var entry = cache[key];
  if (!entry) return;
  entry.watchers.slice().forEach(function (fn) {
    fn(snapshot(key));
  });
}

function snapshot(key) {
  var entry = entryFor(key);
  return {
    value: entry.value,
    error: entry.error,
    loading: entry.loading,
    /* True when there is something to draw. Distinguished from `!loading` because a
       stale-while-revalidate refresh is loading *and* has a value, and a screen in that
       state must keep drawing rather than fall back to a skeleton. */
    ready: entry.value !== undefined
  };
}

/**
 * Read a cached server value, fetching it if needed.
 *
 * `loader` is a function returning a promise. It is called at most once per key while a
 * request is outstanding, so three components asking for the subject catalogue during
 * the same mount produce one HTTP request and share its result.
 */
function read(key, loader, options) {
  var opts = options || {};
  var entry = entryFor(key);
  var fresh = entry.value !== undefined && Date.now() - entry.at < FRESH_MS;

  if (entry.promise) return entry.promise;
  if (fresh && !opts.force) return Promise.resolve(entry.value);

  entry.loading = true;
  entry.error = null;
  announce(key);

  entry.promise = track(loader()).then(
    function (value) {
      entry.value = value;
      entry.error = null;
      entry.at = Date.now();
      entry.loading = false;
      entry.promise = null;
      announce(key);
      return value;
    },
    function (error) {
      entry.error = error;
      entry.loading = false;
      entry.promise = null;
      /*
       * The stale value is kept on failure, on purpose. A transient outage should leave
       * the class list on screen with an error beside it, not replace a working screen
       * with an apology — the registrar can still read what the school looked like a
       * minute ago, which is almost certainly still true.
       */
      announce(key);
      throw error;
    }
  );
  return entry.promise;
}

function watch(key, fn) {
  var entry = entryFor(key);
  entry.watchers.push(fn);
  return function () {
    entry.watchers = entry.watchers.filter(function (item) {
      return item !== fn;
    });
  };
}

/**
 * Drop every cached key beginning with `prefix`, and refetch the ones somebody is
 * currently watching.
 *
 * Called after a write. The prefix is what makes this precise: committing a roster
 * invalidates `roster:` and leaves the subject catalogue, the terms and the year list
 * untouched, so a commit costs one refetch rather than emptying the cache and making
 * the next three screens slow.
 */
function invalidate(prefix) {
  Object.keys(cache).forEach(function (key) {
    if (prefix && key.indexOf(prefix) !== 0) return;
    var entry = cache[key];
    entry.at = 0;
    if (entry.watchers.length && entry.loader) {
      read(key, entry.loader, { force: true }).catch(function () {
        /* Reported through the watcher; nothing to add here. */
      });
    } else {
      announce(key);
    }
  });
}

/** Remember the loader against the key, so `invalidate` can refetch what is on screen. */
function register(key, loader) {
  entryFor(key).loader = loader;
}

function peek(key) {
  return snapshot(key);
}

/* -- Cache keys ------------------------------------------------------------------
 *
 * Named here rather than spelled out at each call site: the whole correctness of the
 * cache rests on two screens asking for the same data producing the same string, and on
 * `invalidate('terms:')` matching every key a term write should drop.
 */
var keys = {
  schools: function (includeInactive) {
    return 'schools:' + (includeInactive ? 'all' : 'active');
  },
  /* Keyed by school, because the route answers differently per school and an unkeyed
     entry would serve the first school's years to the second. */
  years: function (school) {
    return 'years:' + (school || '');
  },
  levels: function (school) {
    return 'levels:' + (school || '');
  },
  classes: function (year) {
    return 'classes:' + (year || '');
  },
  terms: function (year) {
    return 'terms:' + (year || '');
  },
  /* One year and everything attached to it. Under the `years:` prefix so that creating a
     year — which also builds its terms — drops this with the year list it belongs to. */
  yearDetail: function (year) {
    return 'years:detail:' + (year || '');
  },
  /* The year is in the key because it is in the question. Keyed without it, two years
     would share one cached catalogue and the second screen to load would show the
     first year's subjects under the second year's heading. */
  subjects: function (year, includeInactive) {
    return 'subjects:' + (year || '') + ':' + (includeInactive ? 'all' : 'active');
  },
  /* The subjects one rung teaches, which is a different question from the year's whole
     catalogue and so is a different key. Sharing `subjects:` for both would mean the
     first marks screen to load decided what every other screen thought the year taught. */
  gradeSubjects: function (year, yearLevel) {
    return 'subjects:' + (year || '') + ':grade:' + (yearLevel || '');
  },
  /* The assignment board, keyed by year for the same reason the catalogue is. Under the
     `subjects:` prefix on purpose: assigning a subject changes what a rung teaches, and
     `invalidate('subjects:')` after any subject write should drop this too. */
  subjectAssignments: function (year) {
    return 'subjects:assignments:' + (year || '');
  },
  /* A school's Arabic/Languages tracks. Rarely changes, but keyed rather than read
     one-shot because both the school screen and the assignment board ask for it. */
  tracks: function (school) {
    return 'tracks:' + (school || '');
  },
  /*
   * One child's record and the three lists hanging off it. These are keyed — rather than
   * read one-shot — because the record screen shows the same child from four panels at once:
   * her identity, her placements, her guardians, her attendance, and an insights panel that
   * counts all three. Un-keyed, opening one child fired eleven requests for four answers.
   *
   * `invalidate('student:')` after an edit drops every one of them together, which is the
   * property that matters: her name changing in the header and not in the breadcrumb is the
   * bug a shared key makes impossible.
   */
  student: function (number) {
    return 'student:' + (number || '');
  },
  placements: function (number) {
    return 'placements:' + (number || '');
  },
  guardians: function (number) {
    return 'guardians:' + (number || '');
  },
  attendance: function (number, from, to) {
    return 'attendance:' + (number || '') + ':' + (from || '') + ':' + (to || '');
  }
};

/* -- Reference loaders ----------------------------------------------------------- */


var load = {
  schools: function (includeInactive) {
    var key = keys.schools(includeInactive);
    var loader = function () {
      return api.schools(includeInactive);
    };
    register(key, loader);
    return read(key, loader);
  },
  years: function (school) {
    var key = keys.years(school);
    var loader = function () {
      return api.years(school);
    };
    register(key, loader);
    return read(key, loader);
  },
  levels: function (school) {
    var key = keys.levels(school);
    var loader = function () {
      return api.schoolLevels(school);
    };
    register(key, loader);
    return read(key, loader);
  },
  classes: function (year) {
    var key = keys.classes(year);
    var loader = function () {
      return api.classes(year);
    };
    register(key, loader);
    return read(key, loader);
  },
  terms: function (year) {
    var key = keys.terms(year);
    var loader = function () {
      return api.terms(year);
    };
    register(key, loader);
    return read(key, loader);
  },
  subjects: function (year, includeInactive) {
    var key = keys.subjects(year, includeInactive);
    var loader = function () {
      return api.subjects(year, includeInactive);
    };
    register(key, loader);
    return read(key, loader);
  },
  gradeSubjects: function (year, yearLevel) {
    var key = keys.gradeSubjects(year, yearLevel);
    var loader = function () {
      return api.subjects(year, false, yearLevel);
    };
    register(key, loader);
    return read(key, loader);
  },
  subjectAssignments: function (year) {
    var key = keys.subjectAssignments(year);
    var loader = function () {
      return api.subjectAssignments(year);
    };
    register(key, loader);
    return read(key, loader);
  },
  tracks: function (school) {
    var key = keys.tracks(school);
    var loader = function () {
      return api.schoolTracks(school);
    };
    register(key, loader);
    return read(key, loader);
  }
};

/* -- Boot ----------------------------------------------------------------------- */

applyTheme(state.theme);
applyLang(state.lang);

export const Store = {
  get state() {
    return state;
  },
  subscribe: subscribe,
  set: set,
  setSchool: setSchool,
  setYear: setYear,
  setProfile: setProfile,
  setAccount: setAccount,
  can: can,
  canIn: canIn,
  roles: roles,
  setTheme: setTheme,
  setLang: setLang,
  toast: toast,
  dismiss: dismiss,
  track: track,
  read: read,
  watch: watch,
  peek: peek,
  invalidate: invalidate,
  register: register,
  keys: keys,
  load: load
};
