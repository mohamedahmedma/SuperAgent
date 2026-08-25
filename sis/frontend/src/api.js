/*
 * The only place in the registrar UI that speaks to the service.
 *
 * Written as a plain IIFE on `window.SIS` rather than an ES module: the UI is served by
 * StaticFiles with no build step, and a registrar debugging a school laptop must be able
 * to open these files from disk. `<script type="module">` is blocked by the file://
 * origin policy, so a module version of this file would work over http and silently do
 * nothing when double-clicked. There is no bundler here to make that trade-off back.
 *
 * Every screen goes through this file. When the old `structure.html` grew its own
 * `fetchClient()` the two clients drifted — a different base path, a different key header,
 * a different idea of what an error body looks like — and screens started calling routes
 * that had never existed. There is one client, and these are the only paths it knows.
 *
 * The six pages that used to import it are now one page and six view modules, and the rule
 * survived the move unchanged: `sis/tests/test_ui_contract.py` fails the build if anything
 * outside this file calls `fetch`. What did change is that the base path and the key header
 * are now stated in exactly one place for a console that has six screens instead of six
 * documents — the drift this file was written to prevent no longer has a second file to
 * happen in.
 */

/* Same-origin. The UI is mounted at /ui by the service itself, so the API is a
   sibling path and never a configurable host — there is no CORS story to get wrong. */
var BASE = '/v1';

/*
 * Security is explicitly out of scope for this UI: there is no login, no session and no
 * cookie, and this default is why. A registrar who opens the page on a fresh machine
 * gets a working screen instead of an empty form they have no way to fill in. Overwrite
 * it in the header field when the school issues a real key.
 */
var DEFAULT_KEY = 'dev-sis-registrar';

/*
 * sessionStorage, deliberately, and never localStorage.
 *
 * This key is a staff credential that can rewrite a whole school's marks, and school
 * front-office machines are shared and rarely logged out of. localStorage would keep
 * the registrar's key readable by the next person to open the browser, for months.
 * sessionStorage dies with the tab, which is the closest thing to "log out" that a
 * page with no session has.
 */
var STORAGE_KEY = 'sis.api_key';

var DASH = '—'; // em dash: the one rendering of "no mark was recorded"

/* The slot `store.js` keeps the selected school in. Duplicated as a literal rather than
   imported because `api.js` is the lower layer — the store imports it, not the other way
   round — and a cycle between them would break the module graph. Kept in step by name:
   both spell it `sis.school`. */
var SCHOOL_STORAGE_KEY = 'sis.school';

function storedKey() {
  try {
    var raw = window.sessionStorage.getItem(STORAGE_KEY);
    return raw === null || raw === undefined ? '' : String(raw).trim();
  } catch (e) {
    return ''; // Private mode / storage disabled: behave as "nothing stored", not broken.
  }
}

/** The key every request actually sends: what the registrar typed, else the default. */
function getKey() {
  return storedKey() || DEFAULT_KEY;
}

/**
 * The school every request is answered from.
 *
 * Schools are separated physically — one database each — so this header does not narrow a
 * query, it chooses the connection. Read from the same `localStorage` slot the store keeps
 * the selected school in (`store.js`, SCHOOL_KEY) rather than being passed down through
 * every call site: the school is a property of the whole session, and threading it through
 * forty functions would mean forty chances to forget it, each one a request answered from
 * whichever database the service picked.
 *
 * Empty when no school has been chosen, and omitted from the request entirely — a
 * single-school service ignores the header, and a multi-school one refuses the request
 * rather than guessing, which is the answer that surfaces the problem instead of hiding it.
 */
function schoolCode() {
  try {
    var raw = window.localStorage.getItem(SCHOOL_STORAGE_KEY);
    return raw === null || raw === undefined ? '' : String(raw).trim();
  } catch (e) {
    return ''; // Private mode / storage disabled: behave as "nothing chosen".
  }
}

/**
 * Store a key. A blank value clears the override rather than saving an empty string,
 * so emptying the field returns the page to the default instead of sending `X-API-Key:`
 * with nothing after it and 401ing on every screen.
 */
function setKey(value) {
  var text = value === null || value === undefined ? '' : String(value).trim();
  try {
    if (text) {
      window.sessionStorage.setItem(STORAGE_KEY, text);
    } else {
      window.sessionStorage.removeItem(STORAGE_KEY);
    }
    return true;
  } catch (e) {
    return false;
  }
}

function clearKey() {
  try {
    window.sessionStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    /* nothing to clear */
  }
}

/** Whether a request will carry a key at all — true even when it is the default one. */
function hasKey() {
  return getKey().length > 0;
}

/** True while the page is running on the built-in development key. */
function isDefaultKey() {
  return storedKey().length === 0;
}

/* -- Text helpers ------------------------------------------------------------------
 *
 * Both of these live here rather than in each page because both have exactly one
 * correct implementation and several plausible wrong ones, and the wrong ones are
 * invisible until a real school's data hits them.
 */

var ESCAPES = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
  '`': '&#96;'
};

/**
 * Escape text before it is interpolated into markup.
 *
 * Names arrive from uploaded spreadsheets — anybody who can hand the registrar an .xlsx
 * can put a `<script>` tag in the full_name column, and these tables are rendered with
 * innerHTML. Quotes and backticks are escaped as well as angle brackets so the same
 * function is safe inside an attribute value, which is where half the interpolation in
 * these pages happens.
 */
function escapeText(text) {
  if (text === null || text === undefined) return '';
  return String(text).replace(/[&<>"'`]/g, function (ch) {
    return ESCAPES[ch];
  });
}

/** Trim the float noise a percentage picks up in transit: 88.5 stays, 88.50 does not. */
function formatNumber(value) {
  var rounded = Math.round(value * 100) / 100;
  return String(rounded);
}

/**
 * Render one grade line's mark.
 *
 * A blank is a dash and a real zero is `0%`, and the whole point of this function is
 * that the difference survives. `percentage || 0`, `if (percentage)` and `percentage ?
 * ... : '—'` all report a child who scored nothing as unmarked, which is a different
 * and much worse statement about her than the one the service made. The service settles
 * it explicitly with `is_graded`, so branch on that and never on the number's truthiness.
 */
function gradeText(grade) {
  if (!grade || grade.is_graded !== true) return DASH;
  var pct = grade.percentage;
  if (pct === null || pct === undefined || pct === '') return DASH;
  var number = typeof pct === 'number' ? pct : Number(pct);
  if (!isFinite(number)) return DASH; // isFinite(0) is true — a zero still renders.
  return formatNumber(number) + '%';
}

/* -- Errors ------------------------------------------------------------------------ */

/**
 * One error type for every failure, carrying `kind` so the UI can branch without
 * re-reading status numbers, and `code`/`field` so it can point at the cell at fault.
 *
 * The kinds are separated by what the registrar must DO next, which is the only
 * distinction a UI can act on: `network` means try again, `unauthorized` means retype
 * the key, `forbidden` means fetch a different key entirely (retyping cannot help —
 * a reader key is refused by write routes by exact scope equality), `too_large` means
 * split the spreadsheet, `gone` means the preview aged out and the file must be
 * uploaded again, `api` means read the message, `http` means the request never
 * reached the service intact.
 */
function ApiError(kind, status, code, message, field, detail) {
  var self = Object.create(ApiError.prototype);
  self.name = 'ApiError';
  self.kind = kind;
  self.status = status || 0;
  self.code = code || kind;
  self.message = message || 'Request failed.';
  self.field = field || null;
  self.detail = detail || null;
  if (Error.captureStackTrace) Error.captureStackTrace(self, ApiError);
  return self;
}
ApiError.prototype = Object.create(Error.prototype);
ApiError.prototype.constructor = ApiError;

ApiError.prototype.isAuth = function () {
  return this.kind === 'unauthorized' || this.kind === 'forbidden';
};

ApiError.prototype.toString = function () {
  return 'ApiError: ' + this.message + ' (' + this.code + ', HTTP ' + this.status + ')';
};

function query(params) {
  if (!params) return '';
  var search = new URLSearchParams();
  Object.keys(params).forEach(function (name) {
    var value = params[name];
    if (value === null || value === undefined || value === '') return;
    /* Arrays repeat the key rather than joining with commas: the import report reads
       `?outcome=rejected&outcome=updated`, and a comma-joined value arrives as one
       unknown outcome and 422s. */
    if (Array.isArray(value)) {
      value.forEach(function (item) {
        if (item !== null && item !== undefined && item !== '') search.append(name, item);
      });
    } else {
      search.append(name, value);
    }
  });
  var text = search.toString();
  return text ? '?' + text : '';
}

/* A pydantic error's `loc` is like ["body", "starts_on"] or ["query", "academic_year"];
   the first element names the part of the request and is noise to a registrar looking
   for the field that is wrong. */
function locationOf(entry) {
  var loc = entry && entry.loc;
  if (!Array.isArray(loc)) return null;
  var parts = loc
    .filter(function (part, index) {
      if (index === 0 && (part === 'body' || part === 'query' || part === 'path')) {
        return false;
      }
      return part !== null && part !== undefined && part !== '';
    })
    .map(function (part) {
      return String(part);
    });
  return parts.length ? parts.join('.') : null;
}

/*
 * FastAPI's own validation failure is a list: {"detail": [{loc, msg, type}, ...]}. This
 * service normally rewrites that into the same {code, message, field} envelope as every
 * other error, but the raw shape still reaches the browser — a route that 422s before
 * the handlers are reached, or a proxy replaying a stored response — and reading
 * `detail.message` off an array yields undefined and renders a blank red box. Flatten
 * it into a sentence naming the first field, and say how many more there are, because
 * "one of your eleven form fields is wrong" is not an actionable error.
 */
function fromValidationList(entries) {
  var readable = [];
  for (var i = 0; i < entries.length; i += 1) {
    var entry = entries[i];
    if (!entry || typeof entry !== 'object') continue;
    var where = locationOf(entry);
    var msg = typeof entry.msg === 'string' ? entry.msg : 'is not valid';
    readable.push(where ? where + ': ' + msg : msg);
  }
  if (!readable.length) return null;
  var message = readable[0];
  if (readable.length > 1) {
    message += ' (and ' + (readable.length - 1) + ' more field';
    message += readable.length > 2 ? 's)' : ')';
  }
  return { code: 'invalid_value', message: message, field: locationOf(entries[0]) };
}

/*
 * The service answers errors as {"detail": {"code", "message", "field"}} — the envelope
 * is nested under `detail`, because FastAPI's HTTPException wraps it. Reading
 * `body.code` at the top level, which is what the envelope's documentation looks like
 * it promises, yields undefined on every single error and renders "undefined" at the
 * registrar. Unwrap one level, and tolerate every shape that actually arrives.
 */
function unwrap(body) {
  if (!body || typeof body !== 'object') return null;

  var detail = body.detail;

  /* {"detail": [...]}: raw pydantic. */
  if (Array.isArray(detail)) return fromValidationList(detail);

  /* {"detail": {...}}: this service's envelope. */
  if (detail && typeof detail === 'object') {
    if (typeof detail.code === 'string' || typeof detail.message === 'string') {
      var field = typeof detail.field === 'string' ? detail.field : null;
      /* A 422 from this service carries `errors` beside the envelope; when the envelope
         did not name a field, pydantic's list still can. */
      if (!field && Array.isArray(body.errors) && body.errors.length) {
        field = locationOf(body.errors[0]);
      }
      return { code: detail.code || null, message: detail.message || null, field: field };
    }
    return null;
  }

  /* {"detail": "Not Found"}: Starlette's bare string. */
  if (typeof detail === 'string' && detail) {
    return { code: null, message: detail, field: null };
  }

  /* A body that is the envelope itself, unwrapped. */
  if (typeof body.code === 'string' || typeof body.message === 'string') {
    return { code: body.code || null, message: body.message || null, field: body.field || null };
  }

  if (Array.isArray(body.errors) && body.errors.length) {
    return fromValidationList(body.errors);
  }

  return null;
}

function kindFor(status, envelope) {
  if (status === 401) return 'unauthorized';
  if (status === 403) return 'forbidden';
  if (status === 413) return 'too_large';
  if (status === 410) return 'gone';
  return envelope ? 'api' : 'http';
}

function fallbackMessage(status, statusText) {
  if (status === 401) return 'The API key was not accepted. Enter it again.';
  if (status === 403) return 'This key does not have the scope for that action.';
  if (status === 404) return 'Not found.';
  if (status === 410) return 'That preview has expired. Upload the file again.';
  if (status === 413) return 'The file is larger than the service will accept.';
  if (status === 422) return 'The service rejected these values.';
  if (status >= 500) return 'The service failed while handling the request (' + status + ').';
  return 'Request failed (' + status + (statusText ? ' ' + statusText : '') + ').';
}

function request(path, options) {
  var opts = options || {};
  /* `absolute` exists for `/health`, which is the one route outside the versioned
     contract. Every other caller omits it and gets /v1 prepended, which is why no page
     in the app contains a base-path literal. */
  var url = (opts.absolute ? '' : BASE) + path + query(opts.query);
  var headers = { Accept: 'application/json' };
  /* Unconditional: getKey() falls back to the development key, so there is no state in
     which a request leaves without the header and 401s for a reason the page cannot
     explain. */
  headers['X-API-Key'] = getKey();
  /* Conditional, unlike the key above, because there is no sensible default school. A
     service holding one school ignores this; a service holding several refuses a request
     that names none, which is what makes "I forgot to pick a school" a visible error
     rather than one branch's data appearing under another's name. */
  var school = schoolCode();
  if (school) headers['X-School-Code'] = school;

  var init = { method: opts.method || 'GET', headers: headers };

  if (opts.form !== undefined && opts.form !== null) {
    /* No Content-Type here, on purpose. The browser must set it so it can append the
       `boundary=` parameter, and the service parses that boundary out of the header to
       split the upload. Setting 'multipart/form-data' by hand produces a header with
       no boundary and every upload fails as a malformed body. */
    init.body = opts.form;
  } else if (opts.body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(opts.body);
  }
  if (opts.signal) init.signal = opts.signal;

  return fetch(url, init).then(
    function (response) {
      return response.text().then(function (text) {
        var body = null;
        if (text) {
          try {
            body = JSON.parse(text);
          } catch (e) {
            body = null; // A proxy's HTML error page, not the service's envelope.
          }
        }

        if (response.ok) {
          return body;
        }

        var envelope = unwrap(body);
        /* 413 is mapped by status before the body is consulted: a reverse proxy in
           front of the service refuses an oversized upload itself and answers HTML,
           so trusting the envelope alone would report that case as an unreadable
           server error instead of "the file is too big". */
        var kind = kindFor(response.status, envelope);
        throw ApiError(
          kind,
          response.status,
          envelope && envelope.code ? envelope.code : null,
          envelope && envelope.message
            ? envelope.message
            : fallbackMessage(response.status, response.statusText),
          envelope ? envelope.field : null,
          body
        );
      });
    },
    function (cause) {
      if (cause && cause.name === 'AbortError') {
        throw ApiError('aborted', 0, 'aborted', 'Request cancelled.');
      }
      /* fetch rejects only for transport failures — the service being down, DNS, TLS,
         the laptop being off the school network. Reported as its own kind because the
         answer is "try again", never "your key is wrong". */
      throw ApiError(
        'network',
        0,
        'network_error',
        'Could not reach the service. Check that it is running and that this machine is on the school network.'
      );
    }
  );
}

function get(path, params) {
  return request(path, { query: params });
}

function post(path, body, params) {
  return request(path, { method: 'POST', body: body, query: params });
}

function postForm(path, form) {
  /* Caught here rather than at the service: a plain object serialises to
     "[object FormData]"-shaped nonsense that arrives as an unreadable multipart body,
     and the 422 that comes back blames the file. */
  if (!(form instanceof window.FormData)) {
    return Promise.reject(
      ApiError('client', 0, 'invalid_form', 'An upload must be sent as a FormData.')
    );
  }
  return request(path, { method: 'POST', form: form });
}

/*
 * Named endpoints, so a typo'd path fails here in one place rather than in whichever
 * screen happens to use it.
 *
 * This list is the whole API. Every entry below is a route that exists and is covered
 * by the end-to-end run; there is nothing here for a screen that "ought to" have a
 * route — no delete, no update, no list-students-by-year, no per-batch retry. A page
 * that needs one of those leaves the control out rather than calling a path that 404s.
 */
var api = {
  /* -- Schools: the outermost scope ------------------------------------------------
   *
   * Everything below a school belongs to exactly one, reached through a year or a rung.
   * A student is the exception — a child is a person, so she is found by number and her
   * school follows from where she is placed.
   */
  schools: function (includeInactive) {
    return get('/schools', {
      include_inactive: includeInactive ? 'true' : null
    });
  },
  createSchool: function (body) {
    return post('/schools', body);
  },
  /*
   * One school's ladder, grouped by stage. Per school and not global: "Year 1" exists at
   * every branch, so a rung code alone does not identify a rung.
   */
  schoolLevels: function (schoolCode) {
    return get('/schools/' + encodeURIComponent(schoolCode) + '/levels');
  },
  createLevel: function (body) {
    return post('/structure/levels', body);
  },

  createAcademicYear: function (body) {
    return post('/academic-years', body);
  },
  /*
   * `school` narrows both lists. Omitted, the years of every school come back and the
   * ladder comes back empty — because two schools' rungs merged into one list is a list
   * with two `Y1` rows in it and nothing to tell them apart.
   */
  years: function (schoolCode) {
    return get('/structure/years', { school: schoolCode });
  },
  classes: function (academicYear, yearLevel) {
    return get('/structure/classes', { academic_year: academicYear, year_level: yearLevel });
  },
  generateStructure: function (body) {
    return post('/structure/generate', body);
  },

  terms: function (academicYear) {
    return get('/terms', { academic_year: academicYear });
  },
  createTerm: function (body) {
    return post('/terms', body);
  },
  /*
   * The catalogue of one year. `academicYear` is required by the route, not defaulted:
   * a subject belongs to a year, so "the subjects" has no answer, and a client that
   * guessed the current year would quietly show next September's catalogue to a
   * registrar working through last term's marks.
   */
  subjects: function (academicYear, includeInactive) {
    return get('/subjects', {
      academic_year: academicYear,
      include_inactive: includeInactive ? 'true' : null
    });
  },
  createSubject: function (body) {
    return post('/subjects', body);
  },

  /* -- One class at a time ---------------------------------------------------------
   *
   * The generator builds a whole ladder and is right for September. These two are the
   * rest of the year: the extra section opened in November, and a label corrected.
   */
  createClassSection: function (body) {
    return post('/structure/classes', body);
  },
  renameClassSection: function (classCode, academicYear, body) {
    return request('/structure/classes/' + encodeURIComponent(classCode), {
      method: 'PATCH',
      body: body,
      query: { academic_year: academicYear }
    });
  },

  /* -- One child ------------------------------------------------------------------
   *
   * Every write here used to require a spreadsheet. These are the direct paths for the
   * two things that actually fill a registrar's day — a misspelt name, and one child
   * arriving in November — and the import still owns anything touching many children.
   */
  searchStudents: function (query, includeInactive) {
    return get('/students', {
      q: query,
      include_inactive: includeInactive ? 'true' : null
    });
  },
  student: function (studentNumber) {
    return get('/students/' + encodeURIComponent(studentNumber));
  },
  saveStudent: function (body) {
    return post('/students', body);
  },
  updateStudent: function (studentNumber, body) {
    return request('/students/' + encodeURIComponent(studentNumber), {
      method: 'PATCH',
      body: body
    });
  },

  /* -- Placement, which is a dated membership and never a column -------------------
   *
   * There is deliberately no `moveStudent` that takes a new class and nothing else.
   * `transferStudent` closes the open placement and opens the next one in a single
   * request, because between two separate calls the child is in no class at all and a
   * marks upload landing in that window rejects every one of her rows.
   */
  studentPlacements: function (studentNumber) {
    return get('/students/' + encodeURIComponent(studentNumber) + '/placements');
  },
  placeStudent: function (studentNumber, body) {
    return post('/students/' + encodeURIComponent(studentNumber) + '/placements', body);
  },
  transferStudent: function (studentNumber, body) {
    return post('/students/' + encodeURIComponent(studentNumber) + '/transfer', body);
  },
  endPlacement: function (studentNumber, body) {
    return request(
      '/students/' + encodeURIComponent(studentNumber) + '/placements/current',
      { method: 'PATCH', body: body }
    );
  },

  classRoster: function (classCode, academicYear) {
    return get('/classes/' + encodeURIComponent(classCode) + '/students', {
      academic_year: academicYear
    });
  },
  studentGrades: function (studentNumber, termCode) {
    return get('/students/' + encodeURIComponent(studentNumber) + '/grades', {
      term: termCode
    });
  },

  previewRoster: function (form) {
    return postForm('/imports/roster/preview', form);
  },
  commitRoster: function (batchId) {
    return post('/imports/roster/' + encodeURIComponent(batchId) + '/commit');
  },
  previewGuardians: function (form) {
    return postForm('/imports/guardians/preview', form);
  },
  commitGuardians: function (batchId) {
    return post('/imports/guardians/' + encodeURIComponent(batchId) + '/commit');
  },
  studentGuardians: function (studentNumber) {
    return get('/students/' + encodeURIComponent(studentNumber) + '/guardians');
  },
  /*
   * The reverse lookup, and the one route in this list that answers a question a parent
   * asks rather than one a registrar asks: which children may this number be told about.
   * `includeRestricted` is off by default so the answer a caller gets without asking for
   * more is the answer that is safe to read aloud on the phone.
   */
  guardianChildren: function (phone, includeRestricted) {
    return get('/guardians/' + encodeURIComponent(phone) + '/students', {
      include_restricted: includeRestricted ? 'true' : null
    });
  },
  /*
   * Grant or revoke one guardian's sight of one child's records. PATCH, not PUT: the
   * link's relationship and its name are the import's to state, and this route changes
   * exactly one flag on it.
   */
  setRecordsAccess: function (studentNumber, phone, body) {
    return request(
      '/students/' +
        encodeURIComponent(studentNumber) +
        '/guardians/' +
        encodeURIComponent(phone),
      { method: 'PATCH', body: body }
    );
  },
  /*
   * Remove the link entirely — the adult was entered against the wrong child. Distinct
   * from revoking access, which leaves a true relationship on file that simply may not
   * read the marks; a court order is the second, a typo is this one.
   */
  unlinkGuardian: function (studentNumber, phone) {
    return request(
      '/students/' +
        encodeURIComponent(studentNumber) +
        '/guardians/' +
        encodeURIComponent(phone),
      { method: 'DELETE' }
    );
  },
  previewGrades: function (form) {
    return postForm('/imports/grades/preview', form);
  },
  commitGrades: function (batchId) {
    return post('/imports/grades/' + encodeURIComponent(batchId) + '/commit');
  },
  importReport: function (batchId, params) {
    return get('/imports/' + encodeURIComponent(batchId), params);
  },

  createApiKey: function (body) {
    return post('/admin/api-keys', body);
  },

  /* -- The daily register ----------------------------------------------------------
   *
   * `classRegister` returns every child placed in the class that day, and the ones nobody
   * marked come back with `state: null`. That null is a third value beside present and
   * absent, and the screens treat it as one: rendering it as either would state a fact the
   * school did not.
   */
  classRegister: function (classCode, academicYear, onDate) {
    return get('/classes/' + encodeURIComponent(classCode) + '/attendance', {
      academic_year: academicYear,
      on: onDate
    });
  },
  /*
   * PUT, not POST: the request states what the register *is* for that day, so saving the
   * same morning twice corrects it rather than writing a second set of marks beside the
   * first. Children left out of `entries` are untouched, which is what lets a teacher
   * save the twelve present so far and finish later.
   */
  takeRegister: function (classCode, academicYear, onDate, entries) {
    return request('/classes/' + encodeURIComponent(classCode) + '/attendance', {
      method: 'PUT',
      body: { entries: entries },
      query: { academic_year: academicYear, on: onDate }
    });
  },
  studentAttendance: function (studentNumber, fromDate, toDate) {
    return get('/students/' + encodeURIComponent(studentNumber) + '/attendance', {
      from: fromDate,
      to: toDate
    });
  },

  /*
   * Liveness, and the only entry here that is not under /v1. `/health` is the process
   * saying it is running, which is a statement about the deployment rather than about
   * the school's data, so it sits outside the versioned contract and skips the base path.
   *
   * The shell polls this to colour the status dot in the footer. Deliberately not used
   * to gate rendering: a health check that is slow or wrong must not be able to blank a
   * screen whose data has already loaded.
   */
  health: function () {
    return request('/health', { absolute: true });
  }
};

/*
 * Named exports rather than one object on `window`. The reason the old build used a global
 * was that it had no module system; with one, an import is checked at build time — a typo in
 * `SIS.gradeTxet` used to be `undefined` at runtime on one screen.
 */
export { BASE, DEFAULT_KEY, ApiError, getKey, setKey, clearKey, hasKey, isDefaultKey };
export { escapeText as escape, gradeText, request, get, post, postForm, api };
