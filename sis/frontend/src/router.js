/*
 * The router: hash-based, deliberately.
 *
 * The console is served by `StaticFiles` from `sis/web/`, which knows nothing about
 * client routes. With History-API paths, `/ui/imports` would be a 404 on reload and on
 * every link a registrar pastes into an email — the server would have to learn to rewrite
 * unknown paths to index.html, which means the UI could no longer be a directory of files
 * the service serves without configuration. The hash never reaches the server, so
 * `/ui/#/imports?batch=b-12` reloads, bookmarks and shares correctly with no server-side
 * story at all.
 *
 * A route is a name and a view component. Query parameters live in the hash after `?` and
 * are the *only* place screen state is kept that must survive a reload: the batch a
 * registrar is reading, the student number they looked up. Everything else is component
 * state, and is meant to be forgotten.
 *
 * There is no route literal beginning with `/` anywhere in here. `#/structure` is a
 * fragment, not a path, and writing it as `'/structure'` would make the contract suite
 * read it as a call to a route the service does not serve — which is exactly the class of
 * mistake that suite exists to catch.
 */

var routes = [];
var listeners = [];
var current = null;

/** Parse `#/name?a=1&b=2` into `{name, params}`. Anything unrecognised is the home route. */
function parse(hash) {
  var raw = String(hash || '').replace(/^#\/?/, '');
  var split = raw.indexOf('?');
  var name = (split === -1 ? raw : raw.slice(0, split)).replace(/\/+$/, '');
  var search = split === -1 ? '' : raw.slice(split + 1);

  var params = {};
  if (search) {
    new window.URLSearchParams(search).forEach(function (value, key) {
      params[key] = value;
    });
  }
  return { name: name || routes[0].name, params: params };
}

function find(name) {
  for (var i = 0; i < routes.length; i += 1) {
    if (routes[i].name === name) return routes[i];
  }
  return null;
}

function resolve() {
  var parsed = parse(window.location.hash);
  var route = find(parsed.name);

  /*
   * An unknown route redirects home rather than rendering a 404 screen. A registrar
   * only reaches one by editing the address bar or following a link from a version of
   * the console that had a screen this one does not, and in both cases the dashboard is
   * a more useful answer than an apology.
   */
  if (!route) {
    replace(routes[0].name);
    return;
  }

  current = { route: route, params: parsed.params };
  notify();
}

function notify() {
  listeners.slice().forEach(function (fn) {
    fn(current);
  });
}

/** Build the href for a route. The one place `#/` is written. */
function href(name, params) {
  var text = '#/' + name;
  if (params) {
    var search = new window.URLSearchParams();
    Object.keys(params).forEach(function (key) {
      var value = params[key];
      if (value !== null && value !== undefined && value !== '') search.append(key, value);
    });
    var query = search.toString();
    if (query) text += '?' + query;
  }
  return text;
}

/** Navigate, adding a history entry — a click on a nav link. */
function go(name, params) {
  window.location.hash = href(name, params);
}

/*
 * Navigate *without* adding a history entry. Used when a screen records what it is
 * showing in the URL — the batch just previewed, the student just looked up. Those are
 * refinements of the current screen, and pushing each one would mean Back walks
 * backwards through a registrar's typing instead of returning to where they came from.
 */
function replace(name, params) {
  var url = window.location.pathname + window.location.search + href(name, params);
  if (window.history && window.history.replaceState) {
    window.history.replaceState(null, '', url);
    resolve();
  } else {
    window.location.hash = href(name, params);
  }
}

/** Merge parameters into the current route without touching history. */
function setParams(patch) {
  if (!current) return;
  var merged = {};
  Object.keys(current.params).forEach(function (key) {
    merged[key] = current.params[key];
  });
  Object.keys(patch).forEach(function (key) {
    var value = patch[key];
    if (value === null || value === undefined || value === '') {
      delete merged[key];
    } else {
      merged[key] = value;
    }
  });
  replace(current.route.name, merged);
}

function subscribe(fn) {
  listeners.push(fn);
  return function () {
    listeners = listeners.filter(function (item) {
      return item !== fn;
    });
  };
}

/**
 * Register the route table and start listening. Called once, from main.js, with the
 * home route first — `routes[0]` is what an unknown hash falls back to.
 */
function start(table) {
  routes = table.slice();
  window.addEventListener('hashchange', function () {
    resolve();
    /*
     * Scroll to the top on a route change, and only on a route change. A screen that
     * writes a parameter into the hash (a batch id, a looked-up student) calls
     * `setParams`, which goes through replaceState and never fires hashchange — so
     * reading a rejected row four hundred lines down does not throw the registrar back
     * to the page heading.
     */
    window.scrollTo(0, 0);
  });
  resolve();
}

export const Router = {
  start: start,
  subscribe: subscribe,
  href: href,
  go: go,
  replace: replace,
  setParams: setParams,
  get current() {
    return current;
  }
};
