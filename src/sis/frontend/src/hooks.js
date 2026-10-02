/*
 * The five hooks every screen is built from, and the formatters they share.
 *
 * Logic only — no markup, so this file survived the move from Preact-with-htm to React-with-
 * JSX unchanged apart from the import line. That is not luck: the caching, the request
 * cancellation and the form diffing were never framework-specific, and keeping them out of
 * the component file is what made a framework swap a mechanical job rather than a rewrite.
 *
 * `useResource` and `useQuery` are two hooks rather than one on purpose, and the distinction
 * is the most important thing here:
 *
 *   useResource  cached, shared, stale-while-revalidate. For the slow-moving skeleton of the
 *                school — schools, years, rungs, classes, terms, subjects. Two components
 *                asking for the same key produce one request.
 *
 *   useQuery     never cached, cancellable, refetched on dependency change. For the thing the
 *                registrar came to look at — a register, a child's marks, an import report.
 *                Serving yesterday's register out of a cache is the one failure this design
 *                must not introduce.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Store } from './store.js';

/** Re-render this component whenever the shared store changes. */
export function useStore() {
  const [state, setState] = useState(Store.state);
  useEffect(() => Store.subscribe(setState), []);
  return state;
}

/**
 * Read a cached server value.
 *
 * Returns `{value, error, loading, ready, reload}`. `ready` rather than `!loading` is what a
 * screen should branch on: during a stale-while-revalidate refresh both are true, and the
 * screen must keep drawing the rows it already has.
 *
 * `enabled` exists for the year- and school-scoped reads. The selected year is not known
 * until the year list has loaded, so a screen that asks for "the classes of the selected
 * year" during its first render asks for the classes of no year at all — the service answers,
 * the year settles, the key changes, and the same data is fetched again under the right key.
 * Passing `!!year` makes the screen wait until the question is answerable.
 */
export function useResource(key, loader, enabled = true) {
  const [snapshot, setSnapshot] = useState(() => Store.peek(key));

  useEffect(() => {
    if (!enabled) return undefined;
    setSnapshot(Store.peek(key));
    const stop = Store.watch(key, setSnapshot);
    Store.register(key, loader);
    Store.read(key, loader).catch(() => {
      /* The failure is already on the snapshot the watcher delivered. Swallowed so an
         expected error — an offline service — is not also an unhandled rejection, which is
         noise that hides real bugs. */
    });
    return stop;
  }, [key, enabled]);

  const reload = useCallback(
    () => Store.read(key, loader, { force: true }).catch(() => {}),
    [key]
  );

  return { ...snapshot, reload };
}

/**
 * A one-shot server read that is NOT cached.
 *
 * The generation counter is the point of this hook. A registrar who types a student number,
 * waits, then types another has two requests in flight; without the counter the slower one
 * wins whenever it lands second, and the screen shows the marks of a child whose number is no
 * longer in the box. Late answers are dropped.
 */
export function useQuery(run, deps, enabled = true) {
  const [snapshot, setSnapshot] = useState({
    value: undefined,
    error: null,
    loading: false
  });
  const generation = useRef(0);

  const reload = useCallback(() => {
    generation.current += 1;
    const mine = generation.current;
    setSnapshot({ value: undefined, error: null, loading: true });
    return Store.track(run()).then(
      (value) => {
        if (mine === generation.current) {
          setSnapshot({ value, error: null, loading: false });
        }
        return value;
      },
      (error) => {
        if (mine === generation.current) {
          setSnapshot({ value: undefined, error, loading: false });
        }
        throw error;
      }
    );
  }, deps);

  useEffect(() => {
    if (!enabled) {
      generation.current += 1; // Cancels any answer still on its way.
      setSnapshot({ value: undefined, error: null, loading: false });
      return undefined;
    }
    reload().catch(() => {});
    return undefined;
  }, [...deps, enabled]);

  return { ...snapshot, ready: snapshot.value !== undefined, reload };
}

/**
 * A write. Returns `{run, pending, error, reset}` and guarantees `pending` is a truthful
 * answer to "is this button doing something".
 *
 * `pending` is what makes a double-clicked commit safe on the client side — the service
 * refuses the second one anyway, but a registrar who clicks twice should see a disabled
 * button, not an error about a batch that was already applied.
 */
export function useAction(fn) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const alive = useRef(true);

  useEffect(() => () => {
    alive.current = false;
  }, []);

  const run = useCallback(
    (...args) => {
      setError(null);
      setPending(true);
      return Store.track(Promise.resolve().then(() => fn(...args))).then(
        (value) => {
          if (alive.current) setPending(false);
          return value;
        },
        (failure) => {
          /* The error is held on the action, next to the control that caused it. An error in
             a toast is read after the registrar has already started retyping the form. */
          if (alive.current) {
            setPending(false);
            setError(failure);
          }
          throw failure;
        }
      );
    },
    [fn]
  );

  return { run, pending, error, reset: () => setError(null) };
}

/**
 * A form: values, a setter per field, and the field-level error the service named.
 *
 * `errorFor` is the part worth having. The error envelope carries `field`, so a 422 about
 * `starts_on` lands under the start-date box instead of in a banner that makes the registrar
 * re-read a form they have already checked twice.
 *
 * **`changed` and `delta` are two readings of one diff, and they are not interchangeable.**
 *
 *   changed()  a list of `{label, was, now}` — what the confirmation dialog shows. This is
 *              what turns "are you sure?", a question nobody can answer, into
 *              "Sara Mohamd → Sara Mohamed", which anyone can.
 *   delta()    a plain object of the changed fields — what a PATCH sends. Only what changed
 *              goes in it, so saving does not overwrite a phone number another registrar
 *              typed between this screen loading and this save.
 *
 * They are stated separately because a caller that sends `changed()` as a request body sends
 * a JSON *array* the service answers 422 to, and renders `{label, was, now}` where a value
 * belongs — which React refuses to draw at all, taking the whole console down with it. Both
 * are derived from `differing()` below rather than from each other, so a change to what
 * counts as edited reaches the dialog and the request together.
 */
export function useForm(initial) {
  const [values, setValues] = useState(initial);

  /* The one comparison. `String(x ?? '')` rather than `!==` because a date input reports an
     empty box as `''` where the record carries `null`, and those are the same fact. */
  function differing(against) {
    const base = against || initial;
    return Object.keys(values).filter(
      (key) => String(values[key] ?? '') !== String(base[key] ?? '')
    );
  }

  return {
    values,
    set: (name) => (value) => setValues((current) => ({ ...current, [name]: value })),
    replace: setValues,
    reset: (to) => setValues(to === undefined ? initial : to),
    changed(against, labels) {
      const base = against || initial;
      return differing(against).map((key) => ({
        label: (labels && labels[key]) || key,
        was: base[key],
        now: values[key]
      }));
    },
    delta(against) {
      const body = {};
      differing(against).forEach((key) => {
        body[key] = values[key];
      });
      return body;
    },
    errorFor(error, name) {
      if (!error || !error.field) return null;
      return error.field === name || error.field.startsWith(`${name}.`)
        ? error.message
        : null;
    }
  };
}

/* ==================================================================================
 * Formatters
 * ================================================================================== */

/** The one rendering of "no mark was recorded". */
export const DASH = '—';

/** Today as `YYYY-MM-DD`, for prefilling a date field. */
export function today() {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  return `${now.getFullYear()}-${month}-${day}`;
}

/**
 * Pick the name to show in the current reading direction, falling back to the other script
 * rather than to a blank. A row with an empty Arabic name must still be identifiable —
 * showing nothing where a name belongs looks like missing data, and a registrar cannot tell
 * that apart from a child who is genuinely not on file.
 */
export function pickName(item, lang) {
  if (!item) return '';
  const ar = item.name_ar || item.full_name_ar || '';
  const en = item.name_en || item.full_name_en || '';
  return lang === 'ar' ? ar || en : en || ar;
}

/** `3A — Grade 3 Falcons`: the code is the identity, the name is the label. */
export function labelOf(item, lang) {
  if (!item) return '';
  const name = pickName(item, lang);
  return name ? `${item.code} — ${name}` : item.code;
}

/**
 * A date as the service states it. No locale reformatting: `2026-03-01` is unambiguous in
 * both reading directions, and a d/m versus m/d guess on a school calendar is a real hazard.
 */
export function dateText(value) {
  return value ? String(value).slice(0, 10) : DASH;
}

export function countText(value) {
  return typeof value === 'number' ? String(value) : DASH;
}
