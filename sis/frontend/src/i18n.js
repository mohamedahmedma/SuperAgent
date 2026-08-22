/*
 * Translation, keyed by the English string itself.
 *
 * `t('Add school')` returns 'Add school' in English and the Arabic for it in Arabic. There is
 * no key namespace, no `screens.school.actions.add`, and that is the decision this file is
 * really about.
 *
 * Keys of that kind are the usual advice and they cost more than they are worth in a console
 * this size. They put the English in a second file, so reading a screen tells you the shape of
 * the page and not one word of what it says; they go stale silently, because a key that no
 * longer exists renders as `screens.school.actions.add` in production and as nothing in review;
 * and they make every edit two edits. Using the source string as the key means the code still
 * reads as English prose, a missing translation degrades to English rather than to a key, and
 * `ar.js` can be handed to a translator as a list of sentences.
 *
 * What it costs, stated plainly: two identical English strings that need different Arabic
 * cannot be told apart. That has not happened yet; when it does, the fix is a disambiguating
 * suffix on the key — `t('Open|the class')` — and a split on the bar. It is not worth building
 * before it is needed.
 *
 * -- Interpolation ------------------------------------------------------------------
 *
 * A sentence with something in the middle of it — a class code, a count, a link — is one
 * string with `{0}` in it, never two strings glued around a value:
 *
 *     t('Nobody is in {0} yet', [<span className="sis-code">3A</span>])
 *
 * Gluing is the classic i18n bug and Arabic is exactly where it bites. Concatenating
 * `'Nobody is in ' + code + ' yet'` forces the translator to accept the English word order,
 * and in a right-to-left sentence the fragments end up in an order nobody chose. One string
 * with a hole in it lets the Arabic put the hole wherever Arabic puts it.
 *
 * When `parts` is given the result is an array of nodes rather than a string, so it drops
 * straight into JSX. Without `parts` it is a plain string, which is what an attribute needs.
 */
import { AR } from './locale/ar.js';

/*
 * The current language, held here rather than read from the store on every call.
 *
 * `t()` runs hundreds of times per render — every label in a four-hundred-row register — and a
 * store lookup in each is a lookup nobody needs. `Store.setLang` pushes the value in through
 * `setLocale` instead, and the App root re-renders on the same change, so the two never
 * disagree by more than the instant between them.
 */
let current = 'en';

const TABLES = { ar: AR };

/** Called by the store whenever the language changes, and once at boot. */
export function setLocale(lang) {
  current = lang === 'ar' ? 'ar' : 'en';
}

export function locale() {
  return current;
}

/**
 * Translate, and fill in the holes.
 *
 * @param {string} text    the English source string, which is also its key
 * @param {Array}  [parts] values for `{0}`, `{1}`, … — strings or JSX nodes
 * @returns {string|Array} a string when `parts` is absent, otherwise an array of nodes
 */
export function t(text, parts) {
  const table = TABLES[current];
  /* A missing entry falls back to the English rather than to a key or an empty string. A
     half-translated console reads as a console with some English in it, which is honest;
     a half-translated console full of `screens.school.title` reads as broken. */
  const line = (table && table[text]) || text;

  if (!parts || !parts.length) return line;

  /* Split on the holes and interleave. `key` on every node because this returns an array and
     React asks for one, and the index is stable within a single string. */
  const out = [];
  const pieces = String(line).split(/(\{\d+\})/);
  pieces.forEach((piece, index) => {
    const hole = /^\{(\d+)\}$/.exec(piece);
    if (hole) {
      const value = parts[Number(hole[1])];
      out.push(
        value && typeof value === 'object' && 'type' in value
          ? { ...value, key: `p${index}` }
          : value
      );
    } else if (piece) {
      out.push(piece);
    }
  });
  return out;
}

/**
 * Pick the field a record stores in the reading language, falling back to the other one.
 *
 * Here rather than in hooks.js because it is the same decision `t` makes, applied to data
 * instead of to chrome: show what the reader asked for, and show the other thing rather than
 * nothing when it is missing. A child with no Arabic name recorded is not a child with no name.
 */
export function pick(record, en, ar) {
  if (!record) return '';
  return (current === 'ar' ? record[ar] || record[en] : record[en] || record[ar]) || '';
}
