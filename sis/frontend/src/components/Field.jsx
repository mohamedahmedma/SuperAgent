/*
 * The two form controls this console has that Bootstrap does not: a dropdown and a search box.
 *
 * They are one component in two halves on purpose. A registrar's screen is mostly a toolbar of
 * pickers and a box to type a name into, and when those two are a `<select>` and an `<input>`
 * they are visibly different objects — different heights by a pixel or two, different corner
 * radii, a native arrow that ignores the page's colours and sits on the wrong side of an RTL
 * field. Sharing one shell (`.sis-field`) means they line up because they cannot not line up.
 *
 * -- Why the native `<select>` had to go ------------------------------------------
 *
 * The comment this replaced argued for it, and two of its three reasons were sound: the native
 * control brings its own type-ahead, and on a phone it opens the platform picker. What it could
 * not do is anything else. A native option list takes no styling worth having, so in a console
 * whose whole palette is configurable the one control a registrar touches most stayed the
 * operating system's grey. It cannot show a second line, so "3A" and "Year 3, morning" could
 * never appear together. Its arrow is painted at the right-hand edge by a background image, so
 * in Arabic it sat on the wrong side of every field on the screen. And it cannot be filtered,
 * so picking one class out of forty meant type-ahead or scrolling.
 *
 * So the type-ahead is reimplemented here (see `jumpTo`), the filter box is the thing native
 * selects cannot have at all, and the phone case is answered by making the menu large enough to
 * hit rather than by handing the job to the platform.
 *
 * -- What both halves share --------------------------------------------------------
 *
 * `.sis-field` is the shell: height, padding, radius, border, focus ring, disabled state. The
 * dropdown puts a label and a chevron in it; the search box puts an icon and an input in it.
 * Neither one sets a colour of its own — every value comes from the tokens, so both follow the
 * page colour with everything else.
 */
import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { Icon, cx } from './Ui.jsx';
import { t } from '../i18n.js';

/*
 * Above this many options the menu grows a filter box.
 *
 * A threshold rather than always-on. A filter over three academic years is a text box asking a
 * registrar to type instead of point, and it takes the keyboard focus away from the list it is
 * supposed to help with. Above about eight the list stops being scannable and the box earns its
 * place — a class picker in a real school holds forty.
 */
const FILTER_FROM = 8;

/** Does this string contain the query, ignoring case and surrounding space? */
function matches(option, query) {
  if (!query) return true;
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return (
    String(option.label || '').toLowerCase().includes(needle) ||
    String(option.value || '').toLowerCase().includes(needle) ||
    String(option.note || '').toLowerCase().includes(needle)
  );
}

/* -- The dropdown ------------------------------------------------------------------ */

/**
 * A listbox dropdown. Same props the native `<select>` wrapper took, so every call site kept
 * working when this replaced it: `value`, `options` of `{value, label, note}`, `placeholder`,
 * `disabled`, `error`, `onChange`.
 *
 * Keyboard behaviour is the ARIA listbox pattern rather than an approximation of it: Up/Down
 * move the active option, Home/End jump to the ends, Enter and Space choose, Escape closes
 * without choosing, Tab closes and moves on. Typing a letter jumps to the next option starting
 * with it, which is the one behaviour the native control had that a registrar would miss.
 */
export function Select({
  value,
  options = [],
  placeholder,
  strict,
  disabled,
  error,
  className,
  size,
  onChange
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(-1);
  const host = useRef(null);
  const filterBox = useRef(null);
  const listBox = useRef(null);
  const listId = useId();

  const chosen = options.find((option) => String(option.value) === String(value));
  const filtering = options.length >= FILTER_FROM;
  const shown = useMemo(
    () => (filtering ? options.filter((option) => matches(option, query)) : options),
    [options, query, filtering]
  );

  /* Close on a click anywhere else, and on the window scrolling out from under the menu.
     Bound only while open, so a page with a dozen pickers on it carries no idle listeners. */
  useEffect(() => {
    if (!open) return undefined;
    function onDown(event) {
      if (host.current && !host.current.contains(event.target)) setOpen(false);
    }
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  /* Opening puts the caret in the filter box when there is one, and otherwise puts the active
     option on the current value, so Down moves from where you are rather than from the top. */
  useEffect(() => {
    if (!open) {
      setQuery('');
      return;
    }
    const at = options.findIndex((option) => String(option.value) === String(value));
    setActive(at);
    if (filtering && filterBox.current) filterBox.current.focus();
  }, [open]);

  /* Keep the active option in view. `block: 'nearest'` so choosing with the keyboard scrolls
     the menu by one line rather than jumping the option to the middle every time. */
  useEffect(() => {
    if (!open || !listBox.current) return;
    const node = listBox.current.querySelector('[data-active="true"]');
    if (node && node.scrollIntoView) node.scrollIntoView({ block: 'nearest' });
  }, [active, open]);

  function choose(option) {
    setOpen(false);
    if (onChange) onChange(option.value);
  }

  function move(step) {
    if (!shown.length) return;
    const at = shown.findIndex((option) => option === shown[active]);
    const from = at < 0 ? (step > 0 ? -1 : shown.length) : at;
    const next = Math.min(shown.length - 1, Math.max(0, from + step));
    setActive(next);
  }

  /* Type-ahead: the one thing the native control did that this had to keep. */
  function jumpTo(letter) {
    const at = shown.findIndex(
      (option, index) =>
        index > active && String(option.label || '').toLowerCase().startsWith(letter)
    );
    const wrapped =
      at >= 0
        ? at
        : shown.findIndex((option) =>
            String(option.label || '').toLowerCase().startsWith(letter)
          );
    if (wrapped >= 0) setActive(wrapped);
  }

  function onKeyDown(event) {
    if (disabled) return;
    const key = event.key;

    if (!open) {
      if (key === 'ArrowDown' || key === 'ArrowUp' || key === 'Enter' || key === ' ') {
        event.preventDefault();
        setOpen(true);
      }
      return;
    }

    if (key === 'Escape') {
      event.preventDefault();
      setOpen(false);
      return;
    }
    if (key === 'Tab') {
      setOpen(false);
      return;
    }
    if (key === 'ArrowDown') {
      event.preventDefault();
      move(1);
      return;
    }
    if (key === 'ArrowUp') {
      event.preventDefault();
      move(-1);
      return;
    }
    if (key === 'Home') {
      event.preventDefault();
      setActive(0);
      return;
    }
    if (key === 'End') {
      event.preventDefault();
      setActive(shown.length - 1);
      return;
    }
    if (key === 'Enter' || (key === ' ' && !filtering)) {
      event.preventDefault();
      if (shown[active]) choose(shown[active]);
      return;
    }
    if (!filtering && key.length === 1 && /\S/.test(key)) {
      jumpTo(key.toLowerCase());
    }
  }

  const label = chosen ? chosen.label : placeholder || t('Choose…');

  return (
    <div className={cx('sis-field-host', className)} ref={host}>
      <button
        type="button"
        className={cx(
          'sis-field sis-field-trigger',
          size === 'sm' && 'sis-field-sm',
          open && 'is-open',
          error && 'is-invalid',
          !chosen && 'is-empty'
        )}
        disabled={disabled || false}
        aria-haspopup="listbox"
        aria-expanded={open ? 'true' : 'false'}
        aria-controls={open ? listId : undefined}
        onClick={() => !disabled && setOpen(!open)}
        onKeyDown={onKeyDown}
      >
        <span className="sis-field-value">{label}</span>
        <Icon name="chevron" size={14} />
      </button>

      {open ? (
        <div className="sis-menu" role="presentation">
          {filtering ? (
            <div className="sis-menu-filter">
              <Icon name="search" size={14} />
              <input
                ref={filterBox}
                type="text"
                className="sis-menu-filter-input"
                value={query}
                placeholder={t('Filter…')}
                autoComplete="off"
                spellCheck={false}
                aria-label={t('Filter the list')}
                onChange={(event) => {
                  setQuery(event.target.value);
                  setActive(0);
                }}
                onKeyDown={onKeyDown}
              />
            </div>
          ) : null}

          <ul className="sis-menu-list" role="listbox" id={listId} ref={listBox} tabIndex={-1}>
            {shown.length ? (
              shown.map((option, index) => {
                const isChosen = String(option.value) === String(value);
                return (
                  <li key={option.value} role="presentation">
                    <button
                      type="button"
                      role="option"
                      aria-selected={isChosen ? 'true' : 'false'}
                      data-active={index === active ? 'true' : 'false'}
                      className={cx('sis-menu-option', isChosen && 'is-chosen')}
                      /* Pointer moves set the active option so the keyboard and the mouse
                         cannot disagree about which row is highlighted. */
                      onMouseEnter={() => setActive(index)}
                      onClick={() => choose(option)}
                    >
                      <span className="sis-menu-option-body">
                        <span className="sis-menu-option-label">{option.label}</span>
                        {option.note ? (
                          <span className="sis-menu-option-note">{option.note}</span>
                        ) : null}
                      </span>
                      {isChosen ? <Icon name="check" size={14} /> : null}
                    </button>
                  </li>
                );
              })
            ) : (
              <li className="sis-menu-empty">{t('Nothing matches')}</li>
            )}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

/* -- The search box ---------------------------------------------------------------- */

/**
 * The same shell with an input in it instead of a menu.
 *
 * It exists so that a toolbar of a picker and a search box reads as one row of controls rather
 * than as two things that happen to be side by side. Everything visual comes from `.sis-field`,
 * so a change to the field height or the focus ring moves both and cannot move only one.
 *
 * The clear button appears only when there is something to clear, and it is a real button
 * rather than `type="search"`: the native clear affordance is a different shape in every
 * browser, absent in some, and unstyleable in all of them.
 */
export function SearchField({
  value,
  placeholder,
  disabled,
  error,
  className,
  size,
  inputMode,
  onInput,
  onKeyDown
}) {
  return (
    <div className={cx('sis-field-host', className)}>
      <div
        className={cx(
          'sis-field sis-field-search',
          size === 'sm' && 'sis-field-sm',
          error && 'is-invalid'
        )}
      >
        <Icon name="search" size={14} />
        <input
          type="text"
          className="sis-field-input"
          value={value ?? ''}
          placeholder={placeholder || ''}
          disabled={disabled || false}
          inputMode={inputMode}
          autoComplete="off"
          spellCheck={false}
          onChange={(event) => onInput && onInput(event.target.value)}
          onKeyDown={onKeyDown}
        />
        {value ? (
          <button
            type="button"
            className="sis-field-clear"
            aria-label={t('Clear')}
            onClick={() => onInput && onInput('')}
          >
            <Icon name="close" size={13} />
          </button>
        ) : null}
      </div>
    </div>
  );
}
