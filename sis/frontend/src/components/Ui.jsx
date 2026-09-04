/*
 * The component library: Bootstrap markup, mobile first, and no breakpoint of its own.
 *
 * That last clause is the rule this rewrite exists to establish. The console previously had a
 * hand-written component layer with its own media queries and `clamp()` sizing, which meant
 * the responsive behaviour of a screen could not be read from the screen — you had to go and
 * find which stylesheet hid what at which width. Every breakpoint now lives in a class list,
 * next to the element it governs.
 *
 * **Mobile is the default and the desktop is the override**, in that order, because that is
 * the order Bootstrap's breakpoints work in (`col-12 col-md-6`, never the reverse) and
 * because it is the order the users arrive in: most of this console is read on a phone.
 * Three consequences run through everything below.
 *
 *   Buttons stack full width and go inline from `sm` up (`d-grid gap-2 d-sm-flex`), so a
 *   toolbar of three actions is three tappable rows on a phone rather than three squeezed
 *   boxes.
 *
 *   Tables scroll inside `.table-responsive`, and a column may declare the breakpoint below
 *   which it is not worth its width (`hide: 'md'`). A register on a phone shows the number,
 *   the name and the action; placement dates appear when there is room for them.
 *
 *   Forms are a Bootstrap grid, so a five-field form is one column on a phone and three
 *   across on a desktop with no media query anywhere.
 */
import { Fragment, useEffect, useRef, useState } from 'react';
import { Router } from '../router.js';
import { Store } from '../store.js';
import { DASH, countText, useAction, useStore } from '../hooks.js';
import { Icon, cx } from './Icon.jsx';
import { Select, SearchField } from './Field.jsx';
import { t } from '../i18n.js';

/* Re-exported so the fifty call sites that import these from here keep working. `Select`
   and `SearchField` now live in Field.jsx; `Icon` and `cx` in Icon.jsx. */
export { Icon, cx } from './Icon.jsx';
export { Select, SearchField } from './Field.jsx';


/** The utility for "hidden below this breakpoint, table cell at and above it". */
function hiddenBelow(breakpoint) {
  return breakpoint ? `d-none d-${breakpoint}-table-cell` : '';
}

/* ==================================================================================
 * Icons
 *
 * Inline SVG stroked with `currentColor`, so an icon inherits the colour of the control it
 * sits in and needs no dark-theme variant. Fifteen of them on a 24-unit grid. An icon font
 * would be a second network request and a flash of missing glyphs on a school connection.
 * ================================================================================== */

/* ==================================================================================
 * Button
 * ================================================================================== */

/**
 * `variant` maps to a Bootstrap button: primary, danger, quiet (the text button defined
 * through Bootstrap's own `--bs-btn-*` variables in theme.css) or the default outline.
 *
 * `block` is the mobile-first switch — full width up to `sm`, automatic above — used for the
 * one action a screen is really about, where a half-width button on a phone is a smaller
 * target for no reason at all.
 */
export function Button({
  variant,
  size,
  block,
  icon,
  pending,
  pendingLabel,
  disabled,
  type = 'button',
  onClick,
  title,
  className,
  children
}) {
  const kind =
    variant === 'primary'
      ? 'btn-primary'
      : variant === 'danger'
        ? 'btn-danger'
        : variant === 'ghost' || variant === 'quiet'
          ? 'btn-quiet'
          : 'btn-outline-secondary';

  return (
    <button
      type={type}
      className={cx(
        'btn',
        kind,
        size === 'sm' && 'btn-sm',
        block && 'w-100 w-sm-auto',
        'd-inline-flex align-items-center justify-content-center gap-1',
        className
      )}
      disabled={disabled || pending || false}
      onClick={onClick}
      title={title}
    >
      {pending ? (
        <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true" />
      ) : icon ? (
        <Icon name={icon} />
      ) : null}
      {pending && pendingLabel ? pendingLabel : children}
    </button>
  );
}

/* ==================================================================================
 * Form controls
 * ================================================================================== */

/**
 * A labelled control. The caller places it in a grid column, which is what keeps the
 * responsive decision in the view: `<div className="col-12 col-sm-6">` reads as "one per row
 * on a phone, two from small up" at the point where it matters.
 */
export function Field({ label, required, hint, error, className, children }) {
  return (
    <div className={className}>
      <label className="form-label">
        {label}
        {required ? <span className="text-danger"> *</span> : null}
      </label>
      {children}
      {error ? <div className="invalid-feedback d-block">{error}</div> : null}
      {hint && !error ? <div className="form-text">{hint}</div> : null}
    </div>
  );
}

export function Input({
  type = 'text',
  value,
  placeholder,
  disabled,
  inputMode,
  error,
  className,
  onInput,
  onKeyDown
}) {
  return (
    <input
      className={cx('form-control', error && 'is-invalid', className)}
      type={type}
      value={value ?? ''}
      placeholder={placeholder}
      disabled={disabled || false}
      autoComplete="off"
      spellCheck={false}
      inputMode={inputMode}
      onChange={(event) => onInput && onInput(event.target.value)}
      onKeyDown={onKeyDown}
    />
  );
}

/* ==================================================================================
 * Surfaces
 * ================================================================================== */

export function Card({ title, subtitle, actions, footer, tight, className, children }) {
  return (
    <section className={cx('card', className)}>
      {title || actions ? (
        <header className="card-header d-flex flex-wrap align-items-center gap-2">
          <h2 className="h6 mb-0 sis-pull">{title}</h2>
          {subtitle ? <span className="small text-body-tertiary">{subtitle}</span> : null}
          {actions ? <div className="d-flex flex-wrap gap-2">{actions}</div> : null}
        </header>
      ) : null}
      {tight ? children : <div className="card-body">{children}</div>}
      {footer ? (
        <footer className="card-footer d-flex flex-wrap align-items-center gap-2">
          {footer}
        </footer>
      ) : null}
    </section>
  );
}

/**
 * Tiles sit in a Bootstrap row and the caller states the wrap:
 * `row row-cols-2 row-cols-lg-4 g-3` is two across on a phone and four on a desktop. Two
 * rather than one on a phone because a count is short — a single column of four tiles is a
 * screenful of scrolling to read four numbers.
 */
export function Tile({ label, value, loading, note, to, linkText }) {
  const body = (
    <div className="card-body">
      <div className="sis-tile-label">{label}</div>
      <div className="sis-tile-value">
        {loading ? (
          <span className="sis-skel d-block" style={{ width: '3rem', height: '1.4rem' }} />
        ) : (
          countText(value)
        )}
      </div>
      {note ? <div className="small text-body-tertiary">{note}</div> : null}
      {to ? <div className="small mt-2">{linkText || 'Open'} →</div> : null}
    </div>
  );

  return (
    <div className="col">
      {to ? (
        <a className="card h-100 text-decoration-none text-body" href={Router.href(to)}>
          {body}
        </a>
      ) : (
        <div className="card h-100">{body}</div>
      )}
    </div>
  );
}

/* ==================================================================================
 * Badge, chip, alert
 * ================================================================================== */

/**
 * A tinted label: the status ink on its own wash, inside its own border.
 *
 * Built from Bootstrap's `*-subtle` trio rather than from `text-bg-*`, which is the pairing it
 * looks like it should use and cannot. `text-bg-warning` and `text-bg-info` hard-code
 * `color: #000` at build time, on the assumption that those backgrounds are a bright amber and
 * a bright cyan; ours are a dark brown and a dark blue, so a badge came out black on near-black.
 * The subtle trio reads `--bs-*-bg-subtle`, `--bs-*-text-emphasis` and `--bs-*-border-subtle`
 * at use, and theme.css already points all three at this design's status triples.
 */
export function Badge({ tone, className, children }) {
  const kind =
    tone === 'ok'
      ? 'text-success-emphasis bg-success-subtle border-success-subtle'
      : tone === 'warn'
        ? 'text-warning-emphasis bg-warning-subtle border-warning-subtle'
        : tone === 'bad'
          ? 'text-danger-emphasis bg-danger-subtle border-danger-subtle'
          : tone === 'info'
            ? 'text-info-emphasis bg-info-subtle border-info-subtle'
            : 'text-body-secondary bg-body-secondary border-secondary-subtle';
  return <span className={cx('badge border', kind, className)}>{children}</span>;
}

export function Chip({ active, count, onClick, children }) {
  return (
    <button type="button" className={cx('sis-chip', active && 'active')} onClick={onClick}>
      {children}
      {count === undefined ? null : <span className="sis-chip-count">{count}</span>}
    </button>
  );
}

export function Alert({ tone, title, className, children }) {
  const kind =
    tone === 'ok'
      ? 'alert-success'
      : tone === 'warn'
        ? 'alert-warning'
        : tone === 'bad'
          ? 'alert-danger'
          : 'alert-info';
  return (
    <div
      className={cx('alert', kind, 'd-flex gap-2 mb-0', className)}
      role={tone === 'bad' ? 'alert' : 'status'}
    >
      <Icon name={tone === 'ok' ? 'check' : 'alert'} size={18} />
      <div className="flex-grow-1" style={{ minWidth: 0 }}>
        {title ? <div className="fw-semibold">{title}</div> : null}
        {children}
      </div>
    </div>
  );
}

/* ==================================================================================
 * Empty and loading
 * ================================================================================== */

export function Empty({ icon, title, action, children }) {
  return (
    <div className="text-center text-body-secondary py-5 px-3">
      <div className="mb-2">
        <Icon name={icon || 'empty'} size={22} />
      </div>
      <p className="fw-semibold mb-1 text-body">{title}</p>
      {children ? (
        <p className="small mb-3 mx-auto" style={{ maxWidth: '46ch' }}>
          {children}
        </p>
      ) : null}
      {action || null}
    </div>
  );
}

/**
 * A skeleton rather than a spinner for anything table-shaped. A spinner tells a registrar to
 * wait; a skeleton tells them what is arriving, and the screen does not jump when it does,
 * because the placeholder already occupies the height the real rows will.
 */
export function Skeleton({ rows = 3 }) {
  const widths = ['75%', '55%', '40%', '80%', '60%'];
  return (
    <div className="vstack gap-2 p-3" aria-hidden="true">
      {Array.from({ length: rows }, (_, index) => (
        <div key={index} className="sis-skel" style={{ width: widths[index % widths.length] }} />
      ))}
    </div>
  );
}

/* ==================================================================================
 * Errors
 *
 * One renderer for every failure, keyed on the `kind` api.js assigns. The kinds are separated
 * by what the registrar has to DO, which is the only distinction a UI can act on, and the
 * advice below is the whole reason this exists rather than a bare `{error.message}`: "could
 * not reach the service" and "that preview expired" both read as failures, but only one of
 * them is fixed by trying again.
 * ================================================================================== */

const FRIENDLY_ERROR = {
  network: 'This information is temporarily unavailable. Please try again in a moment.',
  unauthorized: 'Your session needs to be refreshed before this information can be shown.',
  forbidden: 'This information is not available for your account.',
  too_large: 'This file is larger than the supported size. Try uploading it in smaller parts.',
  gone: 'This information is no longer available. Please refresh it and try again.',
  http: 'This information is not available yet. Please try again shortly.'
};

export function ErrorNote({ error, title, onRetry }) {
  if (!error) return null;
  const message = FRIENDLY_ERROR[error.kind] || 'This information is not available yet.';
  return (
    <div className="sis-error-note p-3" role="status">
      <Icon name="info" size={18} />
      <div className="flex-grow-1" style={{ minWidth: 0 }}>
        <div className="fw-semibold">{title || t('Information unavailable')}</div>
        <div className="small text-body-secondary mt-1">{t(message)}</div>
        {onRetry ? (
          <div className="mt-2">
            <Button size="sm" icon="refresh" onClick={onRetry}>
              {t('Try again')}
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* ==================================================================================
 * Table
 * ================================================================================== */

/**
 * `columns` is `[{key, header, cell, className, hide}]`.
 *
 * `hide` is the mobile-first lever and the most valuable thing in this component: a column
 * marked `hide: 'md'` is absent below Bootstrap's `md` and present above it. A class register
 * has six useful columns on a desktop and room for three on a phone, and *which* three is a
 * judgement each screen makes about its own data — so it lives in the column definition, next
 * to the data, rather than buried in a stylesheet.
 *
 * The table is wrapped in `.table-responsive`, so anything that still does not fit scrolls
 * inside its own box instead of widening the page and pushing the nav off screen.
 *
 * `onRowActivate` is the double-click gesture, and it is deliberately never the only way to
 * reach what it opens. A double-click does not exist on a phone — a double-tap is a zoom, or
 * nothing, depending on the browser — so a screen that wires this up also puts a real link in
 * a cell. This is the shortcut for someone with a mouse, not the route.
 */
export function Table({
  columns = [],
  rows = [],
  rowKey,
  rowTone,
  rowHref,
  rowLabel,
  loading,
  empty,
  onRowActivate,
  animate = true
}) {
  if (loading && !rows.length) return <Skeleton rows={6} />;
  if (!rows.length) return empty || <Empty title={t('Nothing to show')} />;

  return (
    <div className="table-responsive">
      <table className="table table-hover align-middle mb-0">
        <thead>
          <tr>
            {columns.map((column) => (
              <th
                key={column.key}
                scope="col"
                className={cx(column.className, hiddenBelow(column.hide))}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const tone = rowTone ? rowTone(row) : null;
            /* Capped at twelve steps: past about a quarter second of total stagger the effect
               stops reading as choreography and starts reading as a slow page. */
            const step = index < 12 ? index : 12;
              const href = rowHref ? rowHref(row, index) : null;
              const live = href || onRowActivate;
              return (
                <tr
                  key={rowKey ? rowKey(row, index) : index}
                  className={cx(
                    tone && `sis-row-${tone}`,
                    animate && 'sis-stagger',
                    live && 'sis-row-open'
                  )}
                  style={{ '--i': step }}
                  /* A single click anywhere that is not itself a control. `href` handles the
                     ordinary case through the stretched anchor below; this covers the rows whose
                     target is an action rather than an address. */
                  onClick={
                    onRowActivate && !href
                      ? (event) => {
                          if (event.target.closest('a,button,input,select,label')) return;
                          onRowActivate(row, index);
                        }
                      : undefined
                  }
                >
                  {columns.map((column, cell) => (
                    <td
                      key={column.key}
                      className={cx(column.className, hiddenBelow(column.hide))}
                    >
                      {/*
                        * The stretched anchor rides in the first cell, because a `<tr>` cannot
                        * contain an `<a>` — only cells can — and it is absolutely positioned
                        * against the row, so one anchor in one cell covers all of them.
                        */}
                      {href && cell === 0 ? (
                        <a
                          className="sis-row-target"
                          href={href}
                          aria-label={rowLabel ? rowLabel(row, index) : undefined}
                        />
                      ) : null}
                      {column.cell(row, index)}
                    </td>
                  ))}
                </tr>
              );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Pagination({ total = 0, limit = 50, offset = 0, onChange }) {
  if (total <= limit) return null;
  const from = offset + 1;
  const to = Math.min(offset + limit, total);

  return (
    <div className="d-flex flex-column flex-sm-row align-items-sm-center gap-2 p-3">
      <span className="small text-body-tertiary sis-num flex-sm-grow-1">
        {from}–{to} of {total}
      </span>
      <nav aria-label={t('Pages')}>
        <ul className="pagination pagination-sm mb-0">
          <li className={cx('page-item', offset <= 0 && 'disabled')}>
            <button className="page-link" onClick={() => onChange(Math.max(0, offset - limit))}>
              {t('Previous')}
            </button>
          </li>
          <li className={cx('page-item', to >= total && 'disabled')}>
            <button className="page-link" onClick={() => onChange(offset + limit)}>
              {t('Next')}
            </button>
          </li>
        </ul>
      </nav>
    </div>
  );
}

/* ==================================================================================
 * Drop zone
 * ================================================================================== */

/**
 * A file input that also accepts a drop. The `<input type=file>` stays in the DOM and is
 * merely hidden, so the keyboard path and the screen-reader label are the browser's own — and
 * on a phone, where dragging a file is not a gesture that exists, a tap opens the platform's
 * file picker exactly as it should.
 */
export function Dropzone({ file, label, hint, accept, onFile }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);

  function take(fileList) {
    if (fileList && fileList.length) onFile(fileList[0]);
  }

  return (
    <div
      className={cx('sis-dropzone', over && 'is-over', file && 'has-file')}
      onClick={() => input.current && input.current.click()}
      onDragOver={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        take(event.dataTransfer && event.dataTransfer.files);
      }}
    >
      <Icon name={file ? 'check' : 'upload'} size={22} />
      {file ? (
        <>
          <span className="fw-semibold">{file.name}</span>
          <span className="small">
            {Math.max(1, Math.round(file.size / 1024))} KB — tap to choose a different file
          </span>
        </>
      ) : (
        <>
          <span className="fw-semibold text-body">{label || 'Choose a spreadsheet'}</span>
          <span className="small">Tap to browse, or drop a file here. {hint || ''}</span>
        </>
      )}
      <input
        ref={input}
        type="file"
        className="visually-hidden"
        accept={accept || '.csv,.xlsx,.xls'}
        onChange={(event) => take(event.target.files)}
      />
    </div>
  );
}

/* ==================================================================================
 * Confirmation
 *
 * A Bootstrap modal rendered open rather than driven by Bootstrap's JavaScript: the console
 * has React for behaviour and never loads `bootstrap.bundle.js`, so the markup and classes
 * are Bootstrap's while the open state is ours. `modal-fullscreen-sm-down` is what makes it
 * usable on a phone — below `sm` it fills the screen instead of floating in the middle with
 * the page showing round the edges.
 * ================================================================================== */

/**
 * One side of one line of a diff, always drawable.
 *
 * React refuses to render a plain object as a child and throws, and a throw inside a modal
 * takes the whole console down — a blank page where a confirmation should be. A caller that
 * hands this an object has a bug worth fixing at the call site, but the failure it earns
 * should be an unreadable line in a dialog, not a registrar staring at nothing. `null` and
 * `''` are the ordinary case rather than the defensive one: they mean "this field was empty",
 * which is a fact the diff has to be able to state.
 */
function sideOfDiff(value) {
  if (value === '' || value == null) return DASH;
  if (typeof value === 'object') return JSON.stringify(value);
  return value;
}

export function Confirm({
  title,
  tone,
  confirmLabel,
  changes,
  pending,
  error,
  onCancel,
  onConfirm,
  children
}) {
  const host = useRef(null);

  useEffect(() => {
    function onKey(event) {
      if (event.key === 'Escape' && !pending) onCancel();
    }
    document.addEventListener('keydown', onKey);
    if (host.current) {
      const focusable = host.current.querySelector('button');
      if (focusable) focusable.focus();
    }
    return () => document.removeEventListener('keydown', onKey);
  }, [pending]);

  return (
    <>
      <div className="modal-backdrop show" />
      <div
        className="modal d-block"
        tabIndex="-1"
        role="dialog"
        aria-modal="true"
        onClick={(event) => {
          if (event.target === event.currentTarget && !pending) onCancel();
        }}
      >
        <div
          className="modal-dialog modal-dialog-centered modal-dialog-scrollable modal-fullscreen-sm-down"
          ref={host}
        >
          <div className="modal-content">
            <div className="modal-header">
              <h2 className="modal-title h6 d-flex align-items-center gap-2">
                <Icon name={tone === 'bad' ? 'alert' : 'check'} size={18} />
                {title}
              </h2>
            </div>
            <div className="modal-body vstack gap-3">
              {children}
              {changes && changes.length ? (
                <dl className="sis-diff">
                  {changes.map((change) => (
                    <Fragment key={change.label}>
                      <dt>{sideOfDiff(change.label)}</dt>
                      <dd>
                        <span className="sis-diff-was">{sideOfDiff(change.was)}</span>
                        <span className="text-body-tertiary" aria-hidden="true">
                          →
                        </span>
                        <span className="sis-diff-now">{sideOfDiff(change.now)}</span>
                      </dd>
                    </Fragment>
                  ))}
                </dl>
              ) : null}
              {error ? <ErrorNote error={error} /> : null}
            </div>
            <div className="modal-footer d-grid gap-2 d-sm-flex">
              <Button variant="quiet" disabled={pending} onClick={onCancel}>
                {t('Cancel')}
              </Button>
              <Button
                variant={tone === 'bad' ? 'danger' : 'primary'}
                pending={pending}
                pendingLabel={t('Saving…')}
                onClick={onConfirm}
              >
                {confirmLabel || 'Confirm'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/**
 * Ask, then act. Returns `[dialog, ask]`; render `dialog` anywhere in the screen and call
 * `ask({title, body, changes, tone, confirmLabel, run})` to raise it.
 *
 * The dialog stays open while `run` is in flight and closes on success, so a failure is read
 * in the dialog that caused it rather than in a toast that arrives after the screen has moved
 * on.
 */
export function useConfirm() {
  const [asked, setAsked] = useState(null);
  const action = useAction((run) => run());

  const dialog = asked ? (
    <Confirm
      title={asked.title}
      tone={asked.tone}
      confirmLabel={asked.confirmLabel}
      changes={asked.changes}
      pending={action.pending}
      error={action.error}
      onCancel={() => {
        action.reset();
        setAsked(null);
      }}
      onConfirm={() =>
        action
          .run(asked.run)
          .then(() => setAsked(null))
          .catch(() => {})
      }
    >
      {asked.body}
    </Confirm>
  ) : null;

  return [
    dialog,
    (request) => {
      action.reset();
      setAsked(request);
    }
  ];
}

/* ==================================================================================
 * Toasts
 * ================================================================================== */

/**
 * Bootstrap's toast in its own container. Full width on a phone (`start-0`) and pinned to the
 * corner from `sm` up, because a 24rem card in the corner of a 360px screen is a card with its
 * text wrapped to three words a line.
 */
export function Toasts() {
  const state = useStore();
  if (!state.toasts.length) return null;
  return (
    <div
      className="toast-container position-fixed bottom-0 end-0 start-0 start-sm-auto p-3"
      style={{ zIndex: 1090 }}
      aria-live="polite"
    >
      {state.toasts.map((item) => (
        <div key={item.id} className={cx('toast show', item.tone === 'bad' && 'sis-toast-muted')} role="status">
          <div className="toast-body d-flex gap-2 align-items-start">
            <Icon name={item.tone === 'bad' ? 'info' : 'check'} size={18} />
            <div className="flex-grow-1" style={{ minWidth: 0 }}>
              <div className="fw-semibold">
                {item.tone === 'bad' ? t('Information unavailable') : item.title}
              </div>
              {item.tone === 'bad' ? (
                <div className="small text-body-secondary">
                  {t('This information is temporarily unavailable. Please try again in a moment.')}
                </div>
              ) : item.detail ? (
                <div className="small text-body-secondary">{item.detail}</div>
              ) : null}
            </div>
            <button
              className="btn-close"
              aria-label={t('Dismiss')}
              onClick={() => Store.dismiss(item.id)}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

/* ==================================================================================
 * Page furniture
 * ================================================================================== */

export function Breadcrumbs({ trail = [] }) {
  const crumbs = trail.filter(Boolean);
  if (!crumbs.length) return null;
  return (
    <nav aria-label={t('Where you are')}>
      <ol className="breadcrumb sis-breadcrumb small mb-2">
        {crumbs.map((crumb, index) => {
          const last = index === crumbs.length - 1;
          return (
            <li
              key={crumb.label}
              className={cx('breadcrumb-item sis-breadcrumb-item', last && 'active')}
              aria-current={last ? 'page' : undefined}
            >
              {last || !crumb.to ? (
                crumb.label
              ) : (
                <a href={Router.href(crumb.to, crumb.params)}>{crumb.label}</a>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

/**
 * Stacks on a phone, side by side from `md` up, and the action group is a `d-grid` below `sm`
 * — so two or three buttons are full-width rows on a phone instead of a squeezed line, which
 * was the specific complaint that prompted this rewrite.
 */
export function PageHead({ title, lede, actions }) {
  return (
    <div className="d-flex flex-column flex-md-row align-items-md-end gap-3 mb-4">
      <div className="flex-md-grow-1" style={{ minWidth: 0 }}>
        <h1 className="h4 mb-1">{title}</h1>
        {lede ? (
          <p className="text-body-secondary mb-0" style={{ maxWidth: '62ch' }}>
            {lede}
          </p>
        ) : null}
      </div>
      {actions ? (
        <div className="d-grid gap-2 d-sm-flex flex-wrap">{actions}</div>
      ) : null}
    </div>
  );
}

/**
 * The banner every screen shows when no academic year exists.
 *
 * Extracted because it is the one piece of guidance the console must never get wrong: classes
 * are generated into a year, terms belong to one, and every upload names one — so a school
 * that has not created one can do nothing at all. Four screens showing four versions of this
 * sentence is how a registrar concludes the service is broken.
 */
export function NoYearNotice() {
  return (
    <Alert tone="warn" title={t('No academic year exists yet')}>
      {t('Nothing can be created until one does — classes, terms and every upload are recorded against a year.')} <a href={Router.href('school')}>{t('Create the academic year')}</a>, then come back here.
    </Alert>
  );
}
