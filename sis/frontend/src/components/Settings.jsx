/*
 * Settings: the three things about this console a person owns rather than a school does.
 *
 * The page colour, the appearance and which of a child's two names is shown first are all
 * preferences — they change nothing on the server, they are remembered per browser, and none
 * of them belongs on a screen about children. They used to be two icon buttons wedged into the
 * header beside the year picker, which is where controls go when nobody has decided where they
 * go: no room for a label, no room for a third one, and a theme toggle that cycled through
 * three states with no way to see which state you were in.
 *
 * The dialog exists to give them a room. What that buys, beyond space for labels:
 *
 *   Every option is visible at once. A cycling toggle asks the reader to remember what the
 *   sun icon meant last time; three labelled segments show which is on and what the others
 *   are, which is the whole reason segmented controls exist.
 *
 *   The page colour gets somewhere to be demonstrated. A colour is not a setting anyone can
 *   choose from a hex value, so the panel below carries a working miniature of the console —
 *   a header, a card, a hairline, a button, a field — drawn from the same tokens the real
 *   screens use. It is not a picture of the console; it is the console's own CSS at 1/4 size,
 *   so it cannot show something the app will not do.
 *
 * Changes apply immediately rather than on a Save. There is nothing to validate and nothing to
 * post, the page behind the dialog *is* the preview, and a Save button on a dialog whose whole
 * subject is "what does this look like" would be asking the reader to commit to an appearance
 * they have not seen. `Done` closes; `Reset` puts the stylesheet's own colour back.
 */
import { useEffect, useRef } from 'react';
import { Store } from '../store.js';
import { useStore } from '../hooks.js';
import { Button, Icon, cx } from './Ui.jsx';
import { t } from '../i18n.js';

/*
 * The offered page colours, per appearance.
 *
 * A list of presets *and* a free colour input, which is not redundancy. Almost nobody wants to
 * choose a hex value; what they want is "warmer than this" or "less bright", and five named
 * papers answer that in one click. The input is there because the moment a school has a colour
 * of its own, a menu of five is a menu that does not contain it — and since every surface in
 * the console is mixed from this one value, an arbitrary colour still comes out as a design
 * rather than as a stripe of somebody's brand laid over one.
 *
 * The light list is built around the default: an off-white with a little black in it, warm by
 * two points. Pure white is offered rather than assumed, which is the inverse of how this
 * started — a white page is a choice here, not the ground state.
 */
const PAPERS = {
  light: [
    { value: '#faf9f7', name: 'Paper', note: 'Warm off-white. The default.' },
    { value: '#ffffff', name: 'White', note: 'Pure white, no tint at all.' },
    { value: '#f7f8fa', name: 'Mist', note: 'Cool off-white.' },
    { value: '#fbf7ef', name: 'Cream', note: 'Warmer, closer to paper stock.' },
    { value: '#f5f1e8', name: 'Sand', note: 'Deeper cream, the lowest of these.' }
  ],
  dark: [
    { value: '#1d1d1f', name: 'Charcoal', note: 'Apple’s dark ground. The default.' },
    { value: '#000000', name: 'Black', note: 'True black, for OLED screens.' },
    { value: '#16181d', name: 'Ink', note: 'Cooler and a little deeper.' },
    { value: '#1c1a17', name: 'Umber', note: 'Warm, the dark twin of Cream.' },
    { value: '#22242a', name: 'Steel', note: 'Lifted, for a bright room.' }
  ]
};

const APPEARANCES = [
  { value: 'light', label: 'Light', icon: 'sun' },
  { value: 'dark', label: 'Dark', icon: 'moon' },
  { value: 'system', label: 'System', icon: 'contrast' }
];

const LANGUAGES = [
  { value: 'en', label: 'English', note: 'Latin names first' },
  { value: 'ar', label: 'العربية', note: 'Arabic names first, right to left' }
];

/* -- The miniature ---------------------------------------------------------------- */

/**
 * A working console at a quarter size: header, section, hairline, row, field, button.
 *
 * Every part of it takes its colour from the same custom properties the real screens do, so it
 * updates the instant `--tint` changes and it cannot drift from the app — there is no second
 * palette here to keep in step. What it shows is exactly the set of relationships the tint
 * governs and nothing else: the page against the panel, the panel against a field, a hairline
 * against both, and the one blue on top of all three.
 *
 * It is `aria-hidden` and carries a caption instead. To a screen reader a diagram of four
 * greys is four empty boxes; the sentence beside it is the information.
 */
function Preview() {
  return (
    <figure className="mb-0">
      <div className="sis-preview" aria-hidden="true">
        <div className="sis-preview-bar">
          <span className="sis-preview-mark" />
          <span className="sis-preview-line" style={{ width: '3.5rem' }} />
          <span className="sis-preview-line sis-push" style={{ width: '1.75rem' }} />
        </div>
        <div className="sis-preview-page">
          <div className="sis-preview-card">
            <div className="sis-preview-card-head">
              <span className="sis-preview-line sis-preview-strong" style={{ width: '4rem' }} />
            </div>
            <div className="sis-preview-row">
              <span className="sis-preview-line" style={{ width: '5rem' }} />
              <span className="sis-preview-line sis-preview-accent sis-push" style={{ width: '2rem' }} />
            </div>
            <div className="sis-preview-row">
              <span className="sis-preview-line" style={{ width: '3.5rem' }} />
              <span className="sis-preview-field sis-push" />
            </div>
            <div className="sis-preview-foot">
              <span className="sis-preview-btn" />
              <span className="sis-preview-btn sis-preview-btn-quiet" />
            </div>
          </div>
        </div>
      </div>
      <figcaption className="small text-body-tertiary mt-2">
        {t('The page, a section on it, a hairline, a field and the one blue — the same tokens every screen is drawn from, at a quarter size.')}
      </figcaption>
    </figure>
  );
}

/* -- Sections --------------------------------------------------------------------- */

/** A labelled block. A heading and a sentence saying what the choice below actually does. */
function Group({ title, hint, children }) {
  return (
    <section className="vstack gap-2">
      <div>
        <h3 className="h6 mb-0">{title}</h3>
        {hint ? <p className="small text-body-tertiary mb-0">{hint}</p> : null}
      </div>
      {children}
    </section>
  );
}

/**
 * The segmented control, as a radio group.
 *
 * `role="radiogroup"` and real `aria-checked` rather than a row of buttons: this is one choice
 * among several, and a screen reader announcing "button, Light" three times says nothing about
 * which one is on.
 */
function Segments({ label, value, options, onChange }) {
  return (
    <div className="nav nav-pills flex-nowrap" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value ? 'true' : 'false'}
          className={cx(
            'nav-link d-flex align-items-center gap-2 text-nowrap',
            value === option.value && 'active'
          )}
          onClick={() => onChange(option.value)}
        >
          {option.icon ? <Icon name={option.icon} /> : null}
          {option.label}
        </button>
      ))}
    </div>
  );
}

/* -- The dialog ------------------------------------------------------------------- */

/**
 * The settings dialog. Bootstrap's modal markup rendered open by React, like `Confirm` — the
 * console has React for behaviour and never loads Bootstrap's JavaScript.
 *
 * The backdrop blurs rather than dims (see `.modal-backdrop` in theme.css), and that is load
 * bearing here rather than decorative: this dialog's subject is what a colour does to the page
 * behind it, and a scrim heavy enough to hide that page would hide the only thing worth
 * looking at.
 */
export function Settings({ onClose }) {
  const state = useStore();
  const host = useRef(null);
  const where = Store.appearance();
  const papers = PAPERS[where];
  /* What the page is painted right now. The stored value is null while the stylesheet's own
     default is in force, and a colour input has to be given a concrete colour. */
  const tint = Store.currentTint() || papers[0].value;
  const custom = state.tint[where];

  useEffect(() => {
    function onKey(event) {
      if (event.key === 'Escape') onClose();
    }
    document.addEventListener('keydown', onKey);
    if (host.current) {
      const first = host.current.querySelector('button');
      if (first) first.focus();
    }
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <>
      <div className="modal-backdrop show" />
      <div
        className="modal d-block"
        tabIndex="-1"
        role="dialog"
        aria-modal="true"
        aria-label={t('Settings')}
        onClick={(event) => {
          if (event.target === event.currentTarget) onClose();
        }}
      >
        <div
          className="modal-dialog modal-dialog-centered modal-dialog-scrollable modal-fullscreen-sm-down"
          ref={host}
        >
          <div className="modal-content">
            <div className="modal-header">
              <h2 className="modal-title h6 d-flex align-items-center gap-2">
                <Icon name="settings" size={18} />
                {t('Settings')}
              </h2>
              <button
                type="button"
                className="btn btn-sm btn-quiet sis-push"
                onClick={onClose}
                aria-label={t('Close settings')}
              >
                <Icon name="close" />
              </button>
            </div>

            <div className="modal-body vstack gap-4">
              <Preview />

              <Group
                title={t('Appearance')}
                hint={t('System follows the machine, and changes with it while the tab is open.')}
              >
                <Segments
                  label={t('Appearance')}
                  value={state.theme}
                  options={APPEARANCES}
                  onChange={Store.setTheme}
                />
              </Group>

              <Group
                title={t('Page colour')}
                hint={
                  where === 'dark'
                    ? t('The colour of the dark page. The light one keeps its own, and switching appearance switches between them.')
                    : t('The colour of the page. Every section, hairline and field on it is mixed from this one value, so the whole console follows.')
                }
              >
                <div className="d-flex flex-wrap gap-2" role="radiogroup" aria-label={t('Page colour')}>
                  {papers.map((paper) => (
                    <button
                      key={paper.value}
                      type="button"
                      role="radio"
                      aria-checked={tint === paper.value ? 'true' : 'false'}
                      className={cx('sis-swatch', tint === paper.value && 'active')}
                      style={{ '--swatch': paper.value }}
                      title={`${paper.name} — ${paper.note}`}
                      onClick={() => Store.setTint(paper.value)}
                    >
                      <span className="sis-swatch-chip" />
                      <span className="sis-swatch-name">{paper.name}</span>
                    </button>
                  ))}
                </div>

                <div className="d-flex flex-wrap align-items-center gap-3">
                  <label className="d-flex align-items-center gap-2 mb-0">
                    <Icon name="droplet" />
                    <span className="small">{t('Any colour')}</span>
                    <input
                      type="color"
                      className="form-control form-control-color"
                      value={tint}
                      onInput={(event) => Store.setTint(event.target.value)}
                      aria-label={t('Pick any page colour')}
                    />
                  </label>
                  <Button
                    variant="quiet"
                    size="sm"
                    disabled={!custom}
                    onClick={() => Store.setTint(null)}
                  >
                    {t('Reset to default')}
                  </Button>
                </div>
              </Group>

              <Group
                title={t('Names')}
                hint={t('Which of a child’s two recorded names is shown first. The console’s own wording stays in English — a half-translated interface is worse than an untranslated one.')}
              >
                <Segments
                  label={t('Names')}
                  value={state.lang}
                  options={LANGUAGES}
                  onChange={Store.setLang}
                />
                <p className="small text-body-tertiary mb-0">
                  {LANGUAGES.find((item) => item.value === state.lang).note}.
                </p>
              </Group>
            </div>

            <div className="modal-footer d-grid d-sm-flex">
              <span className="small text-body-tertiary flex-sm-grow-1">
                {t('Remembered in this browser. Nothing here is sent to the service.')}
              </span>
              <Button variant="primary" onClick={onClose}>
                {t('Done')}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
