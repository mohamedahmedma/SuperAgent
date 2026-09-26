/* Browser-owned preferences: appearance and interface language. */
import { useEffect, useRef } from 'react';
import { Store } from '../store.js';
import { useStore } from '../hooks.js';
import { Button, Icon, cx } from './Ui.jsx';
import { t } from '../i18n.js';

const APPEARANCES = [
  { value: 'light', label: 'Light', icon: 'sun' },
  { value: 'dark', label: 'Dark', icon: 'moon' }
];

const LANGUAGES = [
  { value: 'en', label: 'English', shortLabel: 'EN', note: 'Latin names first' },
  { value: 'ar', label: 'Arabic', shortLabel: 'عربي', note: 'Arabic names first, right to left' }
];

function Group({ title, hint, mobileHint, children }) {
  return (
    <section className="sis-settings-group">
      <div className="sis-settings-group-copy">
        <h3 className="h6 mb-0">{title}</h3>
        {hint ? <p className="sis-settings-desktop-hint small text-body-tertiary mb-0">{hint}</p> : null}
        {mobileHint ? <p className="sis-settings-mobile-hint mb-0">{mobileHint}</p> : null}
      </div>
      {children}
    </section>
  );
}

function Segments({ label, value, options, onChange, kind }) {
  return (
    <div className={cx('nav nav-pills flex-nowrap sis-settings-segments', kind && `is-${kind}`)} role="radiogroup" aria-label={label}>
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
          aria-label={t(option.label)}
          title={t(option.label)}
        >
          {option.icon ? <Icon name={option.icon} /> : null}
          <span className="sis-settings-segment-label">{t(option.label)}</span>
          {option.shortLabel ? <span className="sis-settings-segment-short" aria-hidden="true">{option.shortLabel}</span> : null}
        </button>
      ))}
    </div>
  );
}

export function Settings({ onClose }) {
  const state = useStore();
  const host = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    function onKey(event) {
      if (event.key === 'Escape') {
        closeRef.current();
        return;
      }
      if (event.key !== 'Tab' || !host.current) return;
      const focusable = [...host.current.querySelectorAll('button:not(:disabled)')];
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    document.addEventListener('keydown', onKey);
    if (host.current) {
      const first = host.current.querySelector('button');
      if (first) first.focus();
    }
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener('keydown', onKey);
    };
  }, []);

  return (
    <>
      <div className="modal-backdrop sis-settings-backdrop show" />
      <div
        className="modal sis-settings-modal d-block"
        tabIndex="-1"
        role="dialog"
        aria-modal="true"
        aria-labelledby="sis-settings-title"
        onClick={(event) => {
          if (event.target === event.currentTarget) onClose();
        }}
      >
        <div
          className="modal-dialog modal-dialog-centered modal-dialog-scrollable sis-settings-dialog"
          ref={host}
        >
          <div className="modal-content sis-settings-card">
            <div className="modal-header sis-settings-header">
              <div>
                <span className="sis-settings-eyebrow">{t('Preferences')}</span>
                <h2 id="sis-settings-title" className="modal-title h6 d-flex align-items-center gap-2">
                  <span className="sis-settings-title-icon"><Icon name="settings" size={18} /></span>
                  {t('Settings')}
                </h2>
              </div>
              <button
                type="button"
                className="btn btn-sm btn-quiet sis-push sis-settings-close"
                onClick={onClose}
                aria-label={t('Close settings')}
              >
                <Icon name="close" />
              </button>
            </div>

            <div className="modal-body sis-settings-body">
              <Group
                title={t('Appearance')}
                hint={t('Choose a clear light or dark appearance. Your choice is remembered in this browser.')}
                mobileHint={t(state.theme === 'dark' ? 'Dark mode' : 'Light mode')}
              >
                <Segments
                  label={t('Appearance')}
                  value={state.theme}
                  options={APPEARANCES}
                  onChange={Store.setTheme}
                  kind="appearance"
                />
              </Group>

              <Group
                title={t('Language')}
                hint={t('Choose the interface language and reading direction.')}
                mobileHint={t('Choose interface language')}
              >
                <Segments
                  label={t('Language')}
                  value={state.lang}
                  options={LANGUAGES}
                  onChange={Store.setLang}
                  kind="language"
                />
                <p className="sis-settings-language-note small text-body-tertiary mb-0">
                  {t(LANGUAGES.find((item) => item.value === state.lang).note)}.
                </p>
              </Group>
            </div>

            <div className="modal-footer sis-settings-footer d-grid d-sm-flex">
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
