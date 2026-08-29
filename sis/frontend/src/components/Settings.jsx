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
  { value: 'en', label: 'English', note: 'Latin names first' },
  { value: 'ar', label: 'العربية', note: 'Arabic names first, right to left' }
];

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
          {t(option.label)}
        </button>
      ))}
    </div>
  );
}

export function Settings({ onClose }) {
  const state = useStore();
  const host = useRef(null);

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
              <Group
                title={t('Appearance')}
                hint={t('Choose a clear light or dark appearance. Your choice is remembered in this browser.')}
              >
                <Segments
                  label={t('Appearance')}
                  value={state.theme}
                  options={APPEARANCES}
                  onChange={Store.setTheme}
                />
              </Group>

              <Group
                title={t('Language')}
                hint={t('Choose the interface language and reading direction.')}
              >
                <Segments
                  label={t('Language')}
                  value={state.lang}
                  options={LANGUAGES}
                  onChange={Store.setLang}
                />
                <p className="small text-body-tertiary mb-0">
                  {t(LANGUAGES.find((item) => item.value === state.lang).note)}.
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
