/*
 * The icon set and the class-name joiner.
 *
 * Their own module for one structural reason: `Field.jsx` needs both, and `Ui.jsx` needs
 * `Field.jsx` back, so leaving them where they were made the two files import each other.
 * ES modules survive that cycle as long as nothing is read at module scope — which is exactly
 * the kind of "works until somebody adds a constant" arrangement worth not having.
 *
 * `Ui.jsx` re-exports both, so every existing `import { Icon, cx } from './Ui.jsx'` keeps
 * working and no call site had to change.
 */

/** Join class names, dropping the falsy ones. */
export function cx(...parts) {
  return parts.filter(Boolean).join(' ');
}

const PATHS = {
  dashboard: 'M4 13h7V4H4v9zm0 7h7v-5H4v5zm9 0h7V11h-7v9zm0-16v5h7V4h-7z',
  structure: 'M4 6h16M4 12h16M4 18h10',
  upload: 'M12 16V4m0 0L8 8m4-4 4 4M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2',
  people: 'M16 19v-1a4 4 0 0 0-8 0v1M12 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM20 19v-1a3 3 0 0 0-2-2.8',
  marks: 'M5 4h11l3 3v13H5zM9 12h6M9 16h4M9 8h3',
  batches: 'M4 7h16M4 12h16M4 17h16M8 4v16',
  check: 'M20 6 9 17l-5-5',
  alert: 'M12 8v5m0 3h.01M10.3 4.3 2.6 18a1.5 1.5 0 0 0 1.3 2.2h16.2a1.5 1.5 0 0 0 1.3-2.2L13.7 4.3a1.5 1.5 0 0 0-2.6 0z',
  close: 'M18 6 6 18M6 6l12 12',
  refresh: 'M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6',
  sun: 'M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 2v2m0 16v2M4 12H2m20 0h-2M5.6 5.6 4.2 4.2m15.6 1.4 1.4-1.4M5.6 18.4l-1.4 1.4m15.6-1.4 1.4 1.4',
  moon: 'M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z',
  search: 'M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zm5.5-1.5L21 21',
  download: 'M12 4v12m0 0-4-4m4 4 4-4M4 20h16',
  calendar: 'M4 6h16v14H4zM4 10h16M8 3v4m8-4v4',
  empty: 'M4 7h16v13H4zM4 7l2-3h12l2 3M9 12h6',
  /* Sliders rather than a cog: at 16px a cog is a grey blob, and its teeth are the first
     thing to go. Three tracks with a handle on each survives the size. */
  settings: 'M4 21v-6M4 11V3M12 21v-9M12 8V3M20 21v-4M20 13V3M1 15h6M9 8h6M17 17h6',
  /* The half-filled circle that means "follow the machine": neither sun nor moon. */
  contrast: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 3v18a9 9 0 0 0 0-18z',
  droplet: 'M12 3.2 6.8 8.4a7.3 7.3 0 1 0 10.4 0L12 3.2z',
  /* Points down; the dropdown rotates it 180deg when the menu is open, so one glyph does both
     states and they cannot drift apart. */
  chevron: 'm6 9 6 6 6-6'
};

export function Icon({ name, size = 16, weight = 1.7 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={weight}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      style={{ flex: 'none' }}
    >
      <path d={PATHS[name] || ''} />
    </svg>
  );
}

