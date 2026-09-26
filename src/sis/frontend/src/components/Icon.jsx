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
  /* AUREXIS_SIS_NAV_ICONS_V1 â€” distinct registrar navigation glyphs */
  school: 'M4 10.5 12 4l8 6.5V20H4zM8 20v-6h8v6',
  studentAdd: 'M15 19v-1a4 4 0 0 0-8 0v1M11 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM18 8v6m-3-3h6',
  roster: 'M6 4h12v16H6zM9 8h6M12 17v-6m0 0-2.5 2.5M12 11l2.5 2.5',
  guardian: 'M8 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM3 20v-2a5 5 0 0 1 8-4M17 9l4 2v3c0 3-2 5-4 6-2-1-4-3-4-6v-3z',
  roles: 'M4 5h16v14H4zM8 9h4M8 13h5M16 9h.01M16 13h.01',
  teacher: 'M4 5h16v10H9M14 9h3M7 21v-3m-3 3v-2a3 3 0 0 1 6 0v2M7 14a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5z',
  staff: 'M5 20v-2a4 4 0 0 1 4-4h2a4 4 0 0 1 4 4v2M10 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM17 7h4M19 5v4M17 13h4M19 11v4',
  classAssign: 'M4 5h7v6H4zM13 5h7v6h-7zM4 13h7v6H4zM15 16h5m-2.5-2.5V19',
  attendance: 'M5 5h14v15H5zM5 9h14M9 3v4m6-4v4M8 14l2 2 4-4',
  timetable: 'M4 5h16v15H4zM4 10h16M9 5v15m6-15v15M4 15h16',
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
  chat: 'M21 12a8 8 0 0 1-8 8H5l-3 2 1-5a8 8 0 1 1 18-5zM7.5 12h.01M12 12h.01M16.5 12h.01',
  bell: 'M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4',
  bellOff: 'M3 3l18 18M9.4 4.6A6 6 0 0 1 18 10c0 3.6.8 5.3 1.7 6.3M6.3 16.3C7.2 14.9 7 12.9 7 10c0-.6.1-1.2.2-1.7M4 17h13M10 21h4',
  send: 'M3 4l18 8-18 8 3-8-3-8zm3 8h15',
  paperclip: 'M21.4 11.6 12 21a6 6 0 0 1-8.5-8.5l9-9a4 4 0 0 1 5.7 5.7l-9 9a2 2 0 0 1-2.8-2.8l8.3-8.3',
  microphone: 'M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM5 11a7 7 0 0 0 14 0M12 18v3M9 21h6',
  pencil: 'M17 3a2.83 2.83 0 0 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z',
  trash: 'M4 7h16M9 7V4h6v3m3 0-1 14H7L6 7m4 4v6m4-6v6',
  pause: 'M9 5v14M15 5v14',
  play: 'm8 5 11 7-11 7z',
  file: 'M6 3h8l4 4v14H6zM14 3v5h5',
  eye: 'M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6zM12 9a3 3 0 1 1 0 6 3 3 0 0 1 0-6z',
  eyeOff: 'M3 3l18 18M10.6 6.1A10.8 10.8 0 0 1 12 6c6 0 9.5 6 9.5 6a15.2 15.2 0 0 1-2.1 2.8M6.2 6.2C3.8 7.8 2.5 12 2.5 12s3.5 6 9.5 6a9.8 9.8 0 0 0 3.1-.5M9.9 9.9A3 3 0 0 0 14.1 14',
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
  chevron: 'm6 9 6 6 6-6',
  /* A door with an arrow leaving it. The arrow points out of the frame rather than into it,
     which is the only thing separating "sign out" from "sign in" at 16px. */
  signout:'M9 21H5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h4M16 17l5-5-5-5M21 12H9',
  menu: 'M3 12h18M3 6h18M3 18h18',
  expand: 'M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7',
  minimize: 'M4 14h6v6M20 10h-6V4M14 10l7-7M10 14l-7 7',
  layers: 'M12 2 2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
  arrowLeft: 'M19 12H5M12 19l-7-7 7-7',
  arrowRight: 'M5 12h14M12 5l7 7-7 7',
  copy: 'M9 9h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V11a2 2 0 0 1 2-2z M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1',
  info: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z M12 16v-4 M12 8h.01'
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
