import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { api } from '../api.js';
import { useStore } from '../hooks.js';
import { Icon, useConfirm } from '../components/Ui.jsx';
import { t } from '../i18n.js';

const CONVERSATION_POLL_MS = 5000;
const MESSAGE_POLL_MS = 3000;
const PRESENCE_POLL_MS = 3000;

const ARABIC_TO_ENGLISH_WORDS = {
  'مدير المدرسة': 'School Manager',
  'مالك المدرسة': 'School Owner',
  'مشرف دور': 'Grade Supervisor',
  'مشرف صفوف': 'Grade Supervisor',
  'مشرف غياب': 'Attendance Supervisor',
  'مشرف الحضور والغياب': 'Attendance Supervisor',
  'مدرس': 'Teacher',
  'معلم': 'Teacher',
  'هيئة المدرسة': 'School Staff',
  'كل هيئة المدرسة': 'All School Staff',
  'فريق': 'Team',
  'فصل': 'Class',
  'بنين': 'Boys',
  'بنات': 'Girls',
  'الصف 1 الابتدائي': 'Primary 1',
  'الصف 2 الابتدائي': 'Primary 2',
  'الصف 3 الابتدائي': 'Primary 3',
  'الصف 4 الابتدائي': 'Primary 4',
  'الصف 5 الابتدائي': 'Primary 5',
  'الصف 6 الابتدائي': 'Primary 6',
  'الصف 1 الإعدادي': 'Preparatory 1',
  'الصف 2 الإعدادي': 'Preparatory 2',
  'الصف 3 الإعدادي': 'Preparatory 3',
  'الصف الأول الثانوي': 'Secondary 1',
  'الصف الثاني الثانوي': 'Secondary 2',
  'الصف الثالث الثانوي': 'Secondary 3',
  'علمي': 'Scientific',
  'أدبي': 'Literary',
  'علمي علوم': 'Science',
  'علمي رياضة': 'Math',
  'اللغة العربية': 'Arabic',
  'اللغة الإنجليزية': 'English',
  'الرياضيات': 'Mathematics',
  'العلوم': 'Science',
  'الدراسات الاجتماعية': 'Social Studies',
  'الكمبيوتر': 'Computer',
  'الرسم': 'Art',
  'الألعاب': 'Physical Education',
  'التربية الدينية': 'Religion',
  'الفيزياء': 'Physics',
  'الكيمياء': 'Chemistry',
  'الأحياء': 'Biology',
  'التاريخ': 'History',
  'الجغرافيا': 'Geography',
  'الفلسفة': 'Philosophy',
  'المنطق': 'Logic',
  'الجيولوجيا': 'Geology',
  'رياضة 1': 'Mathematics 1',
  'رياضة 2': 'Mathematics 2',
  'رياضة بحتة': 'Pure Mathematics',
  'رياضة تطبيقية': 'Applied Mathematics',
  'محمد': 'Mohamed', 'أحمد': 'Ahmed', 'عمر': 'Omar', 'يوسف': 'Youssef',
  'آدم': 'Adam', 'ياسين': 'Yassin', 'محمود': 'Mahmoud', 'مصطفى': 'Moustafa',
  'زياد': 'Ziad', 'حمزة': 'Hamza', 'علي': 'Ali', 'كريم': 'Kareem',
  'فاطمة': 'Fatma', 'ليلى': 'Layla', 'مريم': 'Mariam', 'ملك': 'Malak',
  'نور': 'Nour', 'سارة': 'Sarah', 'جنى': 'Jana', 'سلمى': 'Salma',
  'فرح': 'Farah', 'هدى': 'Hoda', 'يارا': 'Yara', 'منة': 'Menna',
  'خالد': 'Khaled', 'حسن': 'Hassan', 'إبراهيم': 'Ibrahim', 'طارق': 'Tarek',
  'وليد': 'Walid', 'أيمن': 'Ayman', 'شريف': 'Sherif', 'عمرو': 'Amr',
  'هشام': 'Hesham', 'عبد الرحمن': 'Abdelrahman', 'عبد الحميد': 'Abdel Hamid',
  'السيد': 'Elsayed', 'عثمان': 'Osman', 'منصور': 'Mansour', 'رشاد': 'Rashad',
  'فؤاد': 'Fouad', 'النجار': 'Elnaggar', 'الشاذلي': 'Elshazly', 'الرفاعي': 'Elrefaie',
  'أبو الحسن': 'Aboulhassan', 'سامح': 'Sameh', 'عبد العزيز': 'Abdelaziz',
  'نجوى': 'Nagwa', 'سراج الدين': 'Serageldin', 'بيتر': 'Peter', 'غطاس': 'Ghattas',
  'منى': 'Mona', 'كمال': 'Kamal', 'سليم': 'Selim', 'رندا': 'Randa',
  'وجدي': 'Wagdy', 'شيرين': 'Sherine', 'بشارة': 'Bishara', 'تامر': 'Tamer',
  'حليم': 'Halim', 'إيمان': 'Iman', 'عبد الهادي': 'Abdelhady', 'باسم': 'Bassem',
  'نشأت': 'Nashaat', 'غادة': 'Ghada', 'لطفي': 'Lotfy', 'نبيل': 'Nabil',
  'عامر': 'Amer', 'هالة': 'Hala', 'فاروق': 'Farouk', 'سيد': 'Sayed'
};

function transliterateToEn(text) {
  if (!text || !/[\u0600-\u06FF]/.test(text)) return text || '';
  let result = text;
  const sortedKeys = Object.keys(ARABIC_TO_ENGLISH_WORDS).sort((a, b) => b.length - a.length);
  for (const key of sortedKeys) {
    if (result.includes(key)) {
      result = result.split(key).join(ARABIC_TO_ENGLISH_WORDS[key]);
    }
  }
  return result.replace(/\s+/g, ' ').trim();
}

function localName(row, lang, prefix = 'title') {
  const primary = row && row[`${prefix}_${lang}`];
  const fallback = row && row[`${prefix}_${lang === 'ar' ? 'en' : 'ar'}`];
  let value = primary || fallback || '';
  if (lang === 'en' && /[\u0600-\u06FF]/.test(value)) {
    return transliterateToEn(value);
  }
  return value;
}

export function parseUtcDate(value) {
  if (!value) return new Date(NaN);
  if (value instanceof Date) return value;
  let str = String(value).trim();
  if (!str.endsWith('Z') && !/[+-]\d{2}(?::?\d{2})?$/.test(str)) {
    str = str.replace(' ', 'T') + 'Z';
  }
  return new Date(str);
}

function timeLabel(value, lang) {
  if (!value) return '';
  const date = parseUtcDate(value);
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat(lang === 'ar' ? 'ar-EG' : 'en', {
    hour: 'numeric', minute: '2-digit',
    month: date.toDateString() === new Date().toDateString() ? undefined : 'short',
    day: date.toDateString() === new Date().toDateString() ? undefined : 'numeric'
  }).format(date);
}

function categoryLabel(category) {
  return ({
    school: t('Whole school'),
    leadership: t('School leadership'),
    floor: t('Grade staff'),
    subject: t('Subject teachers'),
    class: t('Class teachers'),
    direct: t('Private chat')
  })[category] || t('Group');
}

function filterLabel(category) {
  return ({
    all: t('All'),
    school: t('School'),
    leadership: t('Supervisors'),
    floor: t('Grade groups'),
    subject: t('Subjects'),
    class: t('Class groups'),
    direct: t('Private')
  })[category] || categoryLabel(category);
}

function mergeMessages(previous, incoming) {
  const byId = new Map();
  previous.concat(incoming).forEach((message) => byId.set(message.id, message));
  return Array.from(byId.values()).sort((a, b) => a.id - b.id);
}

function ConversationButton({ conversation, active, lang, onSelect }) {
  const last = conversation.last_message;
  const lastPreview = last?.deleted
    ? t('Message deleted')
    : (last?.body || ({ image: t('Photo'), audio: t('Voice message'), file: t('Attachment') }[last?.attachment_kind] || ''));
  return (
    <button
      type="button"
      className={`sis-chat-conversation${active ? ' is-active' : ''}`}
      onClick={() => onSelect(conversation.id)}
      aria-current={active ? 'page' : undefined}
    >
      <span className={`sis-chat-avatar is-${conversation.category}`}>
        <Icon name={conversation.kind === 'direct' ? 'people' : 'chat'} size={18} />
        {conversation.kind === 'direct' && !conversation.observer_view ? <span className={`sis-chat-presence-dot${conversation.online ? ' is-online' : ''}`} aria-hidden="true" /> : null}
      </span>
      <span className="sis-chat-conversation-copy">
        <span className="sis-chat-conversation-topline">
          <strong>{localName(conversation, lang)}</strong>
          <time>{timeLabel(last?.created_at || conversation.updated_at, lang)}</time>
        </span>
        <span className="sis-chat-conversation-bottomline">
          <span>{last ? `${localName(last, lang, 'sender_name')}: ${lastPreview}` : (localName(conversation, lang, 'subtitle') || categoryLabel(conversation.category))}</span>
          {conversation.is_muted ? <span className="sis-chat-muted-mark" title={t('Notifications muted')} aria-label={t('Notifications muted')}><Icon name="bellOff" size={14} /></span> : null}
          {conversation.unread_count ? (
            <span className="sis-chat-unread" aria-label={t('{0} unread messages', [conversation.unread_count])}>
              {conversation.unread_count > 99 ? '99+' : conversation.unread_count}
            </span>
          ) : null}
        </span>
      </span>
    </button>
  );
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

const VOICE_WAVE = [10, 18, 13, 25, 32, 19, 28, 14, 22, 35, 20, 12, 29, 38, 24, 17, 31, 21, 13, 27, 34, 18, 25, 11, 30, 22, 16, 36, 26, 14, 23, 18];

function durationLabel(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

function VoiceMessage({ url, declaredDuration = 0 }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(declaredDuration);
  const [speed, setSpeed] = useState(1);
  const safeDuration = Math.max(duration || declaredDuration || 0, 0.1);
  const progress = Math.min(currentTime / safeDuration, 1);

  const toggle = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      try { await audio.play(); setPlaying(true); } catch { setPlaying(false); }
    } else {
      audio.pause();
      setPlaying(false);
    }
  };
  const seek = (event) => {
    const audio = audioRef.current;
    if (!audio) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
    audio.currentTime = ratio * safeDuration;
    setCurrentTime(audio.currentTime);
  };
  const cycleSpeed = () => {
    const speeds = [1, 1.5, 2];
    const next = speeds[(speeds.indexOf(speed) + 1) % speeds.length];
    setSpeed(next);
    if (audioRef.current) audioRef.current.playbackRate = next;
  };

  useEffect(() => {
    const audio = new Audio(url);
    const loaded = () => {
      const measured = audio.duration;
      setDuration(Number.isFinite(measured) ? measured : declaredDuration);
      audio.playbackRate = speed;
    };
    const update = () => setCurrentTime(audio.currentTime);
    const ended = () => { setPlaying(false); setCurrentTime(0); };

    audio.preload = 'metadata';
    audioRef.current = audio;
    audio.addEventListener('loadedmetadata', loaded);
    audio.addEventListener('timeupdate', update);
    audio.addEventListener('ended', ended);
    return () => {
      audio.pause();
      audio.removeEventListener('loadedmetadata', loaded);
      audio.removeEventListener('timeupdate', update);
      audio.removeEventListener('ended', ended);
      audio.removeAttribute('src');
      audioRef.current = null;
    };
  }, [url, declaredDuration]);

  return (
    <div className={`sis-voice-player${playing ? ' is-playing' : ''}`} dir="ltr">
      <button type="button" className="sis-voice-play" onClick={toggle} aria-label={playing ? t('Pause voice message') : t('Play voice message')}>
        <span aria-hidden="true" className="sis-voice-play-ring" />
        <Icon name={playing ? 'pause' : 'play'} size={17} />
      </button>
      <div className="sis-voice-main">
        <div className="sis-voice-heading">
          <span dir="auto"><Icon name="microphone" size={12} /> {t('Voice message')}</span>
          <button type="button" className="sis-voice-speed" onClick={cycleSpeed} aria-label={t('Playback speed {0}', [speed])} title={t('Playback speed {0}', [speed])}>{speed}×</button>
        </div>
        <button type="button" className="sis-voice-wave" onClick={seek} aria-label={t('Seek voice message')}>
          {VOICE_WAVE.map((height, index) => <span key={index} className={(index + 1) / VOICE_WAVE.length <= progress ? 'is-played' : ''} style={{ height: `${height}px` }} />)}
        </button>
        <div className="sis-voice-meta">
          <span>{durationLabel(currentTime)}</span>
          <span>{durationLabel(safeDuration)}</span>
        </div>
      </div>
    </div>
  );
}

function mayWriteChat(profile) {
  if (!profile) return false;
  const chatOverride = (profile.overrides || []).find((row) => row.permission === 'chat.write');
  if (chatOverride) return chatOverride.effect === 'allow';
  if ((profile.permissions || []).includes('chat.write')) return true;

  // Older sessions may predate the chat permissions in the profile payload. The API
  // still checks chat.write on every mutation, so this only keeps the correct control visible.
  const writerRoles = new Set([
    'admin', 'school_owner', 'school_manager', 'principal', 'floor_supervisor',
    'year_supervisor', 'attendance_supervisor', 'teacher'
  ]);
  return (profile.roles || []).some((role) => writerRoles.has(typeof role === 'string' ? role : role.role_code));
}

function AttachmentPreview({ attachment, url, loading, failed, onClose, onDownload }) {
  const downloadRef = useRef(null);
  const mimeType = attachment.mime_type || '';
  // Browsers without an inline PDF viewer (Android Chrome) turn an embedded PDF into a download.
  const canEmbed = mimeType.startsWith('text/') || (mimeType === 'application/pdf' && navigator.pdfViewerEnabled !== false);

  useEffect(() => {
    const previous = document.activeElement;
    const onKeyDown = (event) => { if (event.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKeyDown);
    window.requestAnimationFrame(() => downloadRef.current?.focus());
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [onClose]);

  // Overlays render on <body>: the route and each message run transform animations, which
  // make them the containing block for position: fixed and trap the overlay inside a bubble.
  return createPortal(
    <div className="sis-attachment-preview-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="sis-attachment-preview" role="dialog" aria-modal="true" aria-labelledby={`attachment-preview-${attachment.id}`}>
        <header>
          <button type="button" className="sis-preview-close" onClick={onClose} aria-label={t('Close')}><Icon name="close" size={19} /></button>
          <strong id={`attachment-preview-${attachment.id}`}>{attachment.original_filename}</strong>
          <button ref={downloadRef} type="button" className="sis-preview-download" onClick={onDownload} aria-label={t('Download {0}', [attachment.original_filename]).join('')} title={t('Download {0}', [attachment.original_filename]).join('')}><Icon name="download" size={19} /></button>
        </header>
        <div className="sis-attachment-preview-body">
          {loading ? <span className="spinner-border" role="status"><span className="visually-hidden">{t('Loading…')}</span></span> : null}
          {!loading && failed ? <div className="sis-preview-unavailable"><Icon name="file" size={34} /><strong>{t('Preview unavailable')}</strong><span>{t('The file could not be opened. You can still download it.')}</span></div> : null}
          {!loading && !failed && url && attachment.kind === 'image' ? <img src={url} alt={attachment.original_filename} /> : null}
          {!loading && !failed && url && attachment.kind === 'file' && canEmbed ? <iframe src={url} title={t('Preview')} /> : null}
          {!loading && !failed && url && attachment.kind === 'file' && !canEmbed ? <div className="sis-preview-unavailable"><Icon name="file" size={42} /><strong>{attachment.original_filename}</strong><span>{t('This file type cannot be previewed in the browser. You can download it to open it.')}</span><small>{formatBytes(attachment.size_bytes)}</small></div> : null}
        </div>
      </section>
    </div>,
    document.body
  );
}

function Attachment({ attachment, school }) {
  const [url, setUrl] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (attachment.kind === 'file') return undefined;
    let alive = true;
    let objectUrl = '';
    api.chatAttachmentBlob(school, attachment.id).then((blob) => {
      if (!alive) return;
      objectUrl = URL.createObjectURL(blob);
      setUrl(objectUrl);
    }).catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [attachment.id, attachment.kind, school]);

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const closePreview = useCallback(() => setPreviewOpen(false), []);

  const openPreview = async () => {
    setPreviewOpen(true);
    setFailed(false);
    if (attachment.kind === 'image' && url) return;
    if (previewUrl) return;
    setPreviewLoading(true);
    try {
      const blob = await api.chatAttachmentBlob(school, attachment.id);
      setPreviewUrl(URL.createObjectURL(blob));
    } catch {
      setFailed(true);
    } finally {
      setPreviewLoading(false);
    }
  };

  const download = async () => {
    try {
      const blob = await api.chatAttachmentBlob(school, attachment.id);
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = objectUrl;
      anchor.download = attachment.original_filename;
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch { setFailed(true); }
  };

  if (attachment.kind === 'image' && url) {
    return <><button type="button" className="sis-chat-image" onClick={openPreview} aria-label={t('Open {0}', [attachment.original_filename]).join('')}><img src={url} alt={attachment.original_filename} /></button>{previewOpen ? <AttachmentPreview attachment={attachment} url={url} loading={false} failed={failed} onClose={closePreview} onDownload={download} /> : null}</>;
  }
  if (attachment.kind === 'audio' && url) {
    return <VoiceMessage url={url} declaredDuration={attachment.duration_seconds || 0} />;
  }
  return (<>
    <button type="button" className="sis-chat-file" onClick={attachment.kind === 'audio' ? download : openPreview}>
      <span className="sis-chat-file-icon" aria-hidden="true"><Icon name={attachment.kind === 'audio' ? 'microphone' : 'file'} size={19} /></span>
      <span className="sis-chat-file-copy"><strong>{attachment.original_filename}</strong><small>{failed ? t('Download failed') : formatBytes(attachment.size_bytes)}</small></span>
      <Icon name={attachment.kind === 'audio' ? 'download' : 'eye'} size={17} />
    </button>
    {previewOpen && attachment.kind === 'file' ? <AttachmentPreview attachment={attachment} url={previewUrl} loading={previewLoading} failed={failed} onClose={closePreview} onDownload={download} /> : null}
  </>);
}

function ReceiptTicks({ summary, onClick, showCount = false }) {
  if (!summary) return null;
  const doubled = summary.total > 0 && summary.delivered === summary.total;
  const seen = summary.total > 0 && summary.read === summary.total;
  const label = summary.total
    ? t('{0} read, {1} delivered, {2} total', [summary.read, summary.delivered, summary.total])
    : t('Sent');
  return (
    <button type="button" className={`sis-chat-receipts${seen ? ' is-seen' : ''}${showCount ? ' has-count' : ''}`} onClick={onClick} title={label} aria-label={label}>
      <span className="sis-chat-receipt-ticks" aria-hidden="true"><span>✓</span>{doubled ? <span>✓</span> : null}</span>
      {showCount && summary.total > 0 ? <span className="sis-chat-receipt-count" aria-hidden="true"><Icon name="eye" size={11} />{summary.read}/{summary.total}</span> : null}
    </button>
  );
}

function ReceiptDetails({ rows, lang, loading, onClose }) {
  const groups = [
    ['read', t('Read by'), (row) => row.read_at],
    ['delivered', t('Delivered to'), (row) => !row.read_at && row.delivered_at],
    ['sent', t('Not delivered yet'), (row) => !row.delivered_at]
  ];
  return createPortal(
    <div className="sis-chat-receipt-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="sis-chat-receipt-panel" role="dialog" aria-modal="true" aria-labelledby="receipt-title">
        <header><h2 id="receipt-title">{t('Message info')}</h2><button type="button" onClick={onClose} aria-label={t('Close')}><Icon name="close" /></button></header>
        {loading ? <p className="sis-chat-muted">{t('Loading…')}</p> : groups.map(([key, label, predicate]) => {
          const matches = rows.filter(predicate);
          return <div className="sis-chat-receipt-group" key={key}><h3>{label} <span>{matches.length}</span></h3>{matches.map((row) => <div key={row.user_id}><span className="sis-chat-avatar is-direct"><Icon name="people" size={14} /></span><strong>{localName(row, lang, 'full_name')}</strong><time>{timeLabel(row.read_at || row.delivered_at, lang)}</time></div>)}</div>;
        })}
      </section>
    </div>,
    document.body
  );
}

function TypingIndicator({ names = [], compact = false }) {
  if (!names.length) return null;
  const label = names.length === 1
    ? t('{0} is typing…', [names[0]])
    : t('{0} people are typing…', [names.length]);
  return (
    <span className={`sis-chat-typing${compact ? ' is-compact' : ''}`} role="status" aria-label={label}>
      <span className="sis-chat-typing-dots" aria-hidden="true"><i /><i /><i /></span>
      <span>{label}</span>
    </span>
  );
}

function roleLabel(code) {
  return ({
    admin: t('Admin'),
    school_owner: t('School Owner'),
    school_manager: t('School Manager'),
    floor_supervisor: t('Floor Supervisor'),
    attendance_supervisor: t('Attendance Supervisor'),
    teacher: t('Teacher')
  })[code] || code.replaceAll('_', ' ');
}

function ProfileAssignments({ title, rows, lang }) {
  if (!rows?.length) return null;
  return (
    <div className="sis-chat-profile-section">
      <h3>{title}</h3>
      <div>{rows.map((row) => <span key={row.code}>{localName(row, lang, 'name') || row.code}</span>)}</div>
    </div>
  );
}

function StaffProfile({ person, lang, onClose }) {
  const closeRef = useRef(null);
  useEffect(() => {
    const previous = document.activeElement;
    const handleKey = (event) => { if (event.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKey);
    window.requestAnimationFrame(() => closeRef.current?.focus());
    return () => {
      document.removeEventListener('keydown', handleKey);
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [onClose]);

  return createPortal(
    <div className="sis-chat-profile-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <section className="sis-chat-profile" role="dialog" aria-modal="true" aria-labelledby="staff-profile-title">
        <button ref={closeRef} type="button" className="sis-chat-profile-close" onClick={onClose} aria-label={t('Close')}><Icon name="close" size={18} /></button>
        <div className="sis-chat-profile-identity">
          <span className="sis-chat-profile-avatar"><Icon name="people" size={24} /><span className={`sis-chat-presence-dot${person.online ? ' is-online' : ''}`} aria-hidden="true" /></span>
          <div>
            <h2 id="staff-profile-title">{localName(person, lang, 'full_name')}</h2>
            <p>{localName(person, lang, 'role_caption')}</p>
            <span className={`sis-chat-profile-status${person.online ? ' is-online' : ''}`}><i aria-hidden="true" />{person.online ? t('Online') : t('Offline')}</span>
          </div>
        </div>
        {person.roles?.length ? <div className="sis-chat-profile-section"><h3>{t('Roles')}</h3><div>{person.roles.map((role) => <span key={role}>{roleLabel(role)}</span>)}</div></div> : null}
        <ProfileAssignments title={t('Subjects')} rows={person.subjects} lang={lang} />
        <ProfileAssignments title={t('Grades')} rows={person.grades} lang={lang} />
        <ProfileAssignments title={t('Classes')} rows={person.classes} lang={lang} />
      </section>
    </div>,
    document.body
  );
}

function MessageActionSheet({
  message,
  mine,
  canEdit,
  canDelete,
  canShowReceipts,
  hasText,
  isAdmin,
  lang,
  onClose,
  onEdit,
  onDelete,
  onInfo,
  onCopy,
}) {
  useEffect(() => {
    const handleKey = (event) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  if (!message) return null;

  const snippet = message.body
    ? (message.body.length > 70 ? message.body.slice(0, 70) + '…' : message.body)
    : (message.attachments?.length ? t('Attach files') : '');

  return createPortal(
    <div
      className="sis-chat-action-backdrop"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      onTouchStart={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="sis-chat-action-sheet"
        role="dialog"
        aria-modal="true"
        aria-label={t('Message actions')}
      >
        <div className="sis-chat-action-header">
          <span className="sis-chat-action-handle" aria-hidden="true" />
          {snippet ? <p className="sis-chat-action-snippet">{snippet}</p> : null}
        </div>
        <div className="sis-chat-action-list">
          {hasText ? (
            <button
              type="button"
              className="sis-chat-action-item"
              onClick={() => {
                onCopy();
                onClose();
              }}
            >
              <span className="sis-chat-action-icon"><Icon name="copy" size={16} /></span>
              <span>{t('Copy text')}</span>
            </button>
          ) : null}
          {canEdit ? (
            <button
              type="button"
              className="sis-chat-action-item"
              onClick={() => {
                onEdit();
                onClose();
              }}
            >
              <span className="sis-chat-action-icon"><Icon name="pencil" size={16} /></span>
              <span>{t('Edit message')}</span>
            </button>
          ) : null}
          {canShowReceipts ? (
            <button
              type="button"
              className="sis-chat-action-item"
              onClick={() => {
                onInfo();
                onClose();
              }}
            >
              <span className="sis-chat-action-icon"><Icon name="info" size={16} /></span>
              <span>{t('Message info')}</span>
            </button>
          ) : null}
          {canDelete ? (
            <button
              type="button"
              className="sis-chat-action-item is-danger"
              onClick={() => {
                onDelete();
                onClose();
              }}
            >
              <span className="sis-chat-action-icon"><Icon name="trash" size={16} /></span>
              <span>{t('Delete message')}</span>
            </button>
          ) : null}
        </div>
        <button
          type="button"
          className="sis-chat-action-cancel"
          onClick={onClose}
        >
          {t('Cancel')}
        </button>
      </div>
    </div>,
    document.body
  );
}

export function Chat() {
  const state = useStore();
  const school = state.school;
  const currentUserId = state.profile?.user_id;
  const mayWrite = mayWriteChat(state.profile);
  const isAdmin = Boolean(state.profile?.is_system_admin);
  const [deleteDialog, askDelete] = useConfirm();
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [receiptRows, setReceiptRows] = useState(null);
  const [receiptLoading, setReceiptLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [people, setPeople] = useState([]);
  const [members, setMembers] = useState([]);
  const [profilePerson, setProfilePerson] = useState(null);
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [scopeFilter, setScopeFilter] = useState('all');
  const [loadingConversations, setLoadingConversations] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [loadingEarlier, setLoadingEarlier] = useState(false);
  const [sending, setSending] = useState(false);
  const [openingPerson, setOpeningPerson] = useState(null);
  const [error, setError] = useState(null);
  const [searchError, setSearchError] = useState(null);
  const [editingMessage, setEditingMessage] = useState(null);
  const [savedDraft, setSavedDraft] = useState('');
  const [selectedActionMessage, setSelectedActionMessage] = useState(null);
  const [muteMenuOpen, setMuteMenuOpen] = useState(false);
  const [savingMute, setSavingMute] = useState(false);
  const touchTimerRef = useRef(null);
  const touchStartPosRef = useRef({ x: 0, y: 0 });
  const endRef = useRef(null);
  const messagesRef = useRef(null);
  const composerRef = useRef(null);
  const fileInputRef = useRef(null);
  const recorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);
  const markedThrough = useRef(new Map());
  const stickToBottom = useRef(true);
  const active = conversations.find((item) => item.id === activeId) || null;
  const memberById = new Map(members.map((member) => [member.user_id, member]));
  const activePeer = active?.kind === 'direct' && !active.observer_view
    ? (memberById.get(active.peer_user_id) || members[0] || null)
    : null;
  const headerPerson = active?.kind === 'direct' && !active.observer_view ? (activePeer || {
    user_id: active.peer_user_id,
    full_name_en: active.title_en,
    full_name_ar: active.title_ar,
    role_caption_en: active.subtitle_en,
    role_caption_ar: active.subtitle_ar,
    online: active.online,
    roles: [], subjects: [], grades: [], classes: []
  }) : null;
  const typingMembers = members.filter((member) => member.typing);
  const categories = ['all', 'school', 'leadership', 'floor', 'subject', 'class', 'direct']
    .filter((category) => category === 'all' || conversations.some((item) => item.category === category));
  const scopes = Array.from(new Map(
    conversations
      .filter((item) => item.scope_code)
      .map((item) => [item.scope_code, {
        code: item.scope_code,
        name_en: item.scope_name_en,
        name_ar: item.scope_name_ar
      }])
  ).values());
  const visibleConversations = conversations.filter((conversation) => (
    (categoryFilter === 'all' || conversation.category === categoryFilter) &&
    (scopeFilter === 'all' || conversation.scope_code === scopeFilter)
  ));

  useLayoutEffect(() => {
    const textarea = composerRef.current;
    if (!textarea) return;
    textarea.style.height = 'auto';
    const height = Math.max(44, Math.min(textarea.scrollHeight, 128));
    textarea.style.height = `${height}px`;
    textarea.style.overflowY = textarea.scrollHeight > 128 ? 'auto' : 'hidden';
  }, [draft]);

  const refreshConversations = useCallback(async (quiet = false) => {
    if (!school) return;
    if (!quiet) setLoadingConversations(true);
    try {
      const rows = await api.chatConversations(school);
      setConversations(rows || []);
      setActiveId((selected) => {
        if (selected && rows.some((item) => item.id === selected)) return selected;
        return null;
      });
      setError(null);
    } catch (failure) {
      if (!quiet) setError(failure);
    } finally {
      if (!quiet) setLoadingConversations(false);
    }
  }, [school]);

  // Conversation selection belongs to the user. Polling may briefly return a partial list,
  // so it must never replace the open conversation with the first row automatically.
  useEffect(() => {
    setActiveId(null);
  }, [school]);

  useEffect(() => {
    let alive = true;
    refreshConversations();
    const timer = window.setInterval(() => {
      if (alive) refreshConversations(true);
    }, CONVERSATION_POLL_MS);
    return () => { alive = false; window.clearInterval(timer); };
  }, [refreshConversations]);

  useEffect(() => {
    setMessages([]);
    setDraft('');
    setEditingMessage(null);
    setSavedDraft('');
    setMuteMenuOpen(false);
    if (!school || !activeId) return undefined;
    stickToBottom.current = true;
    let alive = true;
    let first = true;
    const refresh = async () => {
      try {
        const rows = await api.chatMessages(school, activeId);
        if (!alive) return;
        setMessages((existing) => first ? (rows || []) : mergeMessages(existing, rows || []));
        first = false;
        setLoadingMessages(false);
        setError(null);
        const latestId = rows && rows.length ? rows[rows.length - 1].id : null;
        if (latestId && markedThrough.current.get(activeId) !== latestId) {
          await api.markChatRead(school, activeId);
          markedThrough.current.set(activeId, latestId);
          if (!alive) return;
          setConversations((items) => items.map((item) => (
            item.id === activeId ? { ...item, unread_count: 0 } : item
          )));
        }
      } catch (failure) {
        if (alive) { setLoadingMessages(false); setError(failure); }
      }
    };
    setLoadingMessages(true);
    refresh();
    const timer = window.setInterval(refresh, MESSAGE_POLL_MS);
    return () => { alive = false; window.clearInterval(timer); };
  }, [school, activeId]);

  useEffect(() => {
    if (!stickToBottom.current || !messagesRef.current) return;
    const messageList = messagesRef.current;
    messageList.scrollTop = messageList.scrollHeight;
  }, [activeId, messages.length]);

  useEffect(() => {
    setMembers([]);
    if (!school || !activeId) return undefined;
    let alive = true;
    const refreshMembers = async () => {
      try {
        const rows = await api.chatMembers(school, activeId);
        if (alive) {
          setMembers(rows || []);
          setProfilePerson((shown) => shown ? ((rows || []).find((member) => member.user_id === shown.user_id) || shown) : null);
        }
      } catch (failure) {
        if (alive) setError(failure);
      }
    };
    refreshMembers();
    const timer = window.setInterval(refreshMembers, PRESENCE_POLL_MS);
    return () => { alive = false; window.clearInterval(timer); };
  }, [school, activeId]);

  const activelyTyping = Boolean(draft.trim());
  useEffect(() => {
    if (!school || !activeId || !mayWrite) return undefined;
    const publish = (typing) => api.chatPresence(school, { conversation_id: activeId, typing }).catch(() => {});
    if (!activelyTyping) {
      publish(false);
      return undefined;
    }
    publish(true);
    const timer = window.setInterval(() => publish(true), PRESENCE_POLL_MS);
    return () => {
      window.clearInterval(timer);
      publish(false);
    };
  }, [school, activeId, mayWrite, activelyTyping]);

  useEffect(() => {
    const needle = search.trim();
    setSearchError(null);
    if (!school || needle.length < 2) {
      setPeople([]);
      return undefined;
    }
    let alive = true;
    const timer = window.setTimeout(async () => {
      try {
        const rows = await api.chatPeople(school, needle);
        if (alive) setPeople(rows || []);
      } catch (failure) {
        if (alive) setSearchError(failure);
      }
    }, 250);
    return () => { alive = false; window.clearTimeout(timer); };
  }, [school, search]);

  const openPerson = async (person) => {
    setOpeningPerson(person.user_id);
    setSearchError(null);
    try {
      const conversation = await api.openDirectChat(school, person.user_id);
      setConversations((items) => {
        const rest = items.filter((item) => item.id !== conversation.id);
        return [conversation, ...rest];
      });
      setActiveId(conversation.id);
      setSearch('');
      setPeople([]);
    } catch (failure) {
      setSearchError(failure);
    } finally {
      setOpeningPerson(null);
    }
  };

  const changeMute = async (duration) => {
    if (!activeId || savingMute) return;
    setSavingMute(true);
    setError(null);
    try {
      const preference = await api.setChatMute(school, activeId, duration);
      setConversations((items) => items.map((item) => item.id === activeId
        ? { ...item, ...preference }
        : item));
      setMuteMenuOpen(false);
    } catch (failure) {
      setError(failure);
    } finally {
      setSavingMute(false);
    }
  };

  const send = async (filesOverride = null, durationOverride = null) => {
    const body = draft.trim();
    const files = filesOverride || selectedFiles;
    if ((!body && !files.length) || !activeId || sending) return;
    setSending(true);
    setError(null);
    try {
      const message = files.length
        ? await api.sendChatAttachments(school, activeId, files, body, durationOverride)
        : await api.sendChatMessage(school, activeId, body);
      stickToBottom.current = true;
      setMessages((items) => mergeMessages(items, [message]));
      setDraft('');
      setSelectedFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
      await refreshConversations(true);
    } catch (failure) {
      setError(failure);
    } finally {
      setSending(false);
    }
  };

  const requestMessageDelete = (message) => {
    askDelete({
      title: t('Delete this message?'),
      tone: 'bad',
      confirmLabel: t('Delete message'),
      body: <p>{t('Everyone else will see that a message was deleted. Its original content is retained for administrative review.')}</p>,
      run: async () => {
        const deleted = await api.deleteChatMessage(school, activeId, message.id);
        setMessages((items) => items.map((item) => item.id === deleted.id ? deleted : item));
        await refreshConversations(true);
      }
    });
  };

  const handleCopyMessage = (msg) => {
    if (!msg?.body) return;
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(msg.body).catch(() => {});
    } else {
      try {
        const textarea = document.createElement('textarea');
        textarea.value = msg.body;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      } catch {}
    }
  };

  const handleMessageTouchStart = (msg, event) => {
    if (touchTimerRef.current) window.clearTimeout(touchTimerRef.current);
    if (event.target.closest('.sis-voice-wave, .sis-voice-play, .sis-voice-speed')) return;
    const touch = event.touches?.[0];
    if (touch) {
      touchStartPosRef.current = { x: touch.clientX, y: touch.clientY };
    }
    touchTimerRef.current = window.setTimeout(() => {
      touchTimerRef.current = null;
      if (navigator.vibrate) {
        try { navigator.vibrate(40); } catch {}
      }
      setSelectedActionMessage(msg);
    }, 450);
  };

  const handleMessageTouchMove = (event) => {
    if (!touchTimerRef.current) return;
    const touch = event.touches?.[0];
    if (touch) {
      const dx = Math.abs(touch.clientX - touchStartPosRef.current.x);
      const dy = Math.abs(touch.clientY - touchStartPosRef.current.y);
      if (dx > 10 || dy > 10) {
        window.clearTimeout(touchTimerRef.current);
        touchTimerRef.current = null;
      }
    }
  };

  const handleMessageTouchEnd = () => {
    if (touchTimerRef.current) {
      window.clearTimeout(touchTimerRef.current);
      touchTimerRef.current = null;
    }
  };

  useEffect(() => {
    setSelectedActionMessage(null);
  }, [activeId]);

  useEffect(() => () => {
    if (touchTimerRef.current) window.clearTimeout(touchTimerRef.current);
  }, []);

  const handleComposerEnter = (event) => {
    if (event.key !== 'Enter' || event.shiftKey || event.ctrlKey || event.altKey || event.metaKey || event.nativeEvent?.isComposing) return;
    event.preventDefault();
    if (isRecording) {
      if (recordingSeconds >= 1) stopRecording(false, true);
      return;
    }
    if (editingMessage) { submitEdit(); } else { send(); }
    window.requestAnimationFrame(() => composerRef.current?.focus());
  };

  const stopRecording = (discard = false, sendImmediately = false) => {
    window.clearInterval(recordingTimerRef.current);
    const recorder = recorderRef.current;
    const duration = recordingSeconds;
    setIsRecording(false);
    setIsPaused(false);
    if (!recorder || recorder.state === 'inactive') return;
    recorder.ondataavailable = (event) => {
      if (!discard && event.data.size) audioChunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      recorder.stream.getTracks().forEach((track) => track.stop());
      if (!discard && audioChunksRef.current.length) {
        const mimeType = recorder.mimeType || 'audio/webm';
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        const extension = mimeType.includes('ogg') ? 'ogg' : 'webm';
        const file = new File([blob], `voice-${Date.now()}.${extension}`, { type: mimeType });
        if (sendImmediately) send([file], Math.max(1, duration));
        else setSelectedFiles([file]);
      }
      audioChunksRef.current = [];
      recorderRef.current = null;
    };
    recorder.stop();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const startRecordingClock = () => {
        window.clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = window.setInterval(() => {
          if (recorder.state !== 'recording') return;
          setRecordingSeconds((seconds) => {
            if (seconds >= 59) {
              window.clearInterval(recordingTimerRef.current);
              if (recorder.state === 'recording') recorder.pause();
              return 60;
            }
            return seconds + 1;
          });
        }, 1000);
      };
      recorderRef.current = recorder;
      audioChunksRef.current = [];
      setRecordingSeconds(0);
      setIsPaused(false);
      setIsRecording(true);
      recorder.addEventListener('pause', () => {
        window.clearInterval(recordingTimerRef.current);
        setIsPaused(true);
      });
      recorder.addEventListener('resume', () => {
        setIsPaused(false);
        startRecordingClock();
      });
      recorder.start();
      startRecordingClock();
    } catch {
      setError({ message: t('Allow microphone access to record a voice message.') });
    }
  };

  const toggleRecordingPause = () => {
    const recorder = recorderRef.current;
    if (!recorder || recordingSeconds >= 60) return;
    try {
      if (recorder.state === 'recording') recorder.pause();
      else if (recorder.state === 'paused') recorder.resume();
    } catch {
      setError({ message: t('Voice recording could not be paused. Try recording again.') });
    }
  };

  useEffect(() => () => {
    window.clearInterval(recordingTimerRef.current);
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.onstop = () => recorder.stream.getTracks().forEach((track) => track.stop());
      recorder.stop();
    }
  }, []);

  const showReceipts = async (messageId) => {
    setReceiptRows([]);
    setReceiptLoading(true);
    try { setReceiptRows(await api.chatMessageReceipts(school, messageId) || []); }
    catch (failure) { setError(failure); setReceiptRows(null); }
    finally { setReceiptLoading(false); }
  };

  const loadEarlier = async () => {
    if (!messages.length || loadingEarlier) return;
    setLoadingEarlier(true);
    stickToBottom.current = false;
    setError(null);
    try {
      const earlier = await api.chatMessages(school, activeId, messages[0].id);
      setMessages((items) => mergeMessages(earlier || [], items));
    } catch (failure) {
      setError(failure);
    } finally {
      setLoadingEarlier(false);
    }
  };

  const chooseCategory = (category) => {
    setCategoryFilter(category);
    if (['school', 'leadership', 'direct'].includes(category)) setScopeFilter('all');
  };
  const closeProfile = useCallback(() => setProfilePerson(null), []);

  const EDIT_WINDOW_MS = 3600000; // 1 hour
  const canEditMessage = (message) => {
    if (!message || message.deleted) return false;
    if (message.sender_user_id !== currentUserId) return false;
    const msgTime = parseUtcDate(message.created_at).getTime();
    if (Number.isNaN(msgTime)) return false;
    return (Date.now() - msgTime) < EDIT_WINDOW_MS;
  };

  const startEditMessage = (message) => {
    setSavedDraft(draft);
    setDraft(message.body);
    setEditingMessage(message);
    window.requestAnimationFrame(() => composerRef.current?.focus());
  };

  const cancelEdit = () => {
    setDraft(savedDraft);
    setSavedDraft('');
    setEditingMessage(null);
  };

  const submitEdit = async () => {
    const text = draft.trim();
    if (!text || !editingMessage) return;
    setSending(true);
    try {
      const updated = await api.editChatMessage(school, activeId, editingMessage.id, text);
      setMessages((items) => items.map((item) => item.id === updated.id ? updated : item));
      setDraft('');
      setSavedDraft('');
      setEditingMessage(null);
      await refreshConversations(true);
    } catch (failure) {
      setError(failure);
    } finally {
      setSending(false);
    }
  };

  return (
    <section className={`sis-chat-shell${active ? ' has-active-chat' : ''}`} aria-label={t('Staff messages')}>
      <aside className="sis-chat-sidebar">
        <div className="sis-chat-sidebar-head">
          <div>
            <h1>{t('Messages')}</h1>
            <p>{t('Your school groups are updated automatically.')}</p>
          </div>
        </div>

        {mayWrite ? <div className="sis-chat-search">
          <Icon name="search" size={17} />
          <label className="visually-hidden" htmlFor="staff-chat-search">{t('Search staff')}</label>
          <input
            id="staff-chat-search"
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder={t('Search for a staff member')}
            autoComplete="off"
          />
        </div> : null}

        {search.trim().length < 2 && conversations.length ? (
          <div className="sis-chat-filters" aria-label={t('Filter conversations')}>
            <div className="sis-chat-filter-chips">
              {categories.map((category) => (
                <button
                  type="button"
                  key={category}
                  className={categoryFilter === category ? 'is-active' : ''}
                  aria-pressed={categoryFilter === category}
                  onClick={() => chooseCategory(category)}
                >
                  {filterLabel(category)}
                </button>
              ))}
            </div>
            {scopes.length > 1 && !['school', 'leadership', 'direct'].includes(categoryFilter) ? (
              <label className="sis-chat-scope-filter">
                <span>{t('Filter by grade')}</span>
                <select value={scopeFilter} onChange={(event) => setScopeFilter(event.target.value)}>
                  <option value="all">{t('All grade levels')}</option>
                  {scopes.map((scope) => (
                    <option key={scope.code} value={scope.code}>{localName(scope, state.lang, 'name') || scope.code}</option>
                  ))}
                </select>
              </label>
            ) : null}
          </div>
        ) : null}

        {search.trim().length >= 2 ? (
          <div className="sis-chat-search-results" aria-live="polite">
            {searchError ? <p className="sis-chat-error" role="alert">{searchError.message}</p> : null}
            {!searchError && !people.length ? <p className="sis-chat-muted">{t('No staff found')}</p> : null}
            {people.map((person) => (
              <button type="button" key={person.user_id} onClick={() => openPerson(person)} disabled={openingPerson === person.user_id}>
                <span className="sis-chat-avatar is-direct"><Icon name="people" size={17} /><span className={`sis-chat-presence-dot${person.online ? ' is-online' : ''}`} aria-hidden="true" /></span>
                <span><strong>{localName(person, state.lang, 'full_name')}</strong><small>{localName(person, state.lang, 'role_caption') || `@${person.username}`}</small></span>
              </button>
            ))}
          </div>
        ) : (
          <div className="sis-chat-conversations" aria-label={t('Conversations')}>
            {loadingConversations ? <p className="sis-chat-muted">{t('Loading conversations…')}</p> : null}
            {!loadingConversations && !visibleConversations.length ? <p className="sis-chat-muted">{t('No conversations match these filters')}</p> : null}
            {visibleConversations.map((conversation) => (
              <ConversationButton
                key={conversation.id}
                conversation={conversation}
                active={conversation.id === activeId}
                lang={state.lang}
                onSelect={setActiveId}
              />
            ))}
          </div>
        )}
      </aside>

      <div className="sis-chat-thread">
        {active ? (
          <>
            <header className="sis-chat-thread-head">
              <button type="button" className="sis-chat-back" onClick={() => setActiveId(null)} aria-label={t('Back to conversations')}>
                <span aria-hidden="true">←</span>
              </button>
              <span className={`sis-chat-avatar is-${active.category}`}><Icon name={active.kind === 'direct' ? 'people' : 'chat'} size={18} />{active.kind === 'direct' && !active.observer_view ? <span className={`sis-chat-presence-dot${headerPerson?.online ? ' is-online' : ''}`} aria-hidden="true" /> : null}</span>
              {active.kind === 'direct' && active.observer_view ? (
                <div className="sis-chat-group-title">
                  <h2>{localName(active, state.lang)}</h2>
                  <p>{localName(active, state.lang, 'subtitle') || categoryLabel(active.category)}</p>
                </div>
              ) : active.kind === 'direct' ? (
                <button type="button" className="sis-chat-person-title" onClick={() => setProfilePerson(headerPerson)}>
                  <h2>{localName(active, state.lang)}</h2>
                  {activePeer?.typing ? <TypingIndicator names={[localName(activePeer, state.lang, 'full_name')]} compact /> : <p>{localName(headerPerson, state.lang, 'role_caption') || categoryLabel(active.category)} · {headerPerson?.online ? t('Online') : t('Offline')}</p>}
                </button>
              ) : (
                <div className="sis-chat-group-title">
                  <h2>{localName(active, state.lang)}</h2>
                  {typingMembers.length ? <TypingIndicator names={typingMembers.map((member) => localName(member, state.lang, 'full_name'))} compact /> : <p>{categoryLabel(active.category)}</p>}
                </div>
              )}
              <div className="sis-chat-mute-menu">
                <button
                  type="button"
                  className={`sis-chat-mute-trigger${active.is_muted ? ' is-active' : ''}`}
                  onClick={() => setMuteMenuOpen((open) => !open)}
                  aria-label={active.is_muted ? t('Unmute notifications') : t('Mute notifications')}
                  aria-expanded={muteMenuOpen}
                  aria-haspopup="menu"
                  disabled={savingMute}
                >
                  <Icon name={active.is_muted ? 'bellOff' : 'bell'} size={18} />
                </button>
                {muteMenuOpen ? (
                  <div className="sis-chat-mute-popover" role="menu" aria-label={t('Mute notifications')}>
                    <strong>{active.is_muted ? t('Notifications muted') : t('Mute notifications')}</strong>
                    {active.is_muted ? (
                      <button type="button" role="menuitem" onClick={() => changeMute('off')}>{t('Unmute notifications')}</button>
                    ) : <>
                      <button type="button" role="menuitem" onClick={() => changeMute('eight_hours')}>{t('For 8 hours')}</button>
                      <button type="button" role="menuitem" onClick={() => changeMute('one_week')}>{t('For one week')}</button>
                      <button type="button" role="menuitem" onClick={() => changeMute('always')}>{t('Always')}</button>
                    </>}
                  </div>
                ) : null}
              </div>
            </header>

            <div
              ref={messagesRef}
              className="sis-chat-messages"
              aria-live="polite"
              aria-busy={loadingMessages}
              onScroll={(event) => {
                const node = event.currentTarget;
                stickToBottom.current = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
              }}
            >
              {messages.length >= 50 ? (
                <button type="button" className="sis-chat-earlier" onClick={loadEarlier} disabled={loadingEarlier}>
                  {loadingEarlier ? t('Loading…') : t('Load earlier messages')}
                </button>
              ) : null}
              {loadingMessages ? <p className="sis-chat-muted">{t('Loading messages…')}</p> : null}
              {!loadingMessages && !messages.length ? (
                <div className="sis-chat-empty">
                  <Icon name="chat" size={28} />
                  <strong>{t('Start the conversation')}</strong>
                  <span>{t('Messages sent here are visible only to this conversation’s current members.')}</span>
                </div>
              ) : null}
              {messages.map((message) => {
                const mine = message.sender_user_id === currentUserId;
                const sender = memberById.get(message.sender_user_id);
                const mediaOnly = !message.deleted && !message.body && message.attachments?.length;
                const voiceOnly = mediaOnly && message.attachments.length === 1 && message.attachments[0].kind === 'audio';
                const imageOnly = mediaOnly && message.attachments.length === 1 && message.attachments[0].kind === 'image';
                const fileOnly = mediaOnly && message.attachments.length === 1 && message.attachments[0].kind === 'file';
                return (
                  <article
                    className={`sis-chat-message${mine ? ' is-mine' : ''}${message.deleted ? ' is-deleted' : ''}${message.deleted && isAdmin ? ' is-deleted-audit' : ''}${mediaOnly ? ' is-media-only' : ''}${voiceOnly ? ' is-voice-only' : ''}${imageOnly ? ' is-image-only' : ''}${fileOnly ? ' is-file-only' : ''}`}
                    key={message.id}
                    onTouchStart={(event) => handleMessageTouchStart(message, event)}
                    onTouchMove={handleMessageTouchMove}
                    onTouchEnd={handleMessageTouchEnd}
                    onTouchCancel={handleMessageTouchEnd}
                    onContextMenu={(event) => {
                      event.preventDefault();
                      setSelectedActionMessage(message);
                    }}
                  >
                    {!mine ? <div className="sis-chat-sender-block"><button type="button" className="sis-chat-sender" onClick={() => sender && setProfilePerson(sender)} disabled={!sender}><strong>{localName(message, state.lang, 'sender_name')}</strong>{sender ? <span className={`sis-chat-presence-dot${sender.online ? ' is-online' : ''}`} aria-hidden="true" /> : null}</button>{sender ? <small>{localName(sender, state.lang, 'role_caption')}</small> : null}</div> : null}
                    {message.deleted ? <span className="sis-chat-deleted-label"><Icon name="trash" size={14} />{isAdmin ? t('Deleted message') : t('This message was deleted')}</span> : null}
                    {message.body ? <p>{message.body}</p> : null}
                    {isAdmin && message.original_body ? <button type="button" className="sis-chat-original-toggle" onClick={(e) => { const el = e.currentTarget.nextElementSibling; if (el) el.hidden = !el.hidden; }}><Icon name="eye" size={12} />{t('Show original')}</button> : null}
                    {isAdmin && message.original_body ? <p className="sis-chat-original-body" hidden>{message.original_body}</p> : null}
                    {message.attachments?.length ? <div className="sis-chat-attachments">{message.attachments.map((attachment) => <Attachment key={attachment.id} attachment={attachment} school={school} />)}</div> : null}
                    <footer>
                      <time dateTime={message.created_at}>{timeLabel(message.created_at, state.lang)}</time>
                      {message.edited ? <span className="sis-chat-edited-label">{t('edited')}</span> : null}
                      {mine && canEditMessage(message) ? <button type="button" className="sis-chat-edit-message" onClick={() => startEditMessage(message)} aria-label={t('Edit message')} title={t('Edit message')}><Icon name="pencil" size={13} /></button> : null}
                      {mine ? <ReceiptTicks summary={message.receipts} showCount={active.category !== 'direct'} onClick={() => showReceipts(message.id)} /> : null}
                      {mine && !message.deleted ? <button type="button" className="sis-chat-delete-message" onClick={() => requestMessageDelete(message)} aria-label={t('Delete message')} title={t('Delete message')}><Icon name="trash" size={15} /></button> : null}
                    </footer>
                  </article>
                );
              })}
              <div ref={endRef} />
            </div>

            {error ? <p className="sis-chat-error sis-chat-thread-error" role="alert">{error.message}</p> : null}
            {editingMessage ? <div className="sis-chat-editing-banner"><Icon name="pencil" size={14} /><span>{t('Editing message')}</span><button type="button" onClick={cancelEdit} aria-label={t('Cancel editing')}><Icon name="close" size={14} /></button></div> : null}
            {mayWrite ? <form className={`sis-chat-compose${isRecording ? ' is-recording' : ''}`} onSubmit={(event) => { event.preventDefault(); if (editingMessage) { submitEdit(); } else { send(); } }} onKeyDownCapture={handleComposerEnter}>
              <input ref={fileInputRef} className="visually-hidden" type="file" multiple accept="image/*,audio/*,.pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt,.csv,.zip" onChange={(event) => setSelectedFiles(Array.from(event.target.files || []).slice(0, 5))} />
              {selectedFiles.length ? <div className="sis-chat-selected-files">{selectedFiles.map((file, index) => <span key={`${file.name}-${index}`}><Icon name={file.type.startsWith('image/') ? 'eye' : file.type.startsWith('audio/') ? 'microphone' : 'file'} size={14} />{file.name}<button type="button" onClick={() => setSelectedFiles((items) => items.filter((_item, itemIndex) => itemIndex !== index))} aria-label={t('Remove {0}', [file.name])}><Icon name="close" size={13} /></button></span>)}</div> : null}
              {isRecording ? <div className={`sis-chat-recording${isPaused ? ' is-paused' : ''}`}>
                <button type="button" className="sis-record-delete" onClick={() => stopRecording(true)} aria-label={t('Delete recording')} title={t('Delete recording')}><Icon name="trash" size={18} /></button>
                <span className="sis-chat-record-dot" aria-hidden="true" />
                <strong>{durationLabel(recordingSeconds)}</strong>
                <div className="sis-record-wave" aria-hidden="true">{VOICE_WAVE.slice(0, 18).map((height, index) => <span key={index} style={{ height: `${Math.max(7, height * .65)}px`, animationDelay: `${index * -45}ms` }} />)}</div>
                <span className="sis-record-status">{recordingSeconds >= 60 ? t('One minute reached') : isPaused ? t('Recording paused') : t('Recording voice message')}</span>
                {recordingSeconds < 60 ? <button type="button" className="sis-record-pause" onClick={toggleRecordingPause} aria-label={isPaused ? t('Resume recording') : t('Pause recording')} title={isPaused ? t('Resume recording') : t('Pause recording')}><Icon name={isPaused ? 'play' : 'pause'} size={17} /></button> : null}
                <button type="button" className="sis-record-send" onClick={() => stopRecording(false, true)} disabled={recordingSeconds < 1} aria-label={t('Send recording')} title={t('Send recording')}><span className={`sis-chat-send-icon${state.lang === 'ar' ? ' is-rtl' : ''}`}><Icon name="send" size={18} /></span></button>
              </div> : <>
                <button type="button" className="sis-chat-tool" onClick={() => fileInputRef.current?.click()} aria-label={t('Attach files')}><Icon name="paperclip" size={19} /></button>
                <button type="button" className="sis-chat-tool" onClick={startRecording} aria-label={t('Record voice message')}><Icon name="microphone" size={19} /></button>
                <label className="visually-hidden" htmlFor="chat-message">{t('Message')}</label>
                <textarea id="chat-message" ref={composerRef} value={draft} maxLength={4000} rows={1} placeholder={t('Write a message')} onChange={(event) => setDraft(event.target.value)} />
                <button type="submit" className="sis-chat-submit" disabled={(!draft.trim() && !selectedFiles.length) || sending} aria-label={t('Send message')}>{sending ? <span className="spinner-border spinner-border-sm" aria-hidden="true" /> : <span className={`sis-chat-send-icon${state.lang === 'ar' ? ' is-rtl' : ''}`}><Icon name="send" size={19} weight={1.9} /></span>}</button>
              </>}
            </form> : <p className="sis-chat-read-only">{t('This conversation is read-only for your account.')}</p>}
          </>
        ) : (
          <div className="sis-chat-no-selection">
            <Icon name="chat" size={38} />
            <h2>{t('Your staff conversations')}</h2>
            <p>{t('Choose a group or search for a colleague to begin.')}</p>
            {error ? <p className="sis-chat-error" role="alert">{error.message}</p> : null}
          </div>
        )}
      </div>
      {selectedActionMessage ? (
        <MessageActionSheet
          message={selectedActionMessage}
          mine={selectedActionMessage.sender_user_id === currentUserId}
          canEdit={selectedActionMessage.sender_user_id === currentUserId && canEditMessage(selectedActionMessage)}
          canDelete={selectedActionMessage.sender_user_id === currentUserId && !selectedActionMessage.deleted}
          canShowReceipts={selectedActionMessage.sender_user_id === currentUserId || isAdmin}
          hasText={Boolean(selectedActionMessage.body)}
          isAdmin={isAdmin}
          lang={state.lang}
          onClose={() => setSelectedActionMessage(null)}
          onEdit={() => startEditMessage(selectedActionMessage)}
          onDelete={() => requestMessageDelete(selectedActionMessage)}
          onInfo={() => showReceipts(selectedActionMessage.id)}
          onCopy={() => handleCopyMessage(selectedActionMessage)}
        />
      ) : null}
      {receiptRows !== null ? <ReceiptDetails rows={receiptRows} lang={state.lang} loading={receiptLoading} onClose={() => setReceiptRows(null)} /> : null}
      {profilePerson ? <StaffProfile person={profilePerson} lang={state.lang} onClose={closeProfile} /> : null}
      {deleteDialog}
    </section>
  );
}
