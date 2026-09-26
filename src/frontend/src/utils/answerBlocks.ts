/**
 * Records a tool rendered for the reader — drawn rather than printed.
 *
 * A timetable or a term's marks reaches the client twice: as markdown inside the answer,
 * after a `<!--record-block-->` marker, and as an `AnswerBlock`, the same record as data
 * (backend/schemas/chat.py). The backend has already decided everything that is a matter
 * of FACT — which rows, which figures, in which order. What is left here is presentation:
 * which day a phone opens on, where a break sits in the day, what "now" is.
 *
 * Pure throughout. This project's tests run in node with no DOM, so the components stay
 * thin and everything worth testing lives here.
 */
import type {
  AnswerBlock,
  AnswerBlockKind,
  GradeRow,
  GradesBlockData,
  TimetableBlockData,
  TimetableDay,
  TimetablePeriod,
  TimetableSlot,
} from '@/types/chat';

// ---------------------------------------------------------------------------------------
// Reading what arrived
//
// A block comes off the network or out of a stored trace, and it goes straight into a
// component. The backend validates it, but this client also reads traces written by
// older servers and will read ones written by newer ones — so it is read defensively
// here, once, and a block that does not read is dropped. Dropping costs nothing: its
// marker then shows as the markdown it also travels as.
// ---------------------------------------------------------------------------------------

type Loose = Record<string, unknown>;

const isRecord = (value: unknown): value is Loose =>
  !!value && typeof value === 'object' && !Array.isArray(value);

const records = (value: unknown): Loose[] => (Array.isArray(value) ? value.filter(isRecord) : []);

const text = (value: unknown): string => (typeof value === 'string' ? value : '');

const wholeNumber = (value: unknown): number | null => {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() ? Number(value) : NaN;
  return Number.isInteger(n) ? n : null;
};

function readTimetable(data: unknown): TimetableBlockData | null {
  if (!isRecord(data)) return null;
  const days: TimetableDay[] = records(data.days)
    .map((day) => ({
      day: text(day.day),
      label: text(day.label),
      slots: records(day.slots)
        .map((slot): TimetableSlot | null => {
          const period = wholeNumber(slot.period);
          return period === null ? null : { period, subject: text(slot.subject), is_free: slot.is_free === true };
        })
        .filter((slot): slot is TimetableSlot => slot !== null),
    }))
    .filter((day) => day.slots.length > 0);
  if (!days.length) return null;

  const periods: TimetablePeriod[] = records(data.periods)
    .map((period): TimetablePeriod | null => {
      const number = wholeNumber(period.number);
      return number === null
        ? null
        : {
            number,
            label: text(period.label),
            starts_at: text(period.starts_at),
            ends_at: text(period.ends_at),
            is_teaching: period.is_teaching !== false,
          };
    })
    .filter((period): period is TimetablePeriod => period !== null);

  return { class_label: text(data.class_label), term_label: text(data.term_label), periods, days };
}

function readGrades(data: unknown): GradesBlockData | null {
  if (!isRecord(data)) return null;
  const courses: GradeRow[] = records(data.courses).map((course) => ({
    subject: text(course.subject),
    // Absent and null both mean "no grade recorded yet". Never coerced to zero.
    percentage:
      typeof course.percentage === 'number' && Number.isFinite(course.percentage) ? course.percentage : null,
    letter: text(course.letter),
    missing_count: Math.max(0, wholeNumber(course.missing_count) ?? 0),
    in_progress: course.in_progress === true,
  }));
  if (!courses.length) return null;
  return { term_label: text(data.term_label), courses };
}

/** One reader per kind this client draws. A kind missing here is shown as its markdown. */
const READERS: { [K in AnswerBlockKind]: (data: unknown) => Extract<AnswerBlock, { kind: K }>['data'] | null } = {
  timetable: readTimetable,
  grades: readGrades,
};

export const ANSWER_BLOCK_KINDS = Object.keys(READERS) as AnswerBlockKind[];

/** The blocks this client can draw, each checked; anything else is left out. */
export function readAnswerBlocks(value: unknown): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  for (const item of records(value)) {
    const kind = text(item.kind);
    const index = wholeNumber(item.index);
    if (!Object.prototype.hasOwnProperty.call(READERS, kind) || index === null || index < 0) continue;
    const data = READERS[kind as AnswerBlockKind](item.data);
    if (!data) continue;
    blocks.push({ kind, index, language: text(item.language), data } as AnswerBlock);
  }
  return blocks;
}

// ---------------------------------------------------------------------------------------
// The block's own words
// ---------------------------------------------------------------------------------------

export type BlockLanguage = 'ar' | 'en';

const ARABIC_SCRIPT = /[؀-ۿݐ-ݿࢠ-ࣿ]/;

/**
 * The words in a record that were written for THIS reader, as opposed to the school's own.
 *
 * A timetable's day names are rendered in the turn's language (`_day_label` in
 * backend/tools/records.py) while its subjects are always the school's Arabic-first names.
 * So the days are what say which language the record was put into — not the subjects, and
 * not the class name. Marks carry no such words; their subjects are all there is.
 */
function readerWords(block: AnswerBlock): string[] {
  return block.kind === 'timetable'
    ? block.data.days.map((day) => day.label || '')
    : block.data.courses.map((course) => course.subject);
}

/**
 * The language to label a block in, and so its direction.
 *
 * The turn's language when the backend established one. Otherwise the record's own
 * reader-facing words decide, so the block's labels match its days: «الأحد» gets «اليوم»
 * and right-to-left, "Sunday" gets "Today" — never a mix of the two in one table.
 */
export function blockLanguage(block: AnswerBlock): BlockLanguage {
  const declared = (block.language || '').toLowerCase();
  if (declared.startsWith('ar')) return 'ar';
  if (declared) return 'en';
  return readerWords(block).some((word) => ARABIC_SCRIPT.test(word)) ? 'ar' : 'en';
}

export interface BlockCopy {
  timetable: string;
  days: string;
  day: string;
  today: string;
  tomorrow: string;
  now: string;
  free: string;
  break: string;
  grades: string;
  noGrade: string;
  inProgress: string;
  missing: (count: number) => string;
}

const COPY: Record<BlockLanguage, BlockCopy> = {
  ar: {
    timetable: 'الجدول الدراسي',
    days: 'أيام الأسبوع',
    day: 'اليوم',
    today: 'اليوم',
    tomorrow: 'غدًا',
    now: 'الآن',
    free: 'حصة فراغ',
    break: 'استراحة',
    grades: 'الدرجات',
    noGrade: 'لم تُرصد بعد',
    inProgress: 'لم تكتمل بعد',
    missing: (count) => `${count} غير مسلّم`,
  },
  en: {
    timetable: 'Timetable',
    days: 'Days of the week',
    day: 'Day',
    today: 'Today',
    tomorrow: 'Tomorrow',
    now: 'Now',
    free: 'Free period',
    break: 'Break',
    grades: 'Grades',
    noGrade: 'No grade yet',
    inProgress: 'In progress',
    missing: (count) => `${count} not submitted`,
  },
};

export const blockCopy = (language: BlockLanguage): BlockCopy => COPY[language];

// ---------------------------------------------------------------------------------------
// The week, laid out
// ---------------------------------------------------------------------------------------

/** `Date.getDay()` order, which is what a school's day key is matched against. */
const WEEKDAYS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];

/** Weekday names short enough for a school week of them to share a phone's width. */
const SHORT_WEEKDAYS: Record<BlockLanguage, Record<string, string>> = {
  ar: {
    sunday: 'أحد',
    monday: 'اثنين',
    tuesday: 'ثلاثاء',
    wednesday: 'أربعاء',
    thursday: 'خميس',
    friday: 'جمعة',
    saturday: 'سبت',
  },
  en: {
    sunday: 'Sun',
    monday: 'Mon',
    tuesday: 'Tue',
    wednesday: 'Wed',
    thursday: 'Thu',
    friday: 'Fri',
    saturday: 'Sat',
  },
};

/**
 * A day's name on the phone's row of day buttons.
 *
 * Five full Arabic day names need more width than a phone's chat column has, and a row that
 * scrolls hides the end of the week. The calendar's own short forms fit. Only for the
 * calendar's weekdays, matched on the school's day KEY: a school whose days are named
 * anything else ("Day 1") keeps its own name, because no abbreviation of it would be
 * recognised. The full name stays in the button's accessible label and everywhere else.
 */
export function tabLabel(day: { key: string; label: string }, language: BlockLanguage): string {
  return SHORT_WEEKDAYS[language][dayKey(day.key)] || day.label;
}

const dayKey = (day: string): string => (day || '').trim().toLowerCase();

/** `07:45` as minutes since midnight, or null when the school has not fixed the bell. */
export function minutesOf(clock: string | undefined): number | null {
  const match = /^(\d{1,2}):(\d{2})/.exec((clock || '').trim());
  if (!match) return null;
  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  return hours < 24 && minutes < 60 ? hours * 60 + minutes : null;
}

export interface TimetableColumn {
  kind: 'period' | 'break';
  number: number;
  label: string;
  startsAt: string;
  endsAt: string;
  /** The slot the school day is in right now — only ever true on a school day. */
  isNow: boolean;
}

export type TimetableCell =
  | { kind: 'lesson'; subject: string; isFree: boolean; isNow: boolean }
  | { kind: 'break'; isNow: boolean }
  | { kind: 'empty' };

export type TimetableItem =
  | { kind: 'lesson'; period: number; subject: string; isFree: boolean; startsAt: string; endsAt: string; isNow: boolean }
  | { kind: 'break'; label: string; startsAt: string; endsAt: string; isNow: boolean };

export interface TimetableDayView {
  key: string;
  label: string;
  relative: 'today' | 'tomorrow' | null;
  /** One per column, for the grid. */
  cells: TimetableCell[];
  /** The day read top to bottom, for the phone: its lessons, and the breaks between them. */
  items: TimetableItem[];
}

export interface TimetableView {
  columns: TimetableColumn[];
  days: TimetableDayView[];
  /** The day a phone opens on. See `openingDay`. */
  initialIndex: number;
}

/**
 * The week as a grid and as a list of days, from one pass over the record.
 *
 * Columns are the school's day in its own order, breaks included: the facade says a
 * client drawing the day must show a break, and a parent asks when it is. A lesson on a
 * period the grid does not list still gets its column — hiding a lesson the school
 * timetabled would be worse than an untidy grid.
 *
 * On the phone a non-teaching slot is shown wherever a lesson still follows it, and never
 * after the last one. A morning assembly before the first bell is part of her day — it is
 * when she has to arrive — but once her last lesson is over, the break the older years
 * take is not.
 */
export function buildTimetableView(data: TimetableBlockData, now: Date = new Date()): TimetableView {
  const todayKey = WEEKDAYS[now.getDay()];
  const tomorrowKey = WEEKDAYS[(now.getDay() + 1) % 7];
  const minute = now.getHours() * 60 + now.getMinutes();
  const schoolDayToday = data.days.some((day) => dayKey(day.day) === todayKey);

  const byNumber = new Map<number, TimetablePeriod>();
  data.periods.forEach((period) => {
    if (!byNumber.has(period.number)) byNumber.set(period.number, period);
  });
  data.days.forEach((day) =>
    day.slots.forEach((slot) => {
      if (!byNumber.has(slot.period)) byNumber.set(slot.period, { number: slot.period, is_teaching: true });
    }),
  );

  const columns: TimetableColumn[] = [...byNumber.values()]
    .sort((a, b) => a.number - b.number)
    .map((period) => {
      const start = minutesOf(period.starts_at);
      const end = minutesOf(period.ends_at);
      return {
        kind: period.is_teaching === false ? 'break' : 'period',
        number: period.number,
        label: period.label || '',
        startsAt: period.starts_at || '',
        endsAt: period.ends_at || '',
        isNow: schoolDayToday && start !== null && end !== null && start <= minute && minute < end,
      };
    });

  const days: TimetableDayView[] = data.days.map((day) => {
    const key = dayKey(day.day);
    const relative = key === todayKey ? 'today' : key === tomorrowKey ? 'tomorrow' : null;
    const isToday = relative === 'today';
    const slots = new Map(day.slots.map((slot) => [slot.period, slot]));

    const cells: TimetableCell[] = columns.map((column): TimetableCell => {
      if (column.kind === 'break') return { kind: 'break', isNow: isToday && column.isNow };
      const slot = slots.get(column.number);
      return slot
        ? { kind: 'lesson', subject: slot.subject || '', isFree: !!slot.is_free, isNow: isToday && column.isNow }
        : { kind: 'empty' };
    });

    let lastLesson = -1;
    cells.forEach((cell, index) => {
      if (cell.kind === 'lesson') lastLesson = index;
    });

    const items: TimetableItem[] = [];
    cells.forEach((cell, index) => {
      const column = columns[index];
      if (cell.kind === 'lesson') {
        items.push({
          kind: 'lesson',
          period: column.number,
          subject: cell.subject,
          isFree: cell.isFree,
          startsAt: column.startsAt,
          endsAt: column.endsAt,
          isNow: cell.isNow,
        });
      } else if (cell.kind === 'break' && index < lastLesson) {
        items.push({
          kind: 'break',
          label: column.label,
          startsAt: column.startsAt,
          endsAt: column.endsAt,
          isNow: cell.isNow,
        });
      }
    });

    return { key, label: day.label || day.day, relative, cells, items };
  });

  return { columns, days, initialIndex: openingDay(days, now) };
}

/**
 * The day a parent most likely opened the timetable for.
 *
 * Today, while today's lessons last. After the last bell — or on a day the school is shut
 * — the next school day, because a parent checking in the evening is checking tomorrow.
 * A day whose lessons carry no times cannot be said to be over, so it stays today.
 */
function openingDay(days: TimetableDayView[], now: Date): number {
  const minute = now.getHours() * 60 + now.getMinutes();
  const today = days.findIndex((day) => day.relative === 'today');
  if (today >= 0) {
    const ends = days[today].items
      .filter((item) => item.kind === 'lesson')
      .map((item) => minutesOf(item.endsAt))
      .filter((end): end is number => end !== null);
    if (!ends.length || minute < Math.max(...ends)) return today;
  }
  for (let step = 1; step <= 7; step += 1) {
    const index = days.findIndex((day) => day.key === WEEKDAYS[(now.getDay() + step) % 7]);
    if (index >= 0) return index;
  }
  return 0;
}

/**
 * Which way a horizontal swipe moves through the days: +1, -1, or 0 for no swipe.
 *
 * "Forward" is toward the end of the row of days — leftward when the block reads right
 * to left. A finger moving against that direction pulls the next day in, as a carousel
 * would. Mostly-vertical movement is the chat scrolling and is never a swipe.
 */
export function swipeStep(dx: number, dy: number, dir: 'rtl' | 'ltr'): -1 | 0 | 1 {
  if (Math.abs(dx) < 48 || Math.abs(dx) < Math.abs(dy) * 1.5) return 0;
  const forward = dir === 'rtl' ? dx > 0 : dx < 0;
  return forward ? 1 : -1;
}

// ---------------------------------------------------------------------------------------
// The term's marks
// ---------------------------------------------------------------------------------------

export interface GradeRowView {
  subject: string;
  hasGrade: boolean;
  /** The figure as the school reported it — never rounded, recomputed or rescaled. */
  percentText: string;
  /** How much of the bar to fill, 0–100. Zero only for a real 0%, never for a blank. */
  fill: number;
  letter: string;
  missing: number;
  inProgress: boolean;
}

export function buildGradeRows(data: GradesBlockData): GradeRowView[] {
  return data.courses.map((course) => {
    const hasGrade = typeof course.percentage === 'number' && Number.isFinite(course.percentage);
    const percentage = hasGrade ? (course.percentage as number) : 0;
    return {
      subject: course.subject,
      hasGrade,
      percentText: hasGrade ? String(percentage) : '',
      fill: hasGrade ? Math.min(100, Math.max(0, percentage)) : 0,
      letter: course.letter || '',
      missing: course.missing_count || 0,
      inProgress: !!course.in_progress,
    };
  });
}
