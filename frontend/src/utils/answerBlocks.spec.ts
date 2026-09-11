/**
 * How a drawn record reads what arrived, and how it lays the week out.
 *
 * Everything that decides anything about a timetable or a set of marks lives in
 * `answerBlocks.ts` rather than in the components, because these tests run in node with no
 * DOM. The components only bind what these functions return.
 */
import { describe, expect, it } from 'vitest';

import type { AnswerBlock, TimetableBlockData } from '@/types/chat';
import {
  blockLanguage,
  buildGradeRows,
  buildTimetableView,
  minutesOf,
  readAnswerBlocks,
  swipeStep,
  tabLabel,
} from './answerBlocks';

/**
 * The shape of the timetable in the request that started this feature: Sunday-first, bells
 * at 45-minute steps from 07:45, and a break between the fourth and fifth lessons.
 */
const WEEK: TimetableBlockData = {
  class_label: 'الثالث أ',
  term_label: 'الفصل الأول',
  periods: [
    { number: 1, starts_at: '07:45', ends_at: '08:30' },
    { number: 2, starts_at: '08:30', ends_at: '09:15' },
    { number: 3, starts_at: '09:15', ends_at: '10:00' },
    { number: 4, starts_at: '10:00', ends_at: '10:45' },
    { number: 5, label: 'استراحة', starts_at: '10:45', ends_at: '11:15', is_teaching: false },
    { number: 6, starts_at: '11:15', ends_at: '12:00' },
  ],
  days: [
    {
      day: 'sunday',
      label: 'الأحد',
      slots: [
        { period: 1, subject: 'الكيمياء' },
        { period: 2, subject: 'اللغة الإنجليزية' },
        { period: 3, subject: 'اللغة العربية' },
        { period: 4, subject: 'الفيزياء' },
        { period: 6, subject: 'رياضة 1' },
      ],
    },
    {
      day: 'monday',
      label: 'الاثنين',
      slots: [
        { period: 1, subject: 'الأحياء' },
        { period: 2, subject: '', is_free: true },
      ],
    },
    {
      day: 'thursday',
      label: 'الخميس',
      slots: [
        { period: 1, subject: 'اللغة الإنجليزية' },
        { period: 6, subject: 'رياضة 1' },
      ],
    },
  ],
};

// Local times, which is what the component reads. 13 September 2026 is a Sunday.
const SUNDAY = (hours: number, minutes = 0) => new Date(2026, 8, 13, hours, minutes);
const TUESDAY_MORNING = new Date(2026, 8, 15, 9, 0);
const FRIDAY_MORNING = new Date(2026, 8, 18, 8, 0);

const block = (overrides: Partial<Record<string, unknown>> = {}) => ({
  kind: 'timetable',
  index: 0,
  language: 'ar',
  data: WEEK,
  ...overrides,
});

describe('reading blocks off the wire', () => {
  it('keeps a block it can draw', () => {
    const [read] = readAnswerBlocks([block()]);
    expect(read).toMatchObject({ kind: 'timetable', index: 0, language: 'ar' });
    expect((read.data as TimetableBlockData).days.map((day) => day.day)).toEqual([
      'sunday',
      'monday',
      'thursday',
    ]);
  });

  it.each([
    ['a kind this client does not draw', block({ kind: 'attendance' })],
    ['a kind that is only an object key', block({ kind: 'toString' })],
    ['no index', block({ index: undefined })],
    ['a negative index', block({ index: -1 })],
    ['a week with no day on it', block({ data: { periods: [], days: [] } })],
    ['data that is not an object', block({ data: 'rows' })],
  ])('leaves out %s, whose marker then prints as markdown', (_label, candidate) => {
    expect(readAnswerBlocks([candidate])).toEqual([]);
  });

  it('reads nothing out of something that is not a list', () => {
    expect(readAnswerBlocks(undefined)).toEqual([]);
    expect(readAnswerBlocks({ kind: 'timetable' })).toEqual([]);
  });

  it('never turns a blank grade into a zero', () => {
    const [read] = readAnswerBlocks([
      { kind: 'grades', index: 0, data: { courses: [{ subject: 'الأحياء' }, { subject: 'العلوم', percentage: 0 }] } },
    ]);
    expect(read.kind).toBe('grades');
    if (read.kind !== 'grades') return;
    expect(read.data.courses[0].percentage).toBeNull();
    // A real 0% stays a real 0%.
    expect(read.data.courses[1].percentage).toBe(0);
  });
});

describe('which language a block is labelled in', () => {
  const asBlock = (value: unknown) => readAnswerBlocks([value])[0] as AnswerBlock;

  it('follows the turn when the backend named one', () => {
    expect(blockLanguage(asBlock(block({ language: 'ar' })))).toBe('ar');
    // Arabic subject names in an English turn: the labels still follow the turn.
    expect(blockLanguage(asBlock(block({ language: 'en' })))).toBe('en');
  });

  it('falls back to the day names when the turn named no language', () => {
    expect(blockLanguage(asBlock(block({ language: '' })))).toBe('ar');
    // What the backend sends for a turn with no language: day names in English, subjects
    // and class still the school's Arabic. The labels follow the days, so the table never
    // reads «اليوم» beside "Sunday".
    const englishDays = {
      ...WEEK,
      days: [{ day: 'sunday', label: 'Sunday', slots: [{ period: 1, subject: 'الكيمياء' }] }],
    };
    expect(blockLanguage(asBlock(block({ language: '', data: englishDays })))).toBe('en');
  });

  it('reads marks by their subjects, which are all they carry', () => {
    const grades = (subject: string) =>
      asBlock({ kind: 'grades', index: 0, data: { courses: [{ subject, percentage: 80 }] } });
    expect(blockLanguage(grades('اللغة العربية'))).toBe('ar');
    expect(blockLanguage(grades('Mathematics'))).toBe('en');
  });
});

describe('the week, laid out', () => {
  it("keeps the school's day in its own order, the break in its place", () => {
    const view = buildTimetableView(WEEK, SUNDAY(9));
    expect(view.columns.map((column) => `${column.kind}:${column.number}`)).toEqual([
      'period:1',
      'period:2',
      'period:3',
      'period:4',
      'break:5',
      'period:6',
    ]);
    expect(view.days.map((day) => day.label)).toEqual(['الأحد', 'الاثنين', 'الخميس']);
  });

  it('gives every day one cell per column, so the grid lines up', () => {
    const view = buildTimetableView(WEEK, SUNDAY(9));
    view.days.forEach((day) => expect(day.cells).toHaveLength(view.columns.length));
    const monday = view.days[1];
    expect(monday.cells.map((cell) => cell.kind)).toEqual(['lesson', 'lesson', 'empty', 'empty', 'break', 'empty']);
  });

  it("puts a break between a day's lessons, and never after its last one", () => {
    const view = buildTimetableView(WEEK, SUNDAY(9));
    const kinds = (index: number) => view.days[index].items.map((item) => item.kind);
    expect(kinds(0)).toEqual(['lesson', 'lesson', 'lesson', 'lesson', 'break', 'lesson']);
    // Monday's lessons are over before the break: it is not part of her day.
    expect(kinds(1)).toEqual(['lesson', 'lesson']);
    expect(kinds(2)).toEqual(['lesson', 'break', 'lesson']);
  });

  it('keeps a free period a free period, with its times', () => {
    const [, free] = buildTimetableView(WEEK, SUNDAY(9)).days[1].items;
    expect(free).toMatchObject({ kind: 'lesson', isFree: true, startsAt: '08:30', endsAt: '09:15' });
  });

  it('gives a lesson on a period the grid does not list a column of its own', () => {
    const data = { ...WEEK, days: [{ day: 'sunday', label: 'الأحد', slots: [{ period: 9, subject: 'رسم' }] }] };
    const view = buildTimetableView(data, SUNDAY(9));
    expect(view.columns.at(-1)).toMatchObject({ kind: 'period', number: 9, startsAt: '' });
    expect(view.days[0].items.at(-1)).toMatchObject({ kind: 'lesson', period: 9, subject: 'رسم' });
  });

  it('shows a slot before the first lesson, since the day already runs through it', () => {
    // A morning assembly before the first bell: part of her day, and when she must arrive.
    const data: TimetableBlockData = {
      periods: [
        { number: 0, label: 'طابور', starts_at: '07:30', ends_at: '07:45', is_teaching: false },
        { number: 1, starts_at: '07:45', ends_at: '08:30' },
      ],
      days: [{ day: 'sunday', label: 'الأحد', slots: [{ period: 1, subject: 'الكيمياء' }] }],
    };
    const items = buildTimetableView(data, SUNDAY(9)).days[0].items;
    expect(items.map((item) => item.kind)).toEqual(['break', 'lesson']);
  });

  it('marks today and tomorrow against the calendar', () => {
    const view = buildTimetableView(WEEK, SUNDAY(9));
    expect(view.days.map((day) => day.relative)).toEqual(['today', 'tomorrow', null]);
  });

  it('flags the lesson on right now, and only on today', () => {
    const view = buildTimetableView(WEEK, SUNDAY(8, 0));
    expect(view.days[0].items[0]).toMatchObject({ subject: 'الكيمياء', isNow: true });
    expect(view.days[0].items.slice(1).some((item) => item.isNow)).toBe(false);
    // Monday's first lesson is at the same hour, but it is not Monday.
    expect(view.days[1].items[0].isNow).toBe(false);
  });

  it('flags the break when that is where the day is', () => {
    const view = buildTimetableView(WEEK, SUNDAY(10, 50));
    expect(view.columns[4]).toMatchObject({ kind: 'break', isNow: true });
    expect(view.days[0].items[4]).toMatchObject({ kind: 'break', isNow: true });
  });

  it('flags nothing on a day the school is shut', () => {
    const view = buildTimetableView(WEEK, FRIDAY_MORNING);
    expect(view.columns.some((column) => column.isNow)).toBe(false);
  });
});

describe('the day a phone opens on', () => {
  it('is today while its lessons last', () => {
    expect(buildTimetableView(WEEK, SUNDAY(11, 30)).initialIndex).toBe(0);
  });

  it('is the next school day once the last bell has gone', () => {
    // Sunday's last lesson ends at 12:00; a parent looking at 13:00 is looking at Monday.
    expect(buildTimetableView(WEEK, SUNDAY(13)).initialIndex).toBe(1);
  });

  it('skips the days the school does not open', () => {
    // Tuesday and Wednesday are not in this week, so the next school day is Thursday.
    expect(buildTimetableView(WEEK, TUESDAY_MORNING).initialIndex).toBe(2);
    // Friday: Saturday is not a school day here, Sunday is.
    expect(buildTimetableView(WEEK, FRIDAY_MORNING).initialIndex).toBe(0);
  });

  it('stays on today when the school has not fixed its bell times', () => {
    const untimed = { ...WEEK, periods: [] };
    expect(buildTimetableView(untimed, SUNDAY(22)).initialIndex).toBe(0);
  });
});

describe('the day buttons on a phone', () => {
  it('shortens the calendar weekdays so a school week fits across a phone', () => {
    expect(tabLabel({ key: 'wednesday', label: 'الأربعاء' }, 'ar')).toBe('أربعاء');
    expect(tabLabel({ key: 'Thursday', label: 'Thursday' }, 'en')).toBe('Thu');
  });

  it("keeps a school's own name for a day that is not a weekday", () => {
    expect(tabLabel({ key: 'day-1', label: 'اليوم الأول' }, 'ar')).toBe('اليوم الأول');
  });
});

describe('swiping between days', () => {
  it('moves forward against the reading direction, as a carousel does', () => {
    expect(swipeStep(-80, 4, 'ltr')).toBe(1);
    expect(swipeStep(80, 4, 'ltr')).toBe(-1);
    // Right to left, the next day is on the left: a rightward swipe brings it in.
    expect(swipeStep(80, 4, 'rtl')).toBe(1);
    expect(swipeStep(-80, 4, 'rtl')).toBe(-1);
  });

  it('ignores a nudge, and leaves a vertical movement to the chat', () => {
    expect(swipeStep(20, 0, 'ltr')).toBe(0);
    expect(swipeStep(60, 90, 'rtl')).toBe(0);
  });
});

describe('the term’s marks', () => {
  it('shows each figure as the school reported it', () => {
    const [arabic, extra, biology] = buildGradeRows({
      courses: [
        { subject: 'اللغة العربية', percentage: 84.5, letter: 'B', missing_count: 1 },
        { subject: 'الرياضيات', percentage: 105 },
        { subject: 'الأحياء', percentage: null, in_progress: true },
      ],
    });
    expect(arabic).toMatchObject({ hasGrade: true, percentText: '84.5', fill: 84.5, letter: 'B', missing: 1 });
    // Beyond the scale: the bar stops at full, the figure is left exactly as reported.
    expect(extra).toMatchObject({ percentText: '105', fill: 100 });
    // A blank has no figure and no bar — an empty bar would read as a zero.
    expect(biology).toMatchObject({ hasGrade: false, percentText: '', inProgress: true });
  });
});

describe('bell times', () => {
  it.each([
    ['07:45', 465],
    ['07:45:00', 465],
    ['7:05', 425],
    ['', null],
    ['25:00', null],
    ['soon', null],
  ])('reads %s', (clock, minutes) => {
    expect(minutesOf(clock)).toBe(minutes);
  });
});
