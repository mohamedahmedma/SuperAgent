/*
 * Taking the register: one class, one day, one pass down the list.
 *
 * Shaped around what a form teacher actually does at 8am on a phone — read forty names once, tap
 * the two or three who are not there, save — so **the default for a child nobody has touched is
 * nothing, not present.** Defaulting to present would make the fastest possible action ("save
 * without reading") produce a full house every morning, and the register would be worthless in
 * the way that is hardest to notice.
 *
 * Three rules the panel keeps visible:
 *
 * **Unmarked is a third state.** A child with no mark shows a dash and is counted under "not yet
 * marked", never under present or absent. The service returns `state: null` for her and this
 * screen keeps that distinction all the way to the pixel.
 *
 * **Nothing is written until Save.** Tapping through forty children builds local state and one
 * request goes out. A save-per-tap would be forty requests, a spinner between each, and a
 * half-recorded register whenever the signal dropped in the middle — which on a school phone is
 * a real event rather than a hypothetical.
 *
 * **Saving twice corrects rather than duplicates.** The route is a PUT, so a teacher who fixes a
 * mark at lunchtime replaces the morning's statement instead of adding a second one.
 *
 * The four state buttons are a `btn-group` that goes full width on a phone, so each is a third
 * of the screen wide rather than a 40px target next to three others.
 */
import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Store } from '../store.js';
import { DASH, pickName, today, useAction, useQuery, useStore } from '../hooks.js';
import { Badge, Button, Card, Empty, ErrorNote, Input, Table, cx } from './Ui.jsx';
import { t } from '../i18n.js';

/* The four states, in the order a teacher reaches for them: present first because it is the
   overwhelming majority, excused last because it needs a reason typed. */
const STATES = [
  { key: 'present', label: 'Present', short: 'P' },
  { key: 'absent', label: 'Absent', short: 'A' },
  { key: 'late', label: 'Late', short: 'L' },
  { key: 'excused', label: 'Excused', short: 'E' }
];

export function AttendancePanel({ classCode, year, on }) {
  const state = useStore();
  const [day, setDay] = useState(on || today());
  /* Local edits, keyed by student number. Empty until a teacher taps something, and cleared
     when the day changes — carrying yesterday's taps into today would be the worst possible bug
     in this panel. */
  const [draft, setDraft] = useState({});

  const register = useQuery(
    () => api.classRegister(classCode, year, day),
    [classCode, year, day],
    !!(classCode && year)
  );

  useEffect(() => setDraft({}), [day, classCode]);

  const save = useAction((entries) => api.takeRegister(classCode, year, day, entries));

  const report = register.value;
  const lines = (report && report.students) || [];
  const pending = Object.keys(draft);

  function mark(number, next) {
    setDraft((current) => {
      const copy = { ...current };
      /* Tapping the state a child already has clears the draft entry rather than re-stating it:
         a teacher who taps twice by accident should end where they started, not send a no-op
         that stamps the row as freshly recorded. */
      if ((copy[number] || {}).state === next) delete copy[number];
      else copy[number] = { state: next, note: (copy[number] || {}).note || '' };
      return copy;
    });
  }

  function note(number, text) {
    setDraft((current) => ({
      ...current,
      [number]: { state: (current[number] || {}).state || 'excused', note: text }
    }));
  }

  /** What a row shows: the draft if the teacher touched it, else what is on file. */
  const shown = (line) => (draft[line.student_number] || {}).state ?? line.state;
  const shownNote = (line) => (draft[line.student_number] || {}).note ?? (line.note || '');

  /** Mark every child still blank, without overwriting anything already on file or edited. */
  function fillUntouched(value) {
    setDraft((current) => {
      const copy = { ...current };
      lines.forEach((line) => {
        if (copy[line.student_number] || line.state) return;
        copy[line.student_number] = { state: value, note: '' };
      });
      return copy;
    });
  }

  return (
    <Card
      title={t('Attendance')}
      subtitle={
        report
          ? `${report.counts.recorded} of ${report.size} marked on ${report.on_date}`
          : null
      }
      actions={
        <div className="d-flex align-items-center gap-2">
          <Input type="date" className="form-control-sm" value={day} onInput={setDay} />
          <Button size="sm" icon="refresh" onClick={register.reload}>
            {t('Reload')}
          </Button>
        </div>
      }
      footer={
        <div className="d-grid gap-2 d-sm-flex align-items-sm-center w-100">
          <Button
            variant="primary"
            disabled={!pending.length}
            pending={save.pending}
            pendingLabel={t('Saving…')}
            onClick={() =>
              save
                .run(
                  pending.map((number) => ({
                    student_number: number,
                    state: draft[number].state,
                    note: draft[number].note || ''
                  }))
                )
                .then((saved) => {
                  setDraft({});
                  register.reload();
                  Store.toast(
                    'ok',
                    `Register saved for ${day}`,
                    `${saved.counts.recorded} of ${saved.size} marked` +
                      (saved.unmarked ? ` — ${saved.unmarked} still blank` : '')
                  );
                })
                .catch(() => {})
            }
          >
            Save {pending.length || ''} mark(s)
          </Button>
          <Button variant="quiet" disabled={!pending.length} onClick={() => setDraft({})}>
            {t('Discard changes')}
          </Button>
          <span className="small text-body-tertiary">
            {pending.length
              ? `${pending.length} change(s) not yet saved.`
              : 'Nothing is written until you save. Saving again corrects the day rather than adding a second set of marks.'}
          </span>
        </div>
      }
      tight
    >
      <div className="card-body vstack gap-3">
        {report ? (
          <div className="d-flex flex-wrap gap-2">
            <Badge tone="ok">{report.counts.present} present</Badge>
            <Badge tone="bad">{report.counts.absent} absent</Badge>
            <Badge tone="warn">{report.counts.late} late</Badge>
            <Badge tone="info">{report.counts.excused} excused</Badge>
            <Badge>{report.unmarked} not yet marked</Badge>
          </div>
        ) : null}

        <p className="small text-body-tertiary mb-0">
          {t('A child nobody has marked shows')} <span className="sis-ungraded">{DASH}</span> {t('and is counted as')} <strong>{t('not yet marked')}</strong> — never as present. The counts above are over
          the {report ? report.counts.recorded : 0} mark(s) actually recorded, not over the{' '}
          {report ? report.size : 0} children on the register.
        </p>

        <div className="d-grid gap-2 d-sm-flex align-items-sm-center">
          <Button size="sm" onClick={() => fillUntouched('present')}>
            {t('Mark the rest present')}
          </Button>
          <span className="small text-body-tertiary">
            {t('Fills only the children still blank, and leaves every mark already on file alone.')}
          </span>
        </div>
      </div>

      <ErrorNote error={register.error} onRetry={register.reload} />
      <ErrorNote error={save.error} />

      <Table
        loading={register.loading}
        rows={lines}
        rowKey={(row) => row.student_number}
        rowTone={(row) => {
          const value = shown(row);
          if (!value) return null;
          if (value === 'present') return 'ok';
          if (value === 'late' || value === 'excused') return 'warn';
          return 'bad';
        }}
        empty={
          <Empty title={t('Nobody is on this register')}>
            A register is a statement about a day. Either this class is empty, or nobody was
            placed in it on {day}.
          </Empty>
        }
        columns={[
          {
            key: 'number',
            header: t('Student no.'),
            className: 'sis-code',
            hide: 'md',
            cell: (row) => row.student_number
          },
          {
            key: 'name',
            header: t('Child'),
            className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
            cell: (row) => (
              <>
                {pickName(row, state.lang) || (
                  <span className="sis-ungraded">{DASH} name not on file</span>
                )}
                {/* The number rides under the name on a phone, where its own column is gone. */}
                <div className="sis-code sis-xs text-body-tertiary d-md-none">
                  {row.student_number}
                </div>
              </>
            )
          },
          {
            key: 'state',
            header: t('Mark'),
            cell: (row) => {
              const value = shown(row);
              const dirty = !!draft[row.student_number];
              return (
                <div className="vstack gap-1">
                  <div className="btn-group btn-group-sm w-100" role="group">
                    {STATES.map((option) => (
                      <button
                        key={option.key}
                        className={cx(
                          'btn',
                          value === option.key
                            ? option.key === 'present'
                              ? 'btn-primary'
                              : 'btn-danger'
                            : 'btn-outline-secondary'
                        )}
                        onClick={() => mark(row.student_number, option.key)}
                        title={option.label}
                      >
                        {/* The initial on a phone, the word when there is room. Four full words
                            in a button group at 360px is four buttons of two characters each. */}
                        <span className="d-lg-none">{option.short}</span>
                        <span className="d-none d-lg-inline">{option.label}</span>
                      </button>
                    ))}
                  </div>
                  <div className="d-flex flex-wrap gap-2 align-items-center">
                    {value ? null : (
                      <span className="sis-ungraded small text-nowrap">{DASH} not yet marked</span>
                    )}
                    {dirty ? <Badge tone="info">{t('unsaved')}</Badge> : null}
                  </div>
                </div>
              );
            }
          },
          {
            key: 'note',
            header: t('Reason'),
            cell: (row) => {
              const value = shown(row);
              /* Only for an excused absence, which is the one state the service requires a reason
                 for: without it, it cannot be told apart from an ordinary absence marked by
                 mistake. */
              if (value !== 'excused') {
                return shownNote(row) ? (
                  <span className="small text-body-tertiary">{shownNote(row)}</span>
                ) : null;
              }
              return (
                <Input
                  className="form-control-sm"
                  value={shownNote(row)}
                  placeholder={t('Required — e.g. medical appointment')}
                  onInput={(text) => note(row.student_number, text)}
                />
              );
            }
          }
        ]}
      />
    </Card>
  );
}
