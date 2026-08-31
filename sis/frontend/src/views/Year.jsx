/*
 * Year — one academic year: its terms, its subjects, and the ladder generator.
 *
 * Subjects live here rather than on a catalogue screen of their own because a subject now
 * belongs to a year: there is no school-wide list to put on a screen. The consequence is worth
 * restating where a registrar will read it, and the note above the subject table says it in one
 * line — a school that expects `MATH` to carry across years will otherwise discover the truth
 * from a report card.
 *
 * The classes of a year are on the rung screens, reached from the school. This screen is what a
 * year has that a rung does not: terms, a catalogue, and the generator that builds the ladder.
 */
import { useState } from 'react';
import { api } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import { dateText, labelOf, pickName, useAction, useForm, useResource, useStore } from '../hooks.js';
import { t } from '../i18n.js';
import {
  Alert,
  Badge,
  Breadcrumbs,
  Button,
  Card,
  Chip,
  Empty,
  ErrorNote,
  Field,
  Input,
  NoYearNotice,
  PageHead,
  Select,
  Skeleton,
  Table,
  Tile,
  useConfirm
} from '../components/Ui.jsx';
import { byStage } from '../structure.js';

/* -- Add one term ---------------------------------------------------------------- */

function TermForm({ year, count, onSaved }) {
  const form = useForm({
    code: '',
    name_en: '',
    name_ar: '',
    starts_on: '',
    ends_on: '',
    sequence: String((count || 0) + 1),
    is_closed: false
  });

  const save = useAction(() =>
    api.createTerm({
      code: form.values.code.trim(),
      academic_year_code: year,
      name_en: form.values.name_en.trim(),
      name_ar: form.values.name_ar.trim(),
      /* Blank means "not stated", sent as null. See `TermPanel` — an empty string
         would be a 422 about a date format, which is a confusing way to be told that
         leaving an optional field empty is allowed. */
      starts_on: form.values.starts_on || null,
      ends_on: form.values.ends_on || null,
      sequence: Number(form.values.sequence) || 1,
      is_closed: !!form.values.is_closed
    })
  );

  return (
    <form
      className="vstack gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        save
          .run()
          .then((term) => {
            Store.invalidate('terms:');
            Store.toast('ok', t('Term {0} saved', [term.code]));
            form.reset();
            if (onSaved) onSaved(term);
          })
          .catch(() => {});
      }}
    >
      <div className="row g-3">
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('Code')}
          required
          error={form.errorFor(save.error, 'code')}
        >
          <Input
            className="sis-code"
            value={form.values.code}
            placeholder="2026-T1"
            onInput={form.set('code')}
          />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-4" label={t('Name (English)')}>
          <Input value={form.values.name_en} onInput={form.set('name_en')} />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-4" label={t('Name (Arabic)')}>
          <Input className="sis-name-ar" value={form.values.name_ar} onInput={form.set('name_ar')} />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('First day')}
          hint={t('Optional')}
          error={form.errorFor(save.error, 'starts_on')}
        >
          <Input type="date" value={form.values.starts_on} onInput={form.set('starts_on')} />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('Last day')}
          hint={t('Optional')}
          error={form.errorFor(save.error, 'ends_on')}
        >
          <Input type="date" value={form.values.ends_on} onInput={form.set('ends_on')} />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('Sequence')}
          hint={t('Terms sort by this, never by code.')}
        >
          <Input
            type="number"
            inputMode="numeric"
            value={form.values.sequence}
            onInput={form.set('sequence')}
          />
        </Field>
      </div>

      <div className="form-check">
        <input
          className="form-check-input"
          type="checkbox"
          id="term-closed"
          checked={!!form.values.is_closed}
          onChange={(event) => form.set('is_closed')(event.target.checked)}
        />
        <label className="form-check-label small" htmlFor="term-closed">
          {t("Marks for this term are final. Stated by a person, never derived from the end date — a school enters last term's marks in the first week of this one.")}
        </label>
      </div>

      <ErrorNote error={save.error} />
      <div className="d-grid d-sm-block">
        <Button type="submit" variant="primary" pending={save.pending} pendingLabel={t('Saving…')}>
          {t('Add term')}
        </Button>
      </div>
    </form>
  );
}

/* -- Add one subject, to this year ------------------------------------------------ */

function SubjectForm({ year, count, onSaved }) {
  const form = useForm({
    code: '',
    name_en: '',
    name_ar: '',
    display_order: String((count || 0) + 1)
  });

  const save = useAction(() =>
    api.createSubject({
      code: form.values.code.trim(),
      academic_year_code: year,
      name_en: form.values.name_en.trim(),
      name_ar: form.values.name_ar.trim(),
      display_order: Number(form.values.display_order) || 0,
      is_active: true
    })
  );

  return (
    <form
      className="vstack gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        save
          .run()
          .then((subject) => {
            Store.invalidate('subjects:');
            Store.toast('ok', t('Subject {0} added to {1}', [subject.code, year]));
            form.reset();
            if (onSaved) onSaved(subject);
          })
          .catch(() => {});
      }}
    >
      <div className="row g-3">
        <Field
          className="col-12 col-sm-6 col-lg-3"
          label={t('Code')}
          required
          hint={`Identifies the subject within ${year}.`}
          error={form.errorFor(save.error, 'code')}
        >
          <Input
            className="sis-code"
            value={form.values.code}
            placeholder="MATH"
            onInput={form.set('code')}
          />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-3" label={t('Name (English)')}>
          <Input value={form.values.name_en} onInput={form.set('name_en')} />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-3" label={t('Name (Arabic)')}>
          <Input className="sis-name-ar" value={form.values.name_ar} onInput={form.set('name_ar')} />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-3" label={t('Report-card order')}>
          <Input
            type="number"
            inputMode="numeric"
            value={form.values.display_order}
            onInput={form.set('display_order')}
          />
        </Field>
      </div>
      <ErrorNote error={save.error} />
      <div className="d-grid d-sm-block">
        <Button type="submit" variant="primary" pending={save.pending} pendingLabel={t('Saving…')}>
          Add subject to {year}
        </Button>
      </div>
    </form>
  );
}

/* -- Retire or restore one subject ------------------------------------------------
 *
 * Confirmed, because it is structural: retiring a subject takes it out of every picker in the
 * year and changes what a marks upload will accept. It touches no mark already stated against
 * it, and the dialog says so — a registrar who fears losing marks will otherwise leave a dead
 * subject in the list forever.
 */
function RetireSubject({ year, subject }) {
  const [dialog, ask] = useConfirm();

  const save = (active) =>
    api
      .createSubject({
        code: subject.code,
        academic_year_code: year,
        name_en: subject.name_en,
        name_ar: subject.name_ar,
        display_order: subject.display_order,
        is_active: active
      })
      .then(() => {
        Store.invalidate('subjects:');
        Store.toast(
          'ok',
          active ? 'Subject restored' : 'Subject retired',
          `${subject.code} in ${year}`
        );
      });

  return (
    <>
      {dialog}
      <Button
        size="sm"
        variant="quiet"
        onClick={() =>
          subject.is_active
            ? ask({
                title: `Retire ${subject.code}?`,
                tone: 'bad',
                confirmLabel: 'Retire subject',
                changes: [{ label: 'State', was: 'active', now: 'retired' }],
                body: (
                  <>
                    <p>
                      {t('It leaves the pickers for')} <span className="sis-code">{year}</span> {t('and a marks upload naming it will be refused.')}
                    </p>
                    <p className="small text-body-tertiary mb-0">
                      {t("Every mark already stated against it stays exactly as it is — retiring is not deleting, and this year's report cards keep their heading.")}
                    </p>
                  </>
                ),
                run: () => save(false)
              })
            : ask({
                title: `Restore ${subject.code}?`,
                confirmLabel: 'Restore subject',
                changes: [{ label: 'State', was: 'retired', now: 'active' }],
                body: (
                  <p className="mb-0">
                    {t('It returns to the pickers for')} <span className="sis-code">{year}</span> {t('and marks uploads will accept it again.')}
                  </p>
                ),
                run: () => save(true)
              })
        }
      >
        {subject.is_active ? 'Retire' : 'Restore'}
      </Button>
    </>
  );
}

/* -- Which grades teach which subjects -------------------------------------------- */

/*
 * The board. A subject exists in a year; this is where a school says *where* it is taught.
 *
 * Three things about it are decisions rather than styling:
 *
 * **It is not only drag-and-drop.** Dragging is the fastest way to do this with a mouse and
 * it is the interaction that was asked for, so it is here. It is also unusable on a phone,
 * which is where most of this console is read, and unreachable from a keyboard. So every
 * drag has a tap twin: pick a subject up with a tap, and each rung offers a button to place
 * it. The two paths call exactly the same function, which is what stops the touch path from
 * quietly becoming the untested one.
 *
 * **One track at a time.** A bilingual school's Arabic and Languages sections keep separate
 * catalogues, and showing both ladders at once invites the mistake this stage exists to
 * prevent: dropping Physics on a rung that looks right and belongs to the other section.
 * Each rung already belongs to exactly one track, so the tabs filter rather than decide.
 *
 * **A drop that changes nothing is not an error.** The service is idempotent in both
 * directions, so a second drop on the same rung is a no-op there. The board still refuses
 * to send it — a request whose answer is known is a request worth not making — and marks
 * the chip as already placed instead, so the reader can see why nothing happened.
 */

/* The palette's own identity as a drop target. A space, because a code cannot contain one
   (see `_CODE_PATTERN` in the domain), so this can never collide with a rung's code. */
const PALETTE = ' palette';

/** A drop target's classes, with the mid-drag highlight added only while the pointer is on it. */
function dropClass(base, isOver) {
  return isOver ? `${base} is-over` : base;
}

/**
 * The drag payload: which subject, and which rung it was dragged off, if any.
 *
 * Space-separated for the same reason `PALETTE` is a space: codes are letters, digits, `.`,
 * `-` and `_` and nothing else, so a space is a separator neither half can contain.
 */
function dragPayload(subjectCode, fromLevel) {
  return `${subjectCode} ${fromLevel || ''}`;
}

function readPayload(event) {
  /* `text/plain` rather than a custom type: Firefox will not start a drag without a type it
     recognises, and a board that silently refuses to drag in one browser is worse than one
     that carries a slightly ugly payload. */
  const raw = event.dataTransfer.getData('text/plain') || '';
  const [subjectCode, fromLevel] = raw.split(' ');
  return subjectCode ? { subjectCode, fromLevel: fromLevel || '' } : null;
}

function SubjectBoard({ year, school, levels, subjects, lang }) {
  const tracks = useResource(Store.keys.tracks(school), () => api.schoolTracks(school), !!school);
  const board = useResource(
    Store.keys.subjectAssignments(year),
    () => api.subjectAssignments(year),
    !!year
  );

  /* The subject currently picked up by tap or keyboard. Empty means the reader is dragging,
     or doing nothing — both states in which no rung should be offering a place button. */
  const [held, setHeld] = useState('');
  /* The rung the pointer is over mid-drag, for the highlight. Purely visual; the drop reads
     its own target rather than this, so a missed `dragleave` cannot misfile a subject. */
  const [over, setOver] = useState('');

  const trackList = (tracks.value || []).slice();
  const [track, setTrack] = useState('');
  const shown = trackList.some((item) => item.code === track)
    ? track
    : (trackList[0] && trackList[0].code) || '';

  /* Assignments by rung code. A rung the service did not mention teaches nothing, which is
     the same thing to this screen as an empty list. */
  const assigned = {};
  (board.value || []).forEach((row) => {
    assigned[row.year_level_code] = row.subjects || [];
  });
  const has = (levelCode, subjectCode) =>
    (assigned[levelCode] || []).some((item) => item.code === subjectCode);

  const active = subjects.filter((item) => item.is_active);
  const nameOf = (item) => pickName(item, lang) || item.code;

  /*
   * One write path for every gesture: drag, tap and the remove button all end up here.
   *
   * `invalidate('subjects:')` rather than a targeted reload, because the assignment board
   * and the per-rung catalogues a marks screen reads are the same fact under two keys —
   * refreshing one and not the other is how a registrar assigns Physics and then finds it
   * missing from the sheet they open next.
   */
  const save = useAction((subjectCode, levelCode, assign) =>
    api
      .setSubjectAssignment({
        academic_year_code: year,
        subject_code: subjectCode,
        year_level_code: levelCode,
        assigned: assign
      })
      .then(() => Store.invalidate('subjects:'))
  );

  const place = (subjectCode, levelCode, fromLevel) => {
    setHeld('');
    setOver('');
    if (!subjectCode || !levelCode || has(levelCode, subjectCode)) return;
    save
      .run(subjectCode, levelCode, true)
      /* A move is two statements and this is the safe order: the subject is on both rungs
         for a moment rather than on neither, so a failure between them loses nothing. */
      .then(() => (fromLevel ? save.run(subjectCode, fromLevel, false) : null))
      .catch(() => {});
  };

  const remove = (subjectCode, levelCode) => {
    setHeld('');
    setOver('');
    save.run(subjectCode, levelCode, false).catch(() => {});
  };

  /* The selected track's rungs, plus the untracked ones — same rule as the school ladder,
     for the same reason: a rung nobody has placed in a section still teaches children, and a
     board that hides it is a board on which its subjects can never be set. */
  const visible = levels.filter(
    (level) => !shown || !level.track_code || level.track_code === shown
  );
  const grouped = byStage(visible);

  if (!levels.length) {
    return (
      <Empty title={t('No grades on this school yet')}>
        {t('A subject is assigned to a grade, so the ladder has to exist first. Add rungs on the school screen, or generate them below.')}
      </Empty>
    );
  }

  return (
    <div className="vstack gap-3">
      <Alert tone="info">
        {t('A subject appears only where it is assigned. Physics assigned to Secondary does not appear in Primary, and the two academic tracks are assigned separately.')}
      </Alert>

      {trackList.length > 1 ? (
        <div className="d-flex flex-wrap gap-2" role="group" aria-label={t('Academic track')}>
          {trackList.map((item) => (
            <Chip key={item.code} active={item.code === shown} onClick={() => setTrack(item.code)}>
              {nameOf(item)}
            </Chip>
          ))}
        </div>
      ) : null}

      {/* The palette. Also a drop target: dragging a chip back out of a rung is the gesture
          a reader tries first when they want it gone, so it has to mean something. */}
      <div
        className={dropClass('sis-board-palette', over === PALETTE)}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(PALETTE);
        }}
        onDragLeave={() => setOver('')}
        onDrop={(event) => {
          event.preventDefault();
          const payload = readPayload(event);
          setOver('');
          if (payload && payload.fromLevel) remove(payload.subjectCode, payload.fromLevel);
        }}
      >
        <div className="small text-body-secondary mb-2">
          {held
            ? t('Now choose a grade below, or tap the subject again to put it back.')
            : t('Drag a subject onto a grade, or tap it to pick it up.')}
        </div>
        <div className="d-flex flex-wrap gap-2" aria-label={t('Available subjects')}>
          {active.length ? (
            active.map((subject) => (
              <button
                key={subject.code}
                type="button"
                className="sis-chip"
                draggable
                aria-pressed={held === subject.code}
                onDragStart={(event) => {
                  event.dataTransfer.setData('text/plain', dragPayload(subject.code, ''));
                  setHeld('');
                }}
                onClick={() => setHeld(held === subject.code ? '' : subject.code)}
              >
                {nameOf(subject)}
              </button>
            ))
          ) : (
            <span className="small text-body-tertiary">
              {t('No active subject in this year to assign.')}
            </span>
          )}
        </div>
      </div>

      {grouped.length ? (
        grouped.map((group) => (
          <div key={group.stage.key} className="vstack gap-2">
            <h3 className="h6 text-body-secondary mb-0">{t(group.stage.label)}</h3>
            <div className="row g-3">
              {group.levels.map((level) => {
                const rows = assigned[level.code] || [];
                const already = held && has(level.code, held);
                return (
                  <div className="col-12 col-md-6 col-xl-4" key={level.code}>
                    <div
                      className={dropClass('sis-board-grade h-100', over === level.code)}
                      onDragOver={(event) => {
                        event.preventDefault();
                        setOver(level.code);
                      }}
                      onDragLeave={() => setOver('')}
                      onDrop={(event) => {
                        event.preventDefault();
                        const payload = readPayload(event);
                        if (payload) {
                          place(payload.subjectCode, level.code, payload.fromLevel);
                        } else {
                          setOver('');
                        }
                      }}
                    >
                      <div className="d-flex flex-wrap align-items-center gap-2 mb-2">
                        <span className="fw-semibold sis-pull">{nameOf(level)}</span>
                        <Badge>{level.code}</Badge>
                      </div>

                      {rows.length ? (
                        <div className="d-flex flex-wrap gap-2">
                          {rows.map((subject) => (
                            <span
                              key={subject.code}
                              className="sis-board-chip"
                              draggable
                              onDragStart={(event) =>
                                event.dataTransfer.setData(
                                  'text/plain',
                                  dragPayload(subject.code, level.code)
                                )
                              }
                            >
                              <span>{nameOf(subject)}</span>
                              {subject.is_active ? null : (
                                <span className="small">({t('retired')})</span>
                              )}
                              <button
                                type="button"
                                className="sis-board-remove"
                                aria-label={t('Remove {0} from {1}', [
                                  nameOf(subject),
                                  nameOf(level)
                                ])}
                                title={t('Remove assignment')}
                                onClick={() => remove(subject.code, level.code)}
                              >
                                &times;
                              </button>
                            </span>
                          ))}
                        </div>
                      ) : (
                        <span className="small text-body-tertiary">
                          {t('Teaches nothing yet.')}
                        </span>
                      )}

                      {/* The tap twin of the drop. Present only while something is held, so
                          the card is not a wall of buttons the rest of the time. */}
                      {held ? (
                        <div className="d-grid mt-3">
                          <Button
                            size="sm"
                            variant={already ? 'quiet' : 'primary'}
                            disabled={!!already}
                            onClick={() => place(held, level.code, '')}
                          >
                            {already ? t('Already here') : t('Assign here')}
                          </Button>
                        </div>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))
      ) : (
        <Empty title={t('No grades in this track')}>
          {t('Add a rung to this track on the school screen, and it will appear here.')}
        </Empty>
      )}

      <ErrorNote error={save.error} />
    </div>
  );
}

/* -- One term, on its own -------------------------------------------------------- */

/*
 * A term as a panel rather than a table row, and the reason is the shape of the data
 * rather than taste.
 *
 * A row is right for a list you scan and wrong for a thing you edit. Terms are now created
 * for the year — the school said how many it runs, so there are one, two or three of them
 * and never forty — and what a registrar comes here to do is fill in two dates on *one* of
 * them, some weeks after the year was set up. In a table that is a cell you cannot type
 * into, or an inline editor that has to fight the column widths. As a panel it is a small
 * form with the term's name on it, and the thing being edited cannot be mistaken for the
 * term above or below.
 *
 * Dates are optional here in the strong sense: the fields start empty, saving with both
 * empty is a legitimate save, and clearing one that was filled in is how a school says it
 * no longer knows. Nothing defaults them to the year's dates — that would be three terms
 * all claiming the whole year, which reads as a decision the school made.
 */
function TermPanel({ year, term, lang }) {
  const form = useForm({
    starts_on: term.starts_on || '',
    ends_on: term.ends_on || ''
  });

  /* Re-seeded when the term changes underneath the form — after a sync added a term, or
     another registrar dated this one. Keyed on the values themselves rather than on a
     render count, so typing is never interrupted by a refetch that changed nothing. */
  const [seed, setSeed] = useState(`${term.starts_on || ''}|${term.ends_on || ''}`);
  const current = `${term.starts_on || ''}|${term.ends_on || ''}`;
  if (seed !== current) {
    setSeed(current);
    form.reset({ starts_on: term.starts_on || '', ends_on: term.ends_on || '' });
  }

  const save = useAction(() =>
    api.createTerm({
      code: term.code,
      academic_year_code: year,
      name_en: term.name_en,
      name_ar: term.name_ar,
      /* Empty box means "not stated", so it is sent as null and clears the column. An
         empty string would be a 422 about a date format, which is a confusing way to be
         told that leaving a field blank is allowed. */
      starts_on: form.values.starts_on || null,
      ends_on: form.values.ends_on || null,
      sequence: term.sequence,
      is_closed: term.is_closed
    })
  );

  const dirty = current !== `${form.values.starts_on}|${form.values.ends_on}`;

  return (
    <Card
      className="h-100"
      title={
        <span className="d-inline-flex align-items-center gap-2">
          <Badge>{t('Term {0}', [term.sequence])}</Badge>
          <span>{pickName(term, lang) || term.code}</span>
        </span>
      }
      subtitle={term.code}
      actions={
        term.is_closed ? <Badge tone="warn">{t('closed')}</Badge> : <Badge tone="ok">{t('open')}</Badge>
      }
    >
      <form
        className="vstack gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          save
            .run()
            .then((saved) => {
              Store.invalidate('terms:');
              Store.toast('ok', t('Term {0} saved', [saved.code]));
            })
            .catch(() => {});
        }}
      >
        <div className="row g-3">
          <Field
            className="col-12 col-sm-6"
            label={t('First day')}
            hint={t('Optional')}
            error={form.errorFor(save.error, 'starts_on')}
          >
            <Input type="date" value={form.values.starts_on} onInput={form.set('starts_on')} />
          </Field>
          <Field
            className="col-12 col-sm-6"
            label={t('Last day')}
            hint={t('Optional')}
            error={form.errorFor(save.error, 'ends_on')}
          >
            <Input type="date" value={form.values.ends_on} onInput={form.set('ends_on')} />
          </Field>
        </div>

        {term.is_dated ? null : (
          <p className="small text-body-tertiary mb-0">
            {t('No dates yet. The term still holds marks and still closes — dates are only needed to say when it runs.')}
          </p>
        )}

        <div className="d-grid d-sm-flex gap-2">
          <Button type="submit" variant="primary" disabled={!dirty} pending={save.pending}>
            {t('Save dates')}
          </Button>
          <a className="btn btn-outline-secondary" href={Router.href('marks', { term: term.code })}>
            {t('Upload marks')}
          </a>
        </div>
        <ErrorNote error={save.error} />
      </form>
    </Card>
  );
}

/* -- What this year is attached to ----------------------------------------------- */

/*
 * The school, the tracks and the ladder, read in one request rather than four.
 *
 * Four requests can disagree: a rung added between the second and the third draws a school
 * that existed at no instant. It is also the answer to a question a registrar genuinely
 * asks on this screen — "is this year actually wired to anything yet" — which used to
 * require visiting three other screens to answer.
 */
function YearConnections({ year, lang }) {
  const detail = useResource(
    Store.keys.yearDetail(year),
    () => api.academicYear(year),
    !!year
  );
  /* Guarded on the *shape*, not on truthiness. The client answers a failed read with
     whatever it has, and a screen that tests `if (!body)` sails straight through an empty
     array or a half-built object and throws on the first property it reads — taking the
     whole page down rather than the one panel. */
  const body = detail.value;
  if (!body || !body.school) {
    return detail.error ? <ErrorNote error={detail.error} onRetry={detail.reload} /> : <Skeleton rows={2} />;
  }
  const tracks = body.tracks || [];
  const terms = body.terms || [];

  return (
    <div className="vstack gap-3">
      <div className="border rounded p-3">
        <div className="sis-tile-label">{t('School')}</div>
        <div className="d-flex flex-wrap align-items-center gap-2">
          <strong>{pickName(body.school, lang) || body.school.code}</strong>
          <Badge>{body.school.code}</Badge>
        </div>
      </div>

      <div className="row row-cols-2 row-cols-lg-4 g-3">
        <Tile
          label={t('Terms')}
          value={terms.length}
          note={t('{0} selected by the school', [body.school.term_count])}
        />
        <Tile label={t('Grades')} value={tracks.reduce((total, track) => total + (track.year_levels || []).length, 0)} />
        <Tile label={t('Classes')} value={body.class_count} />
        <Tile label={t('Tracks')} value={tracks.filter((track) => track.track_code).length} />
      </div>

      <div className="row g-3">
        {tracks.map((track) => (
          <div className="col-12 col-md-6" key={track.track_code || 'none'}>
            <div className="border rounded p-3 h-100">
              <div className="d-flex flex-wrap align-items-center gap-2 mb-2">
                <strong>{pickName(track, lang) || t('Not yet in a track')}</strong>
                {track.track_code ? <Badge>{track.track_code}</Badge> : null}
              </div>
              <p className="small text-body-secondary mb-2">
                {t('{0} grade(s), {1} class(es)', [(track.year_levels || []).length, track.class_count])}
              </p>
              <div className="d-flex flex-wrap gap-2">
                {(track.year_levels || []).length ? (
                  track.year_levels.map((level) => (
                    <Badge key={level.code}>{pickName(level, lang) || level.code}</Badge>
                  ))
                ) : (
                  <span className="small text-body-tertiary">{t('No grades on this track yet.')}</span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* -- The ladder generator -------------------------------------------------------- */

function Generator({ year }) {
  const form = useForm({ year_count: '6', classes_per_year: '2', suffixes: 'A,B,C,D' });
  const [result, setResult] = useState(null);

  const generate = useAction(() => {
    const suffixes = form.values.suffixes
      .split(',')
      .map((item) => item.trim())
      .filter(Boolean);
    return api.generateStructure({
      academic_year_code: year,
      year_count: Number(form.values.year_count),
      classes_per_year: Number(form.values.classes_per_year),
      class_suffixes: suffixes.length ? suffixes : null
    });
  });

  const levels = Number(form.values.year_count) || 0;
  const perLevel = Number(form.values.classes_per_year) || 0;

  return (
    <div className="vstack gap-3">
      <form
        className="vstack gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          generate
            .run()
            .then((outcome) => {
              setResult(outcome);
              Store.invalidate('years:');
              Store.invalidate('levels:');
              Store.invalidate('classes:');
              Store.toast(
                'ok',
                `${outcome.created_count} item(s) created`,
                `${outcome.existing_count} already existed and were left untouched`
              );
            })
            .catch(() => {});
        }}
      >
        <div className="row g-3">
          <Field
            className="col-12 col-sm-4"
            label={t('Year levels')}
            hint={t('Y1 … Yn, the rungs of the school.')}
          >
            <Input
              type="number"
              inputMode="numeric"
              value={form.values.year_count}
              onInput={form.set('year_count')}
            />
          </Field>
          <Field className="col-12 col-sm-4" label={t('Sections per level')}>
            <Input
              type="number"
              inputMode="numeric"
              value={form.values.classes_per_year}
              onInput={form.set('classes_per_year')}
            />
          </Field>
          <Field
            className="col-12 col-sm-4"
            label={t('Section suffixes')}
            hint={t('Comma separated; the first n are used.')}
          >
            <Input
              className="sis-code"
              value={form.values.suffixes}
              onInput={form.set('suffixes')}
            />
          </Field>
        </div>

        <Alert tone="info">
          {t('Builds')} <strong>{levels}</strong> year level(s) and up to{' '}
          <strong>{levels * perLevel}</strong> {t('section(s) in')} <span className="sis-code">{year}</span>
          . Safe to run twice — anything already there is reported and left exactly as it is.
        </Alert>

        <ErrorNote error={generate.error} />
        <div className="d-grid d-sm-block">
          <Button
            type="submit"
            variant="primary"
            pending={generate.pending}
            pendingLabel={t('Generating…')}
          >
            {t('Generate')}
          </Button>
        </div>
      </form>

      {result ? (
        <div className="vstack gap-2 sis-rise">
          <div className="d-flex flex-wrap gap-2">
            <Badge tone="ok">{result.created_count} created</Badge>
            <Badge>{result.existing_count} already there</Badge>
          </div>
          <Table
            rows={result.items || []}
            rowKey={(row) => `${row.kind}:${row.code}`}
            rowTone={(row) => (row.created ? 'ok' : null)}
            columns={[
              {
                key: 'kind',
                header: t('Kind'),
                hide: 'sm',
                cell: (row) => (row.kind === 'class_section' ? 'Section' : 'Year level')
              },
              { key: 'code', header: 'Code', className: 'sis-code', cell: (row) => row.code },
              {
                key: 'name',
                header: t('Name'),
                className: 'sis-name-en',
                hide: 'md',
                cell: (row) => row.name_en
              },
              {
                key: 'state',
                header: t('Outcome'),
                cell: (row) =>
                  row.created ? <Badge tone="ok">{t('created')}</Badge> : <Badge>{t('already there')}</Badge>
              }
            ]}
          />
        </div>
      ) : null}
    </div>
  );
}

/* -- A card whose form opens and closes ------------------------------------------ */

function Section({ title, subtitle, action, form, children }) {
  const [open, setOpen] = useState(false);
  return (
    <Card
      title={title}
      subtitle={subtitle}
      tight
      actions={
        <Button size="sm" variant={open ? 'quiet' : 'primary'} onClick={() => setOpen(!open)}>
          {open ? 'Close' : action}
        </Button>
      }
    >
      {open ? <div className="card-body border-bottom sis-rise">{form}</div> : null}
      {children}
    </Card>
  );
}

/* -- Screen ---------------------------------------------------------------------- */

export function Year({ params = {} }) {
  const state = useStore();
  const code = params.code || state.year;

  const years = useResource(Store.keys.years(state.school), () => api.years(state.school), !!state.school);
  const terms = useResource(Store.keys.terms(code), () => api.terms(code), !!code);
  /* Inactive subjects included: this is the screen where a retired subject has to be visible,
     or the only sign of it is that its code cannot be reused. */
  const subjects = useResource(
    Store.keys.subjects(code, true),
    () => api.subjects(code, true),
    !!code
  );

  if (!code) {
    return (
      <>
        <PageHead title={t('Year')} />
        <NoYearNotice />
      </>
    );
  }

  const yearList = (years.value && years.value.academic_years) || [];
  const year = yearList.find((item) => item.code === code);
  const nameClass = state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en';

  const termList = (terms.value || []).slice().sort((a, b) => (a.sequence || 0) - (b.sequence || 0));
  const subjectList = (subjects.value || [])
    .slice()
    .sort((a, b) => (a.display_order || 0) - (b.display_order || 0));

  return (
    <>
      <Breadcrumbs
        trail={[
          { label: 'Schools', to: 'school' },
          state.school
            ? { label: state.school, to: 'school', params: { code: state.school } }
            : null,
          { label: code }
        ]}
      />
      <PageHead
        title={code}
        lede={
          year
            ? `${pickName(year, state.lang)} — ${dateText(year.starts_on)} to ${dateText(year.ends_on)}`
            : t('This year is not on file.')
        }
        actions={
          <Button icon="refresh" onClick={() => Store.invalidate('')}>
            {t('Refresh')}
          </Button>
        }
      />

      {years.ready && !year ? (
        <Alert tone="warn" title={t('No such academic year')}>
          {t('Nothing is on file for')} <span className="sis-code">{code}</span>.{' '}
          <a href={Router.href('school')}>{t('Go back to the school')}</a> {t('and pick one that exists.')}
        </Alert>
      ) : null}

      <div className="vstack gap-3">
        <Card
          title={t('This year is attached to')}
          subtitle={t('School, tracks, grades and classes — read together')}
        >
          <YearConnections year={code} lang={state.lang} />
        </Card>

        <Section
          title={t('Terms — {0}', [code])}
          subtitle={
            termList.length
              ? t('{0} term section(s), one panel each', [termList.length])
              : t('none yet')
          }
          action="Add a term"
          form={<TermForm year={code} count={termList.length} />}
        >
          <div className="card-body">
            {termList.length ? (
              <div className="vstack gap-3">
                <Alert tone="info">
                  {t('These sections come from the number of terms the school runs. Dates are optional — a term works without them, and they can be filled in whenever the calendar is settled.')}
                </Alert>
                {/* One column per term, so two terms sit side by side on a laptop and
                    three still fit. Each panel is its own card: the requirement is that a
                    registrar can never be unsure which term they are editing. */}
                <div className="row g-3">
                  {termList.map((term) => (
                    <div
                      className={
                        termList.length > 2 ? 'col-12 col-md-6 col-xl-4' : 'col-12 col-md-6'
                      }
                      key={term.code}
                    >
                      <TermPanel year={code} term={term} lang={state.lang} />
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <Empty title={t('No terms yet')}>
                {t('Terms are created with the year, from the number the school runs. If none are here, this year predates that — add one, or re-save the year from the school screen.')}
              </Empty>
            )}
          </div>
        </Section>

        <Section
          title={t('Subjects taught in {0}', [code])}
          subtitle={t('{0} in this year', [subjectList.length])}
          action="Add subject"
          form={<SubjectForm year={code} count={subjectList.length} />}
        >
          <div className="card-body pb-0">
            <Alert tone="info">
              {t('A subject belongs to this year. The same code in another year is a different subject, so marks are not comparable across years — copy the catalogue forward each September rather than expecting it to carry over.')}
            </Alert>
          </div>
          <Table
            loading={subjects.loading}
            rows={subjectList}
            rowKey={(row) => row.code}
            empty={
              <Empty title={t('No subjects in this year')}>
                {t("A marks upload names a subject from this year's catalogue, so nothing can be recorded until one exists.")}
              </Empty>
            }
            columns={[
              {
                key: 'order',
                header: t('#'),
                className: 'sis-num',
                hide: 'sm',
                cell: (row) => row.display_order
              },
              { key: 'code', header: 'Code', className: 'sis-code', cell: (row) => row.code },
              {
                key: 'en',
                header: t('English'),
                className: 'sis-name-en',
                hide: 'md',
                cell: (row) => row.name_en || <span className="text-body-tertiary">—</span>
              },
              {
                key: 'ar',
                header: t('Arabic'),
                className: 'sis-name-ar',
                hide: 'lg',
                cell: (row) => row.name_ar || <span className="text-body-tertiary">—</span>
              },
              {
                key: 'state',
                header: t('State'),
                cell: (row) =>
                  row.is_active ? <Badge tone="ok">{t('active')}</Badge> : <Badge tone="warn">{t('retired')}</Badge>
              },
              {
                key: 'retire',
                header: '',
                cell: (row) => <RetireSubject year={code} subject={row} />
              }
            ]}
          />
        </Section>

        <Card
          title={t('Which grades teach what')}
          subtitle={t('A subject is taught only where it is placed')}
        >
          <SubjectBoard
            year={code}
            school={state.school}
            levels={(years.value && years.value.year_levels) || []}
            subjects={subjectList}
            lang={state.lang}
          />
        </Card>

        <Card title={t('Generate the ladder')}>
          <Generator year={code} />
        </Card>
      </div>
    </>
  );
}
