/*
 * Klass — one class, and everything a school does to it.
 *
 * Named `Klass` because `class` is a reserved word, which is a small tax for putting the screen
 * where the work is. This is the deepest screen a registrar reaches by clicking rather than
 * searching, and it is the one that earns the hierarchy: the register, the morning's attendance,
 * this class's marks upload, and the edits that change who is in the room — all against a class
 * already chosen, so none of them asks for a class code again.
 *
 * Three things here are worth reading before changing them.
 *
 * **Adding a child is two facts, not one.** A child is a person, and her place in a class is a
 * dated membership of it. So "add a child to this class" is `POST /students` followed by
 * `POST /placements`, and the form says so. Collapsing them into one request would make the
 * screen unable to express the case a school hits in week one — a child who exists but has not
 * been placed yet — and there is a second form for exactly that.
 *
 * **Removing a child ends a placement; it does not delete her.** The button says "Remove from
 * class" and the confirmation says what survives: her record, her marks, her attendance. A
 * school that means to remove a child from the school does it from her own screen.
 *
 * **Moving a child is one request.** `transfer` closes the open placement and opens the next in
 * the same transaction, because between two separate calls she is in no class at all, and a
 * marks upload landing in that window rejects every one of her rows.
 *
 * The tabs are pills that scroll on a phone rather than wrapping to three rows, and the action
 * column is small buttons that wrap — a four-button `btn-group` at 360px is four buttons nobody
 * can hit.
 */
import { useEffect, useRef, useState } from 'react';
import { api } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import {
  DASH,
  dateText,
  labelOf,
  pickName,
  today,
  useAction,
  useForm,
  useQuery,
  useResource,
  useStore
} from '../hooks.js';
import {
  Alert,
  Badge,
  Breadcrumbs,
  Button,
  Card,
  Empty,
  ErrorNote,
  Field,
  Input,
  NoYearNotice,
  PageHead,
  Select,
  Table,
  cx,
  useConfirm
} from '../components/Ui.jsx';
import { AttendancePanel } from '../components/AttendancePanel.jsx';
import { StudentEditorFor } from '../components/StudentEditor.jsx';
import { ImportFlow } from '../components/ImportFlow.jsx';
import { t } from '../i18n.js';

const TABS = [
  { key: 'register', label: 'Register' },
  { key: 'attendance', label: 'Attendance' },
  { key: 'marks', label: 'Marks' }
];

const MARKS_TEMPLATE = {
  name: 'marks-template.csv',
  header: t('student_number,subject_code,percentage')
};

/* -- Add a child, and place her here, in one submit -------------------------------- */

function AddChild({ classCode, year, onSaved }) {
  const form = useForm({
    student_number: '',
    full_name_en: '',
    full_name_ar: '',
    date_of_birth: '',
    contact_phone: '',
    starts_on: today()
  });

  /* Two requests, in order, and the second is the one that puts her in the room. If the
     placement fails the child still exists — which is why the toast names both steps: a
     registrar who reads "saved" and nothing else would not know to go looking for her. */
  const save = useAction(async () => {
    const student = await api.saveStudent({
      student_number: form.values.student_number.trim(),
      full_name_en: form.values.full_name_en.trim(),
      full_name_ar: form.values.full_name_ar.trim(),
      date_of_birth: form.values.date_of_birth || null,
      contact_phone: form.values.contact_phone.trim()
    });
    await api.placeStudent(student.student_number, {
      academic_year_code: year,
      class_code: classCode,
      starts_on: form.values.starts_on
    });
    return student;
  });

  return (
    <form
      className="vstack gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        save
          .run()
          .then((student) => {
            Store.invalidate('roster:');
            Store.toast(
              'ok',
              `${student.student_number} added`,
              `Record created, and placed in ${classCode} from ${form.values.starts_on}.`
            );
            form.reset();
            if (onSaved) onSaved(student);
          })
          .catch(() => {});
      }}
    >
      <div className="row g-3">
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('Student number')}
          required
          hint={t("The school's own identifier. It never changes.")}
          error={form.errorFor(save.error, 'student_number')}
        >
          <Input
            className="sis-code"
            value={form.values.student_number}
            onInput={form.set('student_number')}
          />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('Name (English)')}
          error={form.errorFor(save.error, 'full_name_en')}
        >
          <Input value={form.values.full_name_en} onInput={form.set('full_name_en')} />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-4" label={t('Name (Arabic)')}>
          <Input
            className="sis-name-ar"
            value={form.values.full_name_ar}
            onInput={form.set('full_name_ar')}
          />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('Date of birth')}
          hint={t('Optional. Her age is read from this, never stored beside it.')}
        >
          <Input
            type="date"
            value={form.values.date_of_birth}
            onInput={form.set('date_of_birth')}
          />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-4" label={t('Contact phone')}>
          <Input
            type="tel"
            inputMode="tel"
            value={form.values.contact_phone}
            onInput={form.set('contact_phone')}
          />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('In the class from')}
          required
          hint={t('The day her membership starts, not the day you typed it.')}
          error={form.errorFor(save.error, 'starts_on')}
        >
          <Input type="date" value={form.values.starts_on} onInput={form.set('starts_on')} />
        </Field>
      </div>
      <ErrorNote error={save.error} />
      <div className="d-grid d-sm-block">
        <Button type="submit" variant="primary" pending={save.pending} pendingLabel={t('Saving…')}>
          Add to {classCode}
        </Button>
      </div>
    </form>
  );
}

/* -- Place a child who already has a record --------------------------------------- */

function PlaceExisting({ classCode, year, onSaved }) {
  const form = useForm({ student_number: '', starts_on: today() });
  const save = useAction(() =>
    api.placeStudent(form.values.student_number.trim(), {
      academic_year_code: year,
      class_code: classCode,
      starts_on: form.values.starts_on
    })
  );

  return (
    <form
      className="vstack gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        save
          .run()
          .then(() => {
            Store.invalidate('roster:');
            Store.toast('ok', t('{0} placed in {1}', [form.values.student_number.trim(), classCode]));
            form.reset();
            if (onSaved) onSaved();
          })
          .catch(() => {});
      }}
    >
      <div className="row g-3">
        <Field
          className="col-12 col-sm-6"
          label={t('Student number')}
          required
          hint={t('A child whose record exists already — a transfer in, or one uploaded but never placed.')}
          error={form.errorFor(save.error, 'student_number')}
        >
          <Input
            className="sis-code"
            value={form.values.student_number}
            onInput={form.set('student_number')}
          />
        </Field>
        <Field className="col-12 col-sm-6" label={t('In the class from')} required>
          <Input type="date" value={form.values.starts_on} onInput={form.set('starts_on')} />
        </Field>
      </div>
      <ErrorNote error={save.error} />
      <div className="d-grid d-sm-block">
        <Button type="submit" variant="primary" pending={save.pending} pendingLabel={t('Placing…')}>
          Place in {classCode}
        </Button>
      </div>
    </form>
  );
}

/* -- Move a child to another class in the same year -------------------------------- */

function MoveChild({ student, classCode, year, yearLevel, onDone }) {
  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);
  const form = useForm({ to_class_code: '', on_date: today() });
  const [dialog, ask] = useConfirm();

  /* Same grade only: a move is a change of section, not of year — a child put into a class one
     grade up by a mistyped dropdown is a mistake nobody notices until her marks arrive against
     the wrong subjects. The level filter is skipped only while the classes are still loading,
     because filtering against an undefined level would empty the list and read as "no class to
     move her to". */
  const options = (classes.value || [])
    .filter(
      (section) =>
        section.code !== classCode &&
        (!yearLevel || section.year_level_code === yearLevel)
    )
    .map((section) => ({ value: section.code, label: `${section.code} — ${labelOf(section)}` }));

  return (
    <>
      {dialog}
      <form
        className="vstack gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          ask({
            title: `Move ${student.student_number}?`,
            confirmLabel: 'Move her',
            changes: [
              { label: 'class', was: classCode, now: form.values.to_class_code },
              { label: 'from', was: null, now: form.values.on_date }
            ],
            body: (
              <p className="mb-0">
                One request: her membership of {classCode} closes and the new one opens on{' '}
                {form.values.on_date}, so she is never in no class at all. Nothing already
                recorded against {classCode} changes — a mark stated there stays stated there.
              </p>
            ),
            run: () =>
              api
                .transferStudent(student.student_number, {
                  academic_year_code: year,
                  to_class_code: form.values.to_class_code,
                  on_date: form.values.on_date
                })
                .then(() => {
                  Store.invalidate('roster:');
                  Store.invalidate('placements:');
                  Store.toast(
                    'ok',
                    `${student.student_number} moved`,
                    `${classCode} → ${form.values.to_class_code}`
                  );
                  if (onDone) onDone();
                })
          });
        }}
      >
        <div className="row g-3">
          <Field className="col-12 col-sm-6" label={t('To class')} required>
            <Select
              value={form.values.to_class_code}
              options={options}
              placeholder={classes.loading ? t('Loading…') : t('Choose a class')}
              onChange={form.set('to_class_code')}
            />
          </Field>
          <Field className="col-12 col-sm-6" label={t('From')} required>
            <Input type="date" value={form.values.on_date} onInput={form.set('on_date')} />
          </Field>
        </div>
        <ErrorNote error={classes.error} onRetry={classes.reload} />
        {!classes.loading && !options.length ? (
          <Alert tone="warn" title={t('No other class in this grade')}>
            {t('She can only be moved to another section of the same grade, and this grade has no other. Add one first.')}
          </Alert>
        ) : null}
        <div className="d-grid gap-2 d-sm-flex">
          <Button type="submit" variant="primary" disabled={!form.values.to_class_code}>
            Move out of {classCode}
          </Button>
          <Button variant="quiet" onClick={onDone}>
            {t('Cancel')}
          </Button>
        </div>
      </form>
    </>
  );
}

/* -- Rename, which is a label change and never a code change ---------------------- */

function RenameClass({ section, year, onDone }) {
  const form = useForm({
    name_en: section.name_en || '',
    name_ar: section.name_ar || ''
  });
  const [dialog, ask] = useConfirm();

  const changed = form.changed();
  const diff = Object.keys(changed).map((key) => ({
    label: key.replace(/_/g, ' '),
    was: section[key],
    now: changed[key]
  }));

  return (
    <>
      {dialog}
      <form
        className="vstack gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (!diff.length) {
            Store.toast('info', t('Nothing changed'));
            if (onDone) onDone();
            return;
          }
          ask({
            title: `Rename ${section.code}?`,
            confirmLabel: 'Save the new labels',
            changes: diff,
            body: (
              <p className="mb-0">
                {t('The code')} <strong>{section.code}</strong> {t('does not change, and cannot: every mark, placement and attendance row in the school points at it. This changes only what the class is called on screen and on a report card.')}
              </p>
            ),
            run: () =>
              api
                .renameClassSection(section.code, year, {
                  name_en: form.values.name_en.trim(),
                  name_ar: form.values.name_ar.trim()
                })
                .then(() => {
                  Store.invalidate('classes:');
                  Store.toast('ok', t('{0} renamed', [section.code]));
                  if (onDone) onDone();
                })
          });
        }}
      >
        <div className="row g-3">
          <Field className="col-12 col-sm-6" label={t('Name (English)')}>
            <Input value={form.values.name_en} onInput={form.set('name_en')} />
          </Field>
          <Field className="col-12 col-sm-6" label={t('Name (Arabic)')}>
            <Input
              className="sis-name-ar"
              value={form.values.name_ar}
              onInput={form.set('name_ar')}
            />
          </Field>
        </div>
        <p className="small text-body-tertiary mb-0">
          {t('Labels only. The service takes no capacity here and no rung — moving a class to another rung would carry every enrolment and every mark with it, under a class the children were never in, so it is a new class plus a roster change rather than an edit.')}
        </p>
        <div className="d-grid gap-2 d-sm-flex">
          <Button type="submit" variant="primary" disabled={!diff.length}>
            {diff.length ? `Save ${diff.length} change(s)` : 'Nothing to save'}
          </Button>
          <Button variant="quiet" onClick={onDone}>
            {t('Cancel')}
          </Button>
        </div>
      </form>
    </>
  );
}

/* -- The register ---------------------------------------------------------------- */

function Register({ classCode, year, yearLevel }) {
  const state = useStore();
  const [panel, setPanel] = useState(null); /* 'add' | 'place' | null */
  const [editing, setEditing] = useState('');
  const [moving, setMoving] = useState('');
  const [dialog, ask] = useConfirm();
  /* Edit and Move open a panel under the table. On a register of thirty that panel is below the
     fold, so the click reads as a button that did nothing — the reason to scroll to it is that
     the effect is otherwise invisible, not decoration. */
  const panel_ref = useRef(null);
  useEffect(() => {
    if ((editing || moving) && panel_ref.current) {
      panel_ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [editing, moving]);

  const roster = useQuery(
    () => api.classRoster(classCode, year),
    [classCode, year],
    !!(classCode && year)
  );
  const students = (roster.value && roster.value.students) || [];

  function removeFromClass(student) {
    ask({
      title: `Remove ${student.student_number} from ${classCode}?`,
      tone: 'bad',
      confirmLabel: 'Remove from the class',
      body: (
        <div className="vstack gap-2">
          <p className="mb-0">
            This ends her membership of {classCode} today. It does not delete her: her record, her
            marks and her attendance all stay exactly as they are, and she can be placed in
            another class tomorrow.
          </p>
          <p className="mb-0 small text-body-tertiary">
            {t('If she is moving to another class, use')} <strong>{t('Move')}</strong> {t('instead — it closes this placement and opens the next one together, so she is never in no class at all.')}
          </p>
        </div>
      ),
      run: () =>
        api.endPlacement(student.student_number, { ends_on: today() }).then(() => {
          Store.invalidate('roster:');
          Store.invalidate('placements:');
          Store.toast('ok', t('{0} removed from {1}', [student.student_number, classCode]));
          roster.reload();
        })
    });
  }

  return (
    <div className="vstack gap-4">
      {dialog}

      <Card
        title={t('On the register')}
        subtitle={
          roster.value
            ? `${roster.value.count} child(ren) as of ${dateText(roster.value.as_of)}`
            : null
        }
        actions={
          <div className="d-flex flex-wrap gap-2">
            <Button
              size="sm"
              variant={panel === 'add' ? 'primary' : 'outline'}
              onClick={() => setPanel(panel === 'add' ? null : 'add')}
            >
              {t('Add a child')}
            </Button>
            <Button
              size="sm"
              variant={panel === 'place' ? 'primary' : 'outline'}
              onClick={() => setPanel(panel === 'place' ? null : 'place')}
            >
              {t('Place an existing child')}
            </Button>
            <Button size="sm" icon="refresh" onClick={roster.reload}>
              {t('Reload')}
            </Button>
          </div>
        }
        tight
      >
        {panel === 'add' ? (
          <div className="card-body border-bottom sis-rise">
            <AddChild
              classCode={classCode}
              year={year}
              onSaved={() => {
                setPanel(null);
                roster.reload();
              }}
            />
          </div>
        ) : null}
        {panel === 'place' ? (
          <div className="card-body border-bottom sis-rise">
            <PlaceExisting
              classCode={classCode}
              year={year}
              onSaved={() => {
                setPanel(null);
                roster.reload();
              }}
            />
          </div>
        ) : null}

        <ErrorNote error={roster.error} onRetry={roster.reload} />

        <Table
          loading={roster.loading}
          rows={students}
          rowKey={(row) => row.student_number}
          rowHref={(row) => Router.href('student', { number: row.student_number })}
          rowLabel={(row) => t('Open {0}', [pickName(row, state.lang) || row.student_number]).join('')}
          empty={
            <Empty
              title={t('Nobody is in {0} yet', [classCode])}
              action={
                <Button variant="primary" onClick={() => setPanel('add')}>
                  {t('Add the first child')}
                </Button>
              }
            >
              {t('An empty class is a real state, not a missing one — a class exists from the day the ladder is generated and children arrive later. Add one here, or upload the whole roster from the Roster screen.')}
            </Empty>
          }
          columns={[
            {
              key: 'number',
              header: t('Student no.'),
              className: 'sis-code',
              hide: 'md',
              cell: (row) => (
                <a
                  className="sis-plain"
                  href={Router.href('student', { number: row.student_number })}
                >
                  {row.student_number}
                </a>
              )
            },
            {
              key: 'name',
              header: t('Child'),
              className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
              cell: (row) => (
                <>
                  <a
                    className="fw-semibold"
                    href={Router.href('student', { number: row.student_number })}
                  >
                    {pickName(row, state.lang) || (
                      <span className="sis-ungraded">{DASH} name not on file</span>
                    )}
                  </a>
                  {/* The number rides under the name on a phone, where its own column is gone. */}
                  <div className="sis-code sis-xs text-body-tertiary d-md-none">
                    {row.student_number}
                  </div>
                </>
              )
            },
            {
              key: 'since',
              header: t('In the class since'),
              className: 'sis-num',
              hide: 'lg',
              cell: (row) => (
                <span className="font-monospace small">{dateText(row.starts_on)}</span>
              )
            },
            {
              key: 'actions',
              header: '',
              cell: (row) => (
                <div className="sis-row-actions d-flex flex-wrap gap-1 justify-content-end">
                  <Button
                    size="sm"
                    onClick={() => {
                      setMoving('');
                      setEditing(editing === row.student_number ? '' : row.student_number);
                    }}
                  >
                    {t('Edit')}
                  </Button>
                  <Button
                    size="sm"
                    onClick={() => {
                      setEditing('');
                      setMoving(moving === row.student_number ? '' : row.student_number);
                    }}
                  >
                    {t('Move')}
                  </Button>
                  <Button size="sm" variant="danger" onClick={() => removeFromClass(row)}>
                    {t('Remove')}
                  </Button>
                </div>
              )
            }
          ]}
        />

        {students.length ? (
          <div className="card-footer small text-body-tertiary">
            {t("Double-click a row to open a child's record — or tap her name, which is the same place and works on a phone.")}
          </div>
        ) : null}
      </Card>

      {/* The edit and move forms open under the table rather than inside the row: a form in a
          table cell on a phone is a form in a 90px column. */}
      <div ref={panel_ref} />

      {editing ? (
        <Card className="sis-rise" title={t('Edit {0}', [editing])}>
          {/* The form loads her whole record rather than editing the three columns this table
              happens to carry: a diff whose "was" column is blank because the roster row never
              had a phone number is a diff that lies. */}
          <StudentEditorFor
            studentNumber={editing}
            academicYear={year}
            onDone={() => {
              setEditing('');
              roster.reload();
            }}
          />
        </Card>
      ) : null}

      {moving ? (
        <Card className="sis-rise" title={t('Move {0}', [moving])}>
          <MoveChild
            student={students.find((row) => row.student_number === moving) || {}}
            classCode={classCode}
            year={year}
            yearLevel={yearLevel}
            onDone={() => {
              setMoving('');
              roster.reload();
            }}
          />
        </Card>
      ) : null}
    </div>
  );
}

/* -- This class's marks upload ---------------------------------------------------- */

function ClassMarks({ classCode, year, yearLevel }) {
  const [term, setTerm] = useState('');
  const [subject, setSubject] = useState('');

  const terms = useResource(Store.keys.terms(year), () => api.terms(year), !!year);
  /* The rung's subjects, not the year's. The class is already fixed on this screen, so its
     rung is known — and offering a subject that rung does not teach is offering to file a
     mark against a lesson nobody in this room sat. Falls back to the whole catalogue only
     while the section is still loading and the rung is genuinely unknown. */
  const subjects = useResource(
    yearLevel ? Store.keys.gradeSubjects(year, yearLevel) : Store.keys.subjects(year),
    () => api.subjects(year, false, yearLevel),
    !!year
  );

  const termList = (terms.value || [])
    .slice()
    .sort((a, b) => (a.sequence || 0) - (b.sequence || 0));
  const open = termList.filter((item) => !item.is_closed);

  /* Default to the first term still open for marks. A school uploading in November means this
     term, and a screen defaulting to term one would file the marks a year late without saying
     anything. */
  useEffect(() => {
    if (!term && open.length) setTerm(open[0].code);
  }, [open.length]);

  const chosen = termList.find((item) => item.code === term);

  return (
    <div className="vstack gap-4">
      <Alert tone="info" title={t('Marks for {0}', [classCode])}>
        {t('The class is already fixed, so the file needs two columns:')} <code>{t('student_number')}</code> and{' '}
        <code>{t('percentage')}</code>. A row for a child who is not on this register is rejected rather than filed elsewhere — which is the point of uploading from here rather than from the Marks screen.
      </Alert>

      <ImportFlow
        kind="grades"
        template={MARKS_TEMPLATE}
        label={t('Choose the marks sheet')}
        hint={`Columns: ${MARKS_TEMPLATE.header}, or points and max_points instead of percentage.`}
        ready={!!term}
        blocker={
          <Alert tone="info">
            {t('Choose the term first. A marks file with no term named is a file the service cannot place.')}
          </Alert>
        }
        fields={
          <div className="row g-3">
            <Field className="col-12 col-sm-6" label={t('Term')} required>
              <Select
                value={term}
                options={termList.map((item) => ({
                  value: item.code,
                  label: `${item.code} — ${labelOf(item)}${item.is_closed ? ' (closed)' : ''}`
                }))}
                placeholder={terms.loading ? t('Loading…') : t('Choose a term')}
                onInput={setTerm}
              />
            </Field>
            <Field
              className="col-12 col-sm-6"
              label={t('Subject')}
              hint={t('Optional. Naming one lets the file leave the subject column out.')}
            >
              <Select
                value={subject}
                options={(subjects.value || []).map((item) => ({
                  value: item.code,
                  label: `${item.code} — ${labelOf(item)}`
                }))}
                placeholder={t('Every subject named in the file')}
                onInput={setSubject}
              />
            </Field>
          </div>
        }
        onPreview={(file) => {
          const form = new FormData();
          form.append('file', file);
          form.append('term_code', term);
          form.append('class_code', classCode);
          if (subject) form.append('subject_code', subject);
          return api.previewGrades(form);
        }}
        onCommit={(batchId) => api.commitGrades(batchId)}
        invalidate={['grades:']}
      />

      {chosen && chosen.is_closed ? (
        <Alert tone="warn" title={t('{0} is closed', [chosen.code])}>
          {t("A person marked this term final, so the service will refuse the upload. Reopen the term from the academic year screen if last term's marks are genuinely still arriving.")}
        </Alert>
      ) : null}
    </div>
  );
}

/* -- Screen ---------------------------------------------------------------------- */

export function Klass({ params = {} }) {
  const state = useStore();
  const classCode = params.code || '';
  const year = params.year || state.year;
  const school = params.school || state.school;

  /* The open tab lives in the URL, not in state, and that is a fix rather than a preference.
     App keys the view by route *name*, so arriving at `#/class?code=3A&tab=attendance` while
     already on a class screen does not remount this component — a `useState` initial value
     would keep showing the register, and the Attendance link on the rung screen would appear
     to do nothing. Reading the parameter every render also makes the tab survive a refresh
     and travel in a shared link. */
  const tab = TABS.some((item) => item.key === params.tab) ? params.tab : 'register';
  const [renaming, setRenaming] = useState(false);

  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);
  const section = (classes.value || []).find((item) => item.code === classCode);

  if (!classCode) {
    return (
      <>
        <PageHead title={t('Class')} />
        <Alert tone="warn" title={t('No class chosen')}>
          {t('Open one from a rung.')} <a href={Router.href('school')}>{t('Go to the school')}</a>.
        </Alert>
      </>
    );
  }

  const trail = [
    { label: 'Schools', to: 'school' },
    school ? { label: school, to: 'school', params: { code: school } } : null,
    section && section.year_level_code
      ? {
          label: section.year_level_code,
          to: 'level',
          params: { code: section.year_level_code, year }
        }
      : null,
    { label: classCode }
  ].filter(Boolean);

  if (!year) {
    return (
      <>
        <Breadcrumbs trail={trail} />
        <PageHead title={classCode} />
        <NoYearNotice />
      </>
    );
  }

  return (
    <>
      <Breadcrumbs trail={trail} />
      <PageHead
        title={section ? `${classCode} — ${labelOf(section, state.lang)}` : classCode}
        lede={t('Everything this class does, in {0}.', [year])}
        actions={
          <Button variant={renaming ? 'primary' : 'outline'} onClick={() => setRenaming(!renaming)}>
            {renaming ? t('Close') : t('Rename class')}
          </Button>
        }
      />

      <div className="d-flex flex-wrap gap-2 mb-3">
        <Badge>{year}</Badge>
        {section && section.year_level_code ? (
          <Badge tone="info">{section.year_level_code}</Badge>
        ) : null}
        {section && section.capacity !== null && section.capacity !== undefined ? (
          <Badge>capacity {section.capacity}</Badge>
        ) : null}
      </div>

      {renaming ? (
        <div className="mb-3">
          <Card className="sis-rise" title={t('Rename {0}', [classCode])}>
            <RenameClass
              section={section || { code: classCode }}
              year={year}
              onDone={() => setRenaming(false)}
            />
          </Card>
        </div>
      ) : null}

      {/* Pills that scroll rather than wrap. Three fit across a 360px phone. */}
      <ul
        className="nav nav-pills flex-nowrap overflow-auto mb-3 gap-1"
        style={{ scrollbarWidth: 'none' }}
        role="tablist"
      >
        {TABS.map((item) => (
          <li className="nav-item" key={item.key}>
            <button
              className={cx('nav-link text-nowrap', tab === item.key && 'active')}
              role="tab"
              aria-selected={tab === item.key ? 'true' : 'false'}
              /* `setParams` replaces the hash and re-resolves, so the render below follows
                 from the URL — no second copy of "which tab is open" to fall out of step. */
              onClick={() => Router.setParams({ tab: item.key })}
            >
              {item.label}
            </button>
          </li>
        ))}
      </ul>

      <div className="sis-fade" key={tab}>
        {tab === 'register' ? (
          <Register
            classCode={classCode}
            year={year}
            yearLevel={section && section.year_level_code}
          />
        ) : null}
        {tab === 'attendance' ? (
          <AttendancePanel
            classCode={classCode}
            year={year}
            on={params.on}
            /* Where this class sits, so the panel can ask whether the signed-in person may
               write *this* register rather than registers in general. The rung matters as
               much as the room: a grade supervisor holds the rung and no single class, and
               a question that named only the class would hide the Save button from the
               person whose job it is. */
            scope={{
              school: school,
              yearLevel: section && section.year_level_code,
              classSection: classCode
            }}
          />
        ) : null}
        {tab === 'marks' ? (
          <ClassMarks
            classCode={classCode}
            year={year}
            yearLevel={section && section.year_level_code}
          />
        ) : null}
      </div>
    </>
  );
}
