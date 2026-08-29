/*
 * Student — one child's record, as a card.
 *
 * The deepest screen in the console and the one a parent's question ends at: who she is, who to
 * call, where she has been placed, what the school has stated about her work, and which days she
 * was in the room. Everything on it is either a fact the school recorded or a count of those
 * facts. That constraint is the whole design.
 *
 * **The Insights panel counts; it does not judge.** "Marked present on 41 of 47 recorded days"
 * is a count. "92% attendance" is a rate the school never stated, computed over a denominator
 * the console chose — and a console that publishes it has invented a figure that will be quoted
 * back to it in a meeting. Likewise: no averages across subjects, no ranking against classmates,
 * no "improving" or "at risk". The service reports; this screen arranges.
 *
 * **Unmarked is not absent, and unmarked is not zero.** An attendance day nobody recorded is
 * missing from `days` and excluded from every count, and the panel says how many days it counted
 * over. A subject nobody has marked is `is_graded: false` and renders as a dash through
 * `gradeText`, never as 0%.
 *
 * **Her age is read from her date of birth, never stored beside it.** The service computes it,
 * so it cannot drift; when there is no date of birth there is no age, and both show a dash.
 *
 * **Placements are history, not a column.** The list is every class she has been in, with dates,
 * newest first — a transfer in March leaves October's placement intact rather than rewriting it,
 * and this screen is where that pays off.
 */
import { useState } from 'react';
import { api, gradeText } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import {
  DASH,
  countText,
  dateText,
  labelOf,
  pickName,
  today,
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
  PageHead,
  SearchField,
  Select,
  Skeleton,
  Table,
  Tile
} from '../components/Ui.jsx';
import { StudentEditor } from '../components/StudentEditor.jsx';
import { t } from '../i18n.js';

/** The window the attendance panel opens on: this school year so far, and never "all time". */
function defaultWindow() {
  const now = new Date();
  /* A school year turns over in the summer, so January belongs to the year that started last
     September. Getting this wrong would show an empty register every spring term. */
  const year = now.getMonth() >= 7 ? now.getFullYear() : now.getFullYear() - 1;
  return { from: `${year}-08-01`, to: today() };
}

/* -- Find a child ----------------------------------------------------------------- */

function Finder({ initial }) {
  const state = useStore();
  const [typed, setTyped] = useState(initial || '');
  const [asked, setAsked] = useState('');

  const found = useQuery(() => api.searchStudents(asked), [asked], !!asked);
  const students = (found.value && found.value.students) || [];

  return (
    <div className="vstack gap-4">
      <Card title={t('Find a child')}>
        <form
          className="row g-2 align-items-end"
          onSubmit={(event) => {
            event.preventDefault();
            setAsked(typed.trim());
          }}
        >
          <Field
            className="col-12 col-sm"
            label={t('Student number or name')}
            hint={t('A partial name matches in either script.')}
          >
            {/* No placeholder: the field's own label already says what goes in it, and a
                placeholder repeating it is a second copy that disappears the moment anyone
                types. */}
            <SearchField value={typed} onInput={setTyped} />
          </Field>
          <div className="col-12 col-sm-auto d-grid">
            <Button type="submit" variant="primary" icon="search">
              {t('Search')}
            </Button>
          </div>
        </form>
      </Card>

      <ErrorNote error={found.error} onRetry={found.reload} />

      {asked ? (
        <Card title={t('Results')} subtitle={found.value ? t('{0} found', [found.value.count]) : null} tight>
          <Table
            loading={found.loading}
            rows={students}
            rowKey={(row) => row.student_number}
            rowHref={(row) => Router.href('student', { number: row.student_number })}
            rowLabel={(row) => t('Open {0}', [pickName(row, state.lang) || row.student_number]).join('')}
            empty={
              <Empty title={t('Nothing matches “{0}”', [asked])}>
                {t('Try the student number. A name typed in one script does not match a record that only carries the other.')}
              </Empty>
            }
            columns={[
              {
                key: 'number',
                header: t('Student no.'),
                className: 'sis-code',
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
                header: t('Name'),
                className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
                cell: (row) =>
                  pickName(row, state.lang) || (
                    <span className="sis-ungraded">{DASH} name not on file</span>
                  )
              },
              {
                key: 'age',
                header: t('Age'),
                className: 'sis-num',
                hide: 'sm',
                cell: (row) =>
                  row.age === null || row.age === undefined ? (
                    <span className="sis-ungraded">{DASH}</span>
                  ) : (
                    row.age
                  )
              },
              {
                key: 'active',
                header: '',
                cell: (row) => (row.is_active ? null : <Badge tone="warn">{t('inactive')}</Badge>)
              }
            ]}
          />
        </Card>
      ) : null}
    </div>
  );
}

/* -- Who she is, and who to call -------------------------------------------------- */

function Identity({ student }) {
  const state = useStore();
  const rows = [
    { label: 'Student number', value: student.student_number, mono: true },
    { label: 'Name (English)', value: student.full_name_en },
    { label: 'Name (Arabic)', value: student.full_name_ar, ar: true },
    { label: 'Date of birth', value: student.date_of_birth, mono: true },
    {
      label: 'Age',
      value:
        student.age === null || student.age === undefined ? null : `${student.age} years`,
      note: 'Read from her date of birth, never stored beside it.'
    },
    { label: 'Phone', value: student.contact_phone, mono: true },
    { label: 'Email', value: student.contact_email },
    { label: 'Address', value: student.address }
  ];

  return (
    <Card title={t('Record')} tight>
      <div className="table-responsive">
        <table className="table table-sm align-middle mb-0">
          <tbody>
            {rows.map((row) => (
              <tr key={row.label}>
                <th scope="row" className="text-body-tertiary fw-normal small" style={{ width: '11rem' }}>
                  {row.label}
                </th>
                <td className={row.mono ? 'font-monospace' : row.ar ? 'sis-name-ar' : null}>
                  {row.value ? (
                    row.value
                  ) : (
                    /* Blank is "nobody has stated this", and it says so rather than showing an
                       empty cell a registrar cannot tell apart from a rendering bug. */
                    <span className="sis-ungraded">{DASH} not on file</span>
                  )}
                  {row.note ? (
                    <div className="sis-xs text-body-tertiary">{row.note}</div>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!student.is_active ? (
        <div className="card-footer">
          <Badge tone="warn">{t('inactive')}</Badge>{' '}
          <span className="small text-body-tertiary">
            {t('She is not on any current register. Her record, marks and attendance are unchanged — inactive is a statement about now, not a deletion.')}
          </span>
        </div>
      ) : null}
    </Card>
  );
}

/* -- Where she has been placed ---------------------------------------------------- */

function Placements({ studentNumber }) {
  const state = useStore();
  const placements = useResource(
    Store.keys.placements(studentNumber),
    () => api.studentPlacements(studentNumber),
    !!studentNumber
  );

  /* `.placements`, not the response itself: the route answers
     `{student_number, count, placements: [...]}`. Reading it as a bare array threw
     `rows.filter is not a function` and took the whole screen out — found by walking the
     console against the real service, because the smoke fixture had the shape wrong in the
     same way the screen did. */
  const rows = ((placements.value && placements.value.placements) || [])
    .slice()
    .sort((a, b) => String(b.starts_on).localeCompare(String(a.starts_on)));

  return (
    <Card
      title={t('Placements')}
      subtitle={t('{0} in the record — newest first', [rows.length])}
      actions={
        <Button size="sm" icon="refresh" onClick={placements.reload}>
          {t('Reload')}
        </Button>
      }
      tight
    >
      <ErrorNote error={placements.error} onRetry={placements.reload} />
      <Table
        loading={placements.loading}
        rows={rows}
        rowKey={(row) => `${row.academic_year_code}:${row.class_code}:${row.starts_on}`}
        empty={
          <Empty title={t('She has never been placed in a class')}>
            {t('Her record exists and no class has claimed her. Place her from a class register.')}
          </Empty>
        }
        columns={[
          {
            key: 'class',
            header: t('Class'),
            className: 'sis-code',
            cell: (row) => (
              <a
                href={Router.href('class', {
                  code: row.class_code,
                  year: row.academic_year_code
                })}
              >
                {row.class_code}
              </a>
            )
          },
          {
            key: 'year',
            header: t('Year'),
            className: 'sis-code',
            hide: 'sm',
            cell: (row) => row.academic_year_code
          },
          {
            key: 'from',
            header: t('From'),
            className: 'sis-num',
            cell: (row) => <span className="font-monospace small">{dateText(row.starts_on)}</span>
          },
          {
            key: 'to',
            header: t('To'),
            className: 'sis-num',
            cell: (row) =>
              row.is_open ? (
                <Badge tone="ok">{t('open')}</Badge>
              ) : (
                <span className="font-monospace small">{dateText(row.ends_on)}</span>
              )
          }
        ]}
      />
      <div className="card-footer small text-body-tertiary">
        {t('A placement is a dated membership, so a transfer closes one row and opens another. October still says what it said in October — nothing here is rewritten when she moves.')}
      </div>
    </Card>
  );
}

/* -- Who to call ------------------------------------------------------------------ */

function Guardians({ studentNumber }) {
  const state = useStore();
  const guardians = useResource(
    Store.keys.guardians(studentNumber),
    () => api.studentGuardians(studentNumber),
    !!studentNumber
  );
  const rows = (guardians.value && guardians.value.guardians) || [];

  return (
    <Card
      title={t('Guardians')}
      subtitle={guardians.value ? t('{0} on her contact list', [guardians.value.count]) : null}
      actions={
        <a className="btn btn-sm btn-quiet" href={Router.href('guardians')}>
          {t('Manage')}
        </a>
      }
      tight
    >
      <ErrorNote error={guardians.error} onRetry={guardians.reload} />
      <Table
        loading={guardians.loading}
        rows={rows}
        rowKey={(row) => row.phone}
        empty={
          <Empty title={t('No adult is linked to her')}>
            {t('Nobody here can be told she was absent. Link a guardian from the Guardians screen.')}
          </Empty>
        }
        columns={[
          {
            key: 'name',
            header: t('Adult'),
            className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
            cell: (row) => (
              <>
                {pickName(row, state.lang) || (
                  <span className="sis-ungraded">{DASH} name not on file</span>
                )}
                {/* Who the school rings first. Worth a badge rather than a column: it is true
                    of one adult on the list and the answer to "who do I call". */}
                {row.is_primary_contact ? (
                  <>
                    {' '}
                    <Badge tone="info">{t('first call')}</Badge>
                  </>
                ) : null}
                <div className="font-monospace sis-xs text-body-tertiary">{row.phone}</div>
              </>
            )
          },
          {
            key: 'relationship',
            header: t('Relationship'),
            hide: 'sm',
            cell: (row) => (
              <>
                {row.relationship_type}
                {row.relationship_label ? (
                  <div className="sis-xs text-body-tertiary">“{row.relationship_label}”</div>
                ) : null}
              </>
            )
          },
          {
            key: 'records',
            header: t('May read her records'),
            cell: (row) => (
              <>
                {row.can_view_records ? (
                  <Badge tone="ok">{t('yes')}</Badge>
                ) : (
                  <Badge tone="warn">{t('no')}</Badge>
                )}
                {/* The reason a school gave for withholding access, shown where the decision
                    is. A restriction with no stated reason is the one a registrar reverses by
                    accident. */}
                {row.restriction_note ? (
                  <div className="sis-xs text-body-tertiary">{row.restriction_note}</div>
                ) : null}
              </>
            )
          }
        ]}
      />
    </Card>
  );
}

/* -- What the school has stated about her work ------------------------------------ */

function Marks({ studentNumber }) {
  const state = useStore();
  const [term, setTerm] = useState('');

  const terms = useResource(
    Store.keys.terms(state.year),
    () => api.terms(state.year),
    !!state.year
  );
  const termList = (terms.value || [])
    .slice()
    .sort((a, b) => (a.sequence || 0) - (b.sequence || 0));
  const chosen = term || (termList.length ? termList[0].code : '');

  /* One-shot rather than keyed: the term is a control on this panel, and a registrar
     stepping through four terms wants each read fresh rather than four cache entries. */
  const report = useQuery(
    () => api.studentGrades(studentNumber, chosen),
    [studentNumber, chosen],
    !!(studentNumber && chosen)
  );
  const card = report.value;
  const lines = (card && card.grades) || [];

  if (!state.year) {
    return (
      <Card title={t('Marks')}>
        <Alert tone="info" title={t('Choose a year first')}>
          {t('Terms belong to an academic year, and so does a report card. Pick one in the header.')}
        </Alert>
      </Card>
    );
  }

  return (
    <Card
      title={t('Marks')}
      subtitle={
        card
          ? `${card.graded_count} of ${card.subject_count} subject(s) marked in ${card.term_code}`
          : null
      }
      actions={
        <Select
          size="sm"
          value={chosen}
          options={termList.map((item) => ({
            value: item.code,
            label: `${item.code}${item.is_closed ? ' (closed)' : ''}`
          }))}
          placeholder={terms.loading ? t('Loading…') : t('Choose a term')}
          onInput={setTerm}
        />
      }
      tight
    >
      <ErrorNote error={report.error} onRetry={report.reload} />

      {card ? (
        <div className="card-body pb-0">
          <div className="d-flex flex-wrap gap-2">
            <Badge tone="info">{card.term_code}</Badge>
            {card.term_is_closed ? <Badge tone="warn">{t('term closed')}</Badge> : null}
            {card.class_code ? (
              <Badge>
                in {card.class_code} this term
              </Badge>
            ) : (
              <Badge tone="warn">{t('no placement covered this term')}</Badge>
            )}
          </div>
        </div>
      ) : null}

      <Table
        loading={report.loading}
        rows={lines}
        rowKey={(row) => row.subject_code}
        empty={
          <Empty title={t('No subject rows for this term')}>
            {t('Either the year has no subjects yet, or nothing has been uploaded against this term.')}
          </Empty>
        }
        columns={[
          {
            key: 'subject',
            header: t('Subject'),
            cell: (row) => (
              <>
                <span className="sis-code">{row.subject_code}</span>
                <div className={state.lang === 'ar' ? 'sis-name-ar small' : 'small'}>
                  {pickName(
                    {
                      name_ar: row.subject_name_ar,
                      name_en: row.subject_name_en
                    },
                    state.lang
                  )}
                </div>
              </>
            )
          },
          {
            key: 'mark',
            header: t('Stated'),
            className: 'sis-num',
            cell: (row) => (
              /* Through `gradeText`, always. `row.percentage || DASH` would print a dash for a
                 genuine zero, and `?? 0` would print 0% for a subject nobody has marked. Both
                 read as harmless and both put a figure in front of a parent that the school
                 never stated. */
              <span className={row.is_graded ? 'fw-semibold' : 'sis-ungraded'}>
                {gradeText(row)}
              </span>
            )
          },
          {
            key: 'points',
            header: t('Points'),
            className: 'sis-num',
            hide: 'md',
            cell: (row) =>
              row.points === null || row.points === undefined ? (
                <span className="sis-ungraded">{DASH}</span>
              ) : (
                <span className="font-monospace small">
                  {row.points}
                  {row.max_points ? ` / ${row.max_points}` : ''}
                </span>
              )
          }
        ]}
      />

      <div className="card-footer small text-body-tertiary">
        {t('Figures exactly as the school stated them, in subject order. Nothing here is averaged, weighted or ranked, and a dash is a subject awaiting a mark — never a zero.')}
      </div>
    </Card>
  );
}

/* -- Which days she was in the room, and the counts over them ---------------------- */

function Attendance({ studentNumber }) {
  const [range, setRange] = useState(defaultWindow);

  const record = useResource(
    Store.keys.attendance(studentNumber, range.from, range.to),
    () => api.studentAttendance(studentNumber, range.from, range.to),
    !!studentNumber
  );
  const counts = (record.value && record.value.counts) || null;
  const days = (record.value && record.value.days) || [];

  return (
    <Card
      title={t('Attendance')}
      subtitle={
        record.value
          ? `${dateText(record.value.from_date)} to ${dateText(record.value.to_date)}, both included`
          : null
      }
      actions={
        <div className="d-flex flex-wrap gap-2 align-items-end">
          <Input
            type="date"
            className="form-control-sm"
            value={range.from}
            onInput={(value) => setRange({ ...range, from: value })}
          />
          <Input
            type="date"
            className="form-control-sm"
            value={range.to}
            onInput={(value) => setRange({ ...range, to: value })}
          />
        </div>
      }
      tight
    >
      <ErrorNote error={record.error} onRetry={record.reload} />

      {counts ? (
        <div className="card-body">
          <div className="row g-2 row-cols-2 row-cols-sm-3 row-cols-lg-6">
            <div className="col">
              <Tile label={t('Recorded days')} value={countText(counts.recorded)} />
            </div>
            <div className="col">
              <Tile label={t('Present')} value={countText(counts.present)} />
            </div>
            <div className="col">
              <Tile label={t('Late')} value={countText(counts.late)} />
            </div>
            <div className="col">
              <Tile label={t('Absent')} value={countText(counts.absent)} />
            </div>
            <div className="col">
              <Tile label={t('Excused')} value={countText(counts.excused)} />
            </div>
            <div className="col">
              <Tile
                label={t('In the room')}
                value={countText(counts.in_the_room)}
                note="Present plus late."
              />
            </div>
          </div>
          <p className="small text-body-tertiary mt-3 mb-0">
            Counted over the {counts.recorded} day(s) somebody actually marked in this window. A
            day nobody marked is not in this list and is in none of these counts — it is not an
            absence.
          </p>
        </div>
      ) : null}

      <Table
        loading={record.loading}
        rows={days.slice().reverse()}
        rowKey={(row) => row.on_date}
        rowTone={(row) =>
          row.state === 'present' ? 'ok' : row.state === 'absent' ? 'bad' : 'warn'
        }
        empty={
          <Empty title={t('No marks in this window')}>
            {t('Nobody took a register for her between these two dates. Widen the window, or take one from her class.')}
          </Empty>
        }
        columns={[
          {
            key: 'date',
            header: t('Day'),
            className: 'sis-num',
            cell: (row) => <span className="font-monospace small">{dateText(row.on_date)}</span>
          },
          { key: 'state', header: 'Marked', cell: (row) => row.state },
          {
            key: 'class',
            header: t('In class'),
            className: 'sis-code',
            hide: 'sm',
            /* The class stored on the mark, not her class now: a transfer in March must leave
               October's register saying 3A. */
            cell: (row) => row.class_code
          },
          {
            key: 'note',
            header: t('Reason'),
            hide: 'md',
            cell: (row) =>
              row.note ? (
                <span className="small">{row.note}</span>
              ) : (
                <span className="sis-ungraded">{DASH}</span>
              )
          }
        ]}
      />
    </Card>
  );
}

/* -- Insights: counts and facts, and nothing derived ------------------------------ */

function Insights({ student, studentNumber }) {
  const range = defaultWindow();

  /* The same three keys the panels below use, so this counts what they show and costs no
     extra request. */
  const attendance = useResource(
    Store.keys.attendance(studentNumber, range.from, range.to),
    () => api.studentAttendance(studentNumber, range.from, range.to),
    !!studentNumber
  );
  const placements = useResource(
    Store.keys.placements(studentNumber),
    () => api.studentPlacements(studentNumber),
    !!studentNumber
  );
  const guardians = useResource(
    Store.keys.guardians(studentNumber),
    () => api.studentGuardians(studentNumber),
    !!studentNumber
  );

  if (attendance.loading || placements.loading || guardians.loading) return <Skeleton rows={4} />;

  const counts = (attendance.value && attendance.value.counts) || null;
  const rows = (placements.value && placements.value.placements) || [];
  const open = rows.filter((row) => row.is_open);
  const contacts = (guardians.value && guardians.value.guardians) || [];
  const readers = contacts.filter((row) => row.can_view_records);

  /* Every line below is a count of something recorded, or the plain absence of a record. There
     is deliberately no rate, no average and no trend: the school stated marks and register
     entries, and a percentage computed here is a figure the school would be asked to defend. */
  const facts = [
    counts
      ? {
          label: 'Days marked in this school year so far',
          value: `${counts.recorded}`,
          note: `${dateText(range.from)} to ${dateText(range.to)}`
        }
      : null,
    counts
      ? {
          label: 'Days marked present or late',
          value: `${counts.in_the_room} of ${counts.recorded} recorded`,
          note: 'A count over recorded days, not a rate over the term.'
        }
      : null,
    counts && counts.away
      ? {
          label: 'Days marked absent or excused',
          value: `${counts.away} of ${counts.recorded} recorded`,
          note: `${counts.absent} absent, ${counts.excused} excused.`
        }
      : null,
    {
      label: 'Classes in her record',
      value: `${rows.length}`,
      note: open.length
        ? `Currently in ${open.map((row) => row.class_code).join(', ')}.`
        : t('No open placement — she is on no current register.')
    },
    {
      label: 'Adults on her contact list',
      value: `${contacts.length}`,
      note: `${readers.length} may read her records.`
    },
    {
      label: 'Date of birth',
      value: student.date_of_birth ? dateText(student.date_of_birth) : 'not on file',
      note:
        student.age === null || student.age === undefined
          ? t('No age can be stated without one.')
          : `${student.age} years old today.`
    }
  ].filter(Boolean);

  return (
    <Card title={t('Insights')} subtitle={t('Counts of what the school recorded — nothing derived')} tight>
      <div className="table-responsive">
        <table className="table table-sm align-middle mb-0">
          <tbody>
            {facts.map((fact) => (
              <tr key={fact.label}>
                <th scope="row" className="fw-normal small text-body-tertiary">
                  {fact.label}
                </th>
                <td>
                  <span className="fw-semibold">{fact.value}</span>
                  {fact.note ? (
                    <div className="sis-xs text-body-tertiary">{fact.note}</div>
                  ) : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card-footer small text-body-tertiary">
        {t('No attendance rate, no subject average, no ranking and no trend. Each of those is a figure the school never stated, computed over a denominator this screen would have chosen for it.')}
      </div>
    </Card>
  );
}

/* -- Screen ---------------------------------------------------------------------- */

export function Student({ params = {} }) {
  const state = useStore();
  const number = params.number || '';
  const [editing, setEditing] = useState(false);

  const record = useResource(Store.keys.student(number), () => api.student(number), !!number);

  /* No number in the URL is not an error: it is the Find-a-child screen, which is what the nav
     link points at. */
  if (!number) {
    return (
      <>
        <PageHead
          title={t('Find a child')}
          lede={t('By student number, or by part of a name in either script.')}
        />
        <Finder />
      </>
    );
  }

  if (record.loading && !record.value) {
    return (
      <>
        <PageHead title={number} />
        <Skeleton rows={6} />
      </>
    );
  }

  if (record.error) {
    return (
      <>
        <PageHead title={number} />
        <ErrorNote error={record.error} onRetry={record.reload} />
        <Finder initial={number} />
      </>
    );
  }

  const student = record.value;
  if (!student) return null;

  const name = pickName(student, state.lang);

  return (
    <>
      <Breadcrumbs
        trail={[
          { label: 'Schools', to: 'school' },
          { label: 'Find a child', to: 'student' },
          { label: number }
        ]}
      />
      <PageHead
        title={name || number}
        lede={
          name
            ? `Student number ${student.student_number}.`
            : t('No name is on file for her, which is a gap in the record rather than a rendering fault.')
        }
        actions={
          <>
            <Button variant={editing ? 'primary' : 'outline'} onClick={() => setEditing(!editing)}>
              {editing ? t('Close') : t('Edit her record')}
            </Button>
            <Button icon="refresh" onClick={record.reload}>
              {t('Reload')}
            </Button>
          </>
        }
      />

      {editing ? (
        <div className="mb-3">
          <Card className="sis-rise" title={t('Edit {0}', [student.student_number])}>
            <StudentEditor
              student={student}
              onDone={() => {
                setEditing(false);
                record.reload();
              }}
            />
          </Card>
        </div>
      ) : null}

      {/* Two columns from `lg` up, one on a phone, and the order is the order a question gets
          asked: who she is, who to call, then the work and the days. */}
      <div className="row g-3">
        <div className="col-12 col-lg-6">
          <div className="vstack gap-3">
            <Identity student={student} />
            <Guardians studentNumber={number} />
            <Placements studentNumber={number} />
          </div>
        </div>
        <div className="col-12 col-lg-6">
          <div className="vstack gap-3">
            <Insights student={student} studentNumber={number} />
            <Marks studentNumber={number} />
            <Attendance studentNumber={number} />
          </div>
        </div>
      </div>
    </>
  );
}
