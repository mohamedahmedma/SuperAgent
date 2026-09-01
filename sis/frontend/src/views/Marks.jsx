/*
 * Marks — upload a term's stated figures, and read one child's back.
 *
 * The invariant this whole screen is built around: **a blank grade is null, never 0.**
 *
 * Zero is a mark a child can earn. The absence of a mark is not. Every cell here branches on
 * `is_graded` and never on the number, and the rendering goes through `gradeText`, which is the
 * one function in the console allowed to decide what a mark looks like. The idioms that break
 * this — `percentage || '—'`, `percentage ?? 0`, `if (percentage)` — all read as harmless and
 * all produce the same visible failure: a parent told their child scored 0% in a subject nobody
 * has marked yet, or a genuine zero disappearing from a report. The contract suite greps this
 * file for each of them.
 *
 * Nothing here averages, weights, ranks or drops a lowest mark. The service reports what the
 * school recorded; a console that computed a mean would be inventing a figure the school never
 * stated and putting it in front of a parent.
 */
import { useEffect, useState } from 'react';
import { api, gradeText } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import { DASH, labelOf, pickName, useQuery, useResource, useStore } from '../hooks.js';
import {
  Alert,
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Field,
  Input,
  NoYearNotice,
  PageHead,
  Select,
  Table
} from '../components/Ui.jsx';
import { ImportFlow } from '../components/ImportFlow.jsx';
import { t } from '../i18n.js';

const TEMPLATE = {
  name: 'marks-template.csv',
  header: t('student_number,subject_code,percentage')
};

/* -- One child's term ------------------------------------------------------------ */

function StudentMarks({ initial, initialTerm }) {
  const state = useStore();
  const [typed, setTyped] = useState(initial || '');
  const [asked, setAsked] = useState(initial || '');
  const [term, setTerm] = useState(initialTerm || '');

  const terms = useResource(
    Store.keys.terms(state.year),
    () => api.terms(state.year),
    !!state.year
  );
  const termList = (terms.value || []).slice().sort((a, b) => (a.sequence || 0) - (b.sequence || 0));

  /* Default to the first term of the selected year once the list arrives, so the screen is
     usable after typing a number and nothing else. */
  useEffect(() => {
    if (!term && termList.length) setTerm(termList[0].code);
  }, [termList.length]);

  const report = useQuery(
    () => api.studentGrades(asked, term),
    [asked, term],
    !!(asked && term)
  );

  const card = report.value;
  const grades = (card && card.grades) || [];

  return (
    <Card
      title={t("One child's marks")}
      subtitle={
        card ? `${card.graded_count} of ${card.subject_count} subject(s) marked` : 'Stated figures, exactly as recorded'
      }
      actions={
        asked && term ? (
          <Button size="sm" icon="refresh" onClick={report.reload}>
            {t('Reload')}
          </Button>
        ) : null
      }
      tight
    >
      <div className="card-body">
        <form
          className="row g-2 align-items-end"
          onSubmit={(event) => {
            event.preventDefault();
            const value = typed.trim();
            setAsked(value);
            Router.setParams({ student: value || null, term: term || null });
          }}
        >
          <Field className="col-12 col-sm-5" label={t('Student number')}>
            <Input className="sis-code" value={typed} placeholder="10432" onInput={setTyped} />
          </Field>
          <Field className="col-8 col-sm-4" label={t('Term')}>
            <Select
              className="sis-code"
              value={term}
              placeholder={termList.length ? null : '— no terms —'}
              options={termList.map((item) => ({
                value: item.code,
                label: labelOf(item, state.lang)
              }))}
              onChange={setTerm}
            />
          </Field>
          <div className="col-4 col-sm-3 d-grid">
            <Button
              type="submit"
              variant="primary"
              icon="search"
              disabled={!typed.trim() || !term}
            >
              {t('Look up')}
            </Button>
          </div>
        </form>
      </div>

      <ErrorNote error={report.error} onRetry={report.reload} />

      {!asked || !term ? (
        <Empty title={t('No child looked up')}>{t('Type a student number and choose a term.')}</Empty>
      ) : (
        <>
          {card ? (
            <div className="card-body pt-0">
              <div className="row row-cols-1 row-cols-sm-3 g-3">
                <div className="col">
                  <div className="sis-tile-label">{t('Child')}</div>
                  <div className={state.lang === 'ar' ? 'sis-name-ar fw-semibold' : 'sis-name-en fw-semibold'}>
                    {pickName(card, state.lang)}
                  </div>
                  <div className="sis-code sis-xs text-body-tertiary">{card.student_number}</div>
                </div>
                <div className="col">
                  <div className="sis-tile-label">{t('Term')}</div>
                  <div className="d-flex align-items-center gap-2">
                    <span className="sis-code">{card.term_code}</span>
                    {card.term_is_closed ? (
                      <Badge tone="warn">{t('closed')}</Badge>
                    ) : (
                      <Badge tone="ok">{t('open')}</Badge>
                    )}
                  </div>
                </div>
                <div className="col">
                  <div className="sis-tile-label">{t('Class this term')}</div>
                  <div>
                    {card.class_code ? (
                      <span className="sis-code">{card.class_code}</span>
                    ) : (
                      <span className="sis-ungraded">{DASH} no placement for this term</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          <Table
            loading={report.loading}
            rows={grades}
            rowKey={(row) => row.subject_code}
            empty={
              <Empty title={t('No subject rows for this term')}>
                {t('Either no marks have been uploaded for this term, or she has no placement in it.')}
              </Empty>
            }
            columns={[
              {
                key: 'subject',
                header: t('Subject'),
                className: 'sis-code',
                cell: (row) => row.subject_code
              },
              {
                key: 'name',
                header: t('Name'),
                className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
                hide: 'sm',
                cell: (row) => {
                  const name =
                    state.lang === 'ar'
                      ? row.subject_name_ar || row.subject_name_en
                      : row.subject_name_en || row.subject_name_ar;
                  return name || <span className="sis-ungraded">{DASH}</span>;
                }
              },
              {
                key: 'mark',
                header: t('Mark'),
                className: 'sis-num',
                cell: (row) => (
                  /* The one renderer. Returns an em dash when `is_graded` is false and "0%"
                     when the child genuinely scored nothing. */
                  <strong className={row.is_graded ? 'font-monospace' : 'font-monospace sis-ungraded'}>
                    {gradeText(row)}
                  </strong>
                )
              },
              {
                key: 'points',
                header: t('Points'),
                className: 'sis-num',
                hide: 'lg',
                cell: (row) => {
                  /* Points are stated separately and never recomputed from the percentage, so
                     they are absent independently of it. Explicit null checks, because
                     `points || DASH` would erase a real 0 here too. */
                  if (row.points === null || row.points === undefined) {
                    return <span className="sis-ungraded">{DASH}</span>;
                  }
                  const outOf =
                    row.max_points === null || row.max_points === undefined ? null : row.max_points;
                  return (
                    <span className="font-monospace">
                      {row.points}
                      {outOf === null ? '' : ` / ${outOf}`}
                    </span>
                  );
                }
              },
              {
                key: 'state',
                header: '',
                cell: (row) =>
                  row.is_graded ? null : <span className="sis-xs sis-ungraded">{t('not marked')}</span>
              }
            ]}
          />

          {card && card.subject_count > card.graded_count ? (
            <p className="small text-body-tertiary p-3 mb-0">
              {card.subject_count - card.graded_count} subject(s) have no mark recorded for this
              term. That is an absence, not a zero.
            </p>
          ) : null}
        </>
      )}
    </Card>
  );
}

/* -- Teacher workspace ---------------------------------------------------------- */

function TeacherMarks() {
  const state = useStore();
  const year = state.year;
  const [term, setTerm] = useState('');
  const [assignmentKey, setAssignmentKey] = useState('');
  const [assessmentName, setAssessmentName] = useState('');
  const [maxPoints, setMaxPoints] = useState('');
  const [draft, setDraft] = useState({});
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const terms = useResource(Store.keys.terms(year), () => api.terms(year), !!year);
  const teaching = useQuery(() => api.teachingAssignments(year), [year], !!year);
  const assignments = (teaching.value && teaching.value.assignments) || [];
  const selected = assignments.find((row) =>
    `${row.class_code}\u0000${row.subject_code}` === assignmentKey
  );
  const invalidMarks = Object.values(draft).some((value) =>
    value !== '' && (!Number.isFinite(Number(value)) || Number(value) < 0 ||
      !Number(maxPoints) || Number(value) > Number(maxPoints))
  );
  const enteredMarks = Object.entries(draft).filter(([, value]) => value !== '');
  const sheet = useQuery(
    () => api.classMarkSheet(selected.class_code, year, term, selected.subject_code),
    [year, term, assignmentKey],
    !!(selected && term)
  );

  useEffect(() => {
    if (!term && terms.value && terms.value.length) setTerm(terms.value[0].code);
  }, [term, (terms.value || []).length]);
  useEffect(() => {
    if (!assignmentKey && assignments.length) {
      setAssignmentKey(`${assignments[0].class_code}\u0000${assignments[0].subject_code}`);
    }
  }, [assignmentKey, assignments.length]);
  useEffect(() => {
    const next = {};
    ((sheet.value && sheet.value.students) || []).forEach((student) => {
      next[student.student_number] = '';
    });
    setDraft(next);
  }, [sheet.value]);

  if (!year) return <><PageHead title={t('Marks')} /><NoYearNotice /></>;

  const save = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await api.recordClassMarks(selected.class_code, year, {
        term_code: term,
        subject_code: selected.subject_code,
        marks: enteredMarks.map(([number, value]) => ({
          student_number: number,
          points: Number(value),
          max_points: Number(maxPoints),
          clear: false
        }))
      });
      Store.toast(t('Marks saved.'), 'success');
      sheet.reload();
    } catch (error) {
      setSaveError(error);
    } finally {
      setSaving(false);
    }
  };

  return <>
    <PageHead title={t('Marks')} lede={t('Only your assigned classes and subjects are shown.')} />
    <Card title={t('Enter class marks')} tight>
      <div className="card-body row g-3">
        <Field className="col-12 col-md-8" label={t('Assessment name')}>
          <Input value={assessmentName} placeholder={t('January monthly exam')} onInput={setAssessmentName} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Maximum mark')} required>
          <Input type="number" min="0.01" step="0.01" value={maxPoints} onInput={setMaxPoints} />
        </Field>
        <Field className="col-12 col-md-8" label={t('Class')}>
          <Select value={assignmentKey} strict options={assignments.map((row) => ({
            value: `${row.class_code}\u0000${row.subject_code}`,
            label: `${pickName(row, state.lang) || row.class_code} · ${state.lang === 'ar' ? (row.subject_name_ar || row.subject_code) : (row.subject_name_en || row.subject_code)} · ${row.year_level_code}`
          }))} onChange={setAssignmentKey} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Term')}>
          <Select value={term} strict options={(terms.value || []).map((row) => ({
            value: row.code, label: labelOf(row, state.lang)
          }))} onChange={setTerm} />
        </Field>
      </div>
      <ErrorNote error={teaching.error || sheet.error || saveError} onRetry={sheet.reload} />
      {!teaching.loading && !assignments.length ?
        <Empty title={t('No teaching assignments')}>{t('Ask your school manager to assign your subjects and classes.')}</Empty> :
        <Table loading={sheet.loading} rows={(sheet.value && sheet.value.students) || []}
          rowKey={(row) => row.student_number} columns={[
            { key: 'number', header: t('Student number'), className: 'sis-code', cell: (row) => row.student_number },
            { key: 'name', header: t('Student'), cell: (row) => pickName(row, state.lang) },
            { key: 'mark', header: assessmentName ? `${assessmentName} / ${maxPoints || '—'}` : t('Mark'), cell: (row) => <Input type="number" min="0" max={maxPoints || undefined} step="0.01"
              value={draft[row.student_number] === undefined ? '' : draft[row.student_number]}
              disabled={!sheet.value || !sheet.value.may_record || sheet.value.term_is_closed}
              onInput={(value) => setDraft((old) => ({ ...old, [row.student_number]: value }))} /> }
          ]} />}
      {sheet.value ? <div className="card-body d-flex flex-column align-items-end gap-2">
        {invalidMarks ? <span className="small text-danger">
          {t('Every mark must be between zero and the maximum mark.')}
        </span> : null}
        <Button variant="primary" icon="save" disabled={saving || !enteredMarks.length || invalidMarks || !maxPoints || !assessmentName.trim() || !sheet.value.may_record || sheet.value.term_is_closed}
          onClick={save}>{saving ? t('Saving…') : t('Save marks')}</Button>
      </div> : null}
    </Card>
  </>;
}

/* -- Screen ---------------------------------------------------------------------- */

function RegistrarMarks({ params = {} }) {
  const state = useStore();
  const year = state.year;
  const [term, setTerm] = useState('');
  const [subject, setSubject] = useState('');
  const [classCode, setClassCode] = useState('');

  const terms = useResource(Store.keys.terms(year), () => api.terms(year), !!year);
  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);

  /*
   * The subjects the picker may offer.
   *
   * With no class chosen this is the year's whole catalogue, because the sheet is allowed to
   * name its own classes and the upload may span several rungs. Name a class and it narrows
   * to what *that* rung is assigned to teach: the class fixes the rung, and a picker still
   * offering Physics to a Primary class is offering to file a mark against a lesson those
   * children never sat.
   *
   * Both branches are one `useResource` with a key that carries the rung, so switching class
   * refetches rather than re-labelling the previous rung's list.
   */
  const chosenClass = (classes.value || []).find((item) => item.code === classCode);
  const rung = (chosenClass && chosenClass.year_level_code) || '';
  const subjects = useResource(
    rung ? Store.keys.gradeSubjects(year, rung) : Store.keys.subjects(year, false),
    () => api.subjects(year, false, rung),
    !!year
  );

  /* A subject that the newly chosen class does not teach cannot stay selected: it would be
     posted with the upload and refused, with nothing on screen to say why. */
  const offered = subjects.value || [];
  useEffect(() => {
    if (subject && offered.length && !offered.some((item) => item.code === subject)) {
      setSubject('');
    }
  }, [rung, offered.length]);

  const termList = (terms.value || []).slice().sort((a, b) => (a.sequence || 0) - (b.sequence || 0));

  if (!year) {
    return (
      <>
        <PageHead title={t('Marks')} />
        <NoYearNotice />
      </>
    );
  }

  const chosenTerm = termList.find((item) => item.code === term);

  const fields = (
    <div className="vstack gap-3">
      <div className="row g-3">
        <Field
          className="col-12 col-sm-4"
          label={t('Term')}
          required
          hint={t('Marks are recorded against a term, always.')}
        >
          <Select
            className="sis-code"
            value={term}
            strict
            placeholder={t('— choose a term —')}
            options={termList.map((item) => ({
              value: item.code,
              label: labelOf(item, state.lang) + (item.is_closed ? ' (closed)' : '')
            }))}
            onChange={setTerm}
          />
        </Field>
        <Field
          className="col-12 col-sm-4"
          label={t('Subject')}
          hint={
            rung
              ? t('Optional. Only what {0} is assigned to teach.', [rung])
              : t('Optional. Leave empty if the sheet has a subject_code column.')
          }
        >
          <Select
            className="sis-code"
            value={subject}
            placeholder={t('— from the file —')}
            options={offered.map((item) => ({
              value: item.code,
              label: labelOf(item, state.lang)
            }))}
            onChange={setSubject}
          />
        </Field>
        <Field
          className="col-12 col-sm-4"
          label={t('Class')}
          hint={t('Optional. Narrows which children the sheet may name.')}
        >
          <Select
            className="sis-code"
            value={classCode}
            placeholder={t('— from the file —')}
            options={(classes.value || []).map((item) => ({
              value: item.code,
              label: labelOf(item, state.lang)
            }))}
            onChange={setClassCode}
          />
        </Field>
      </div>

      {chosenTerm && chosenTerm.is_closed ? (
        <Alert tone="warn" title={t('This term is closed')}>
          {t('Its marks are final. The upload will be checked as usual and the service will say what it refuses.')}
        </Alert>
      ) : null}
    </div>
  );

  return (
    <>
      <PageHead
        title={t('Marks')}
        lede={t('Stated figures, reported exactly as the school recorded them. Nothing here is averaged, weighted or ranked — and a blank is never a zero.')}
      />

      <div className="vstack gap-4">
        <ImportFlow
          kind="grades"
          template={TEMPLATE}
          label={t('Choose the marks sheet')}
          hint={`Columns: ${TEMPLATE.header}, or points and max_points instead of percentage.`}
          ready={!!term}
          blocker={
            <Alert tone="info">
              {t('Choose the term first. A marks file with no term named is a file the service cannot place.')}
            </Alert>
          }
          fields={fields}
          onPreview={(file) => {
            const form = new FormData();
            form.append('file', file);
            form.append('term_code', term);
            if (subject) form.append('subject_code', subject);
            if (classCode) form.append('class_code', classCode);
            return api.previewGrades(form);
          }}
          onCommit={(batchId) => api.commitGrades(batchId)}
        />

        <StudentMarks initial={params.student} initialTerm={params.term} />
      </div>
    </>
  );
}

export function Marks({ params = {} }) {
  useStore();
  if (Store.roles().indexOf('teacher') >= 0) return <TeacherMarks />;
  if (Store.can('grades.write')) return <RegistrarMarks params={params} />;
  return <><PageHead title={t('Marks')} /><StudentMarks initial={params.student} initialTerm={params.term} /></>;
}
