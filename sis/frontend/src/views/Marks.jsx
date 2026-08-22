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

/* -- Screen ---------------------------------------------------------------------- */

export function Marks({ params = {} }) {
  const state = useStore();
  const year = state.year;
  const [term, setTerm] = useState('');
  const [subject, setSubject] = useState('');
  const [classCode, setClassCode] = useState('');

  const terms = useResource(Store.keys.terms(year), () => api.terms(year), !!year);
  /* This year's catalogue. A subject belongs to a year now, so the picker cannot offer a code
     the chosen term's year does not teach. */
  const subjects = useResource(
    Store.keys.subjects(year, false),
    () => api.subjects(year, false),
    !!year
  );
  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);

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
          hint={t('Optional. Leave empty if the sheet has a subject_code column.')}
        >
          <Select
            className="sis-code"
            value={subject}
            placeholder={t('— from the file —')}
            options={(subjects.value || []).map((item) => ({
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
