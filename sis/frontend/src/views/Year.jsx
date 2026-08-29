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
  Empty,
  ErrorNote,
  Field,
  Input,
  NoYearNotice,
  PageHead,
  Select,
  Table,
  useConfirm
} from '../components/Ui.jsx';

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
      starts_on: form.values.starts_on,
      ends_on: form.values.ends_on,
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
          required
          error={form.errorFor(save.error, 'starts_on')}
        >
          <Input type="date" value={form.values.starts_on} onInput={form.set('starts_on')} />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('Last day')}
          required
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
        <Section
          title={t('Terms — {0}', [code])}
          subtitle={t('{0} in this year', [termList.length])}
          action="Add term"
          form={<TermForm year={code} count={termList.length} />}
        >
          <Table
            loading={terms.loading}
            rows={termList}
            rowKey={(row) => row.code}
            empty={
              <Empty title={t('No terms yet')}>
                {t('Marks are recorded against a term, so no mark can be uploaded until one exists.')}
              </Empty>
            }
            columns={[
              { key: 'seq', header: '#', className: 'sis-num', hide: 'sm', cell: (row) => row.sequence },
              { key: 'code', header: 'Code', className: 'sis-code', cell: (row) => row.code },
              {
                key: 'name',
                header: t('Name'),
                className: nameClass,
                hide: 'md',
                cell: (row) => pickName(row, state.lang)
              },
              {
                key: 'span',
                header: t('Runs'),
                className: 'sis-num',
                hide: 'lg',
                cell: (row) => (
                  <span className="font-monospace small">
                    {dateText(row.starts_on)} → {dateText(row.ends_on)}
                  </span>
                )
              },
              {
                key: 'state',
                header: t('Marks'),
                cell: (row) =>
                  row.is_closed ? <Badge tone="warn">{t('closed')}</Badge> : <Badge tone="ok">{t('open')}</Badge>
              },
              {
                key: 'upload',
                header: '',
                hide: 'md',
                cell: (row) => (
                  <a
                    className="btn btn-sm btn-quiet"
                    href={Router.href('marks', { term: row.code })}
                  >
                    {t('Upload marks')}
                  </a>
                )
              }
            ]}
          />
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

        <Card title={t('Generate the ladder')}>
          <Generator year={code} />
        </Card>
      </div>
    </>
  );
}
