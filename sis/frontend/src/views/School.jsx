/*
 * School — one branch: its academic years, and its ladder grouped by division.
 *
 * The ladder is the point of the screen. A fourteen-rung school is unreadable as a flat list, so
 * the rungs are grouped into garden, primary, preparatory and secondary, in that order, with
 * anything unclassified last — a rung nobody has grouped yet belongs at the bottom rather than
 * above the kindergarten.
 *
 * A stage is a label and carries no rules: nothing is barred from a class, a term or a subject
 * because of it. That is why reclassifying a rung is an ordinary edit rather than a confirmed
 * one — it moves a card between two headings and touches nothing else.
 *
 * On a phone each rung is a stacked card rather than a row of five things, and the class count
 * and the division picker wrap under the code. The rung itself stays the tap target.
 */
import { useEffect, useState } from 'react';
import { ApiError, api } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import { dateText, pickName, useAction, useForm, useResource, useStore } from '../hooks.js';
import { t } from '../i18n.js';
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Field,
  Input,
  PageHead,
  Select,
  Skeleton,
  Table,
  Tile,
  useConfirm
} from '../components/Ui.jsx';

/* The order a school lists its own divisions in, and the labels it uses. */
const STAGES = [
  { key: 'garden', label: 'Garden' },
  { key: 'primary', label: 'Primary' },
  { key: 'preparatory', label: 'Preparatory' },
  { key: 'secondary', label: 'Secondary' },
  { key: 'unspecified', label: 'Not yet grouped' }
];

/* -- Add a school ---------------------------------------------------------------- */

function SchoolForm({ onSaved }) {
  const form = useForm({ code: '', name_en: '', name_ar: '' });
  const save = useAction(() =>
    api.createSchool({
      code: form.values.code.trim(),
      name_en: form.values.name_en.trim(),
      name_ar: form.values.name_ar.trim(),
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
          .then((school) => {
            Store.invalidate('schools:');
            Store.setSchool(school.code);
            Store.toast('ok', `School ${school.code} saved`);
            form.reset();
            if (onSaved) onSaved(school);
          })
          .catch(() => {});
      }}
    >
      <div className="row g-3">
        <Field
          className="col-12 col-sm-4"
          label={t('School code')}
          required
          hint={t('Immutable. Every year and rung in the school points at it.')}
          error={form.errorFor(save.error, 'code')}
        >
          <Input
            className="sis-code"
            value={form.values.code}
            placeholder="NC"
            onInput={form.set('code')}
          />
        </Field>
        <Field className="col-12 col-sm-4" label={t('Name (English)')}>
          <Input value={form.values.name_en} onInput={form.set('name_en')} />
        </Field>
        <Field className="col-12 col-sm-4" label={t('Name (Arabic)')}>
          <Input className="sis-name-ar" value={form.values.name_ar} onInput={form.set('name_ar')} />
        </Field>
      </div>
      <p className="small text-body-tertiary mb-0">
        {t("Year codes are unique across the whole service, so give this school's years a code of their own —")} <span className="sis-code">{t('NC-2025-2026')}</span> rather than{' '}
        <span className="sis-code">2025-2026</span> — if another branch already uses the plain one.
      </p>
      <ErrorNote error={save.error} />
      <div className="d-grid d-sm-block">
        <Button type="submit" variant="primary" pending={save.pending} pendingLabel={t('Saving…')}>
          {t('Add school')}
        </Button>
      </div>
    </form>
  );
}

/* -- Add an academic year -------------------------------------------------------
 *
 * On this screen because this is where a school with no years lands, and a school with no years
 * can do nothing at all: classes are generated into a year, terms belong to one, subjects are
 * taught in one and every upload names one.
 */

function YearForm({ school, onSaved }) {
  const form = useForm({
    code: '',
    name_en: '',
    name_ar: '',
    starts_on: '',
    ends_on: '',
    is_current: true
  });

  const save = useAction(() =>
    api.createAcademicYear({
      code: form.values.code.trim(),
      school_code: school,
      name_en: form.values.name_en.trim(),
      name_ar: form.values.name_ar.trim(),
      starts_on: form.values.starts_on,
      ends_on: form.values.ends_on,
      is_current: !!form.values.is_current
    })
  );

  return (
    <form
      className="vstack gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        save
          .run()
          .then((year) => {
            Store.invalidate('years:');
            Store.setYear(year.code);
            Store.toast('ok', `Academic year ${year.code} saved`, school);
            form.reset();
            if (onSaved) onSaved(year);
          })
          .catch(() => {});
      }}
    >
      <div className="row g-3">
        <Field
          className="col-12 col-sm-6 col-lg-4"
          label={t('Year code')}
          required
          hint={t('Unique across the service, so prefix it when another branch uses the plain code.')}
          error={form.errorFor(save.error, 'code')}
        >
          <Input
            className="sis-code"
            value={form.values.code}
            placeholder={`${school || 'NC'}-2026-2027`}
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
          hint={t('Inclusive.')}
          error={form.errorFor(save.error, 'ends_on')}
        >
          <Input type="date" value={form.values.ends_on} onInput={form.set('ends_on')} />
        </Field>
      </div>

      <div className="form-check">
        <input
          className="form-check-input"
          type="checkbox"
          id="year-current"
          checked={!!form.values.is_current}
          onChange={(event) => form.set('is_current')(event.target.checked)}
        />
        <label className="form-check-label small" htmlFor="year-current">
          Make this the working year for this school. Each school has its own current year;
          marking this one does not touch another branch's.
        </label>
      </div>

      <ErrorNote error={save.error} />
      <div className="d-grid d-sm-block">
        <Button type="submit" variant="primary" pending={save.pending} pendingLabel={t('Saving…')}>
          {t('Add academic year')}
        </Button>
      </div>
    </form>
  );
}

/* -- Add a rung ------------------------------------------------------------------ */

function LevelForm({ school, count, onSaved }) {
  const form = useForm({
    code: '',
    name_en: '',
    name_ar: '',
    display_order: String((count || 0) + 1),
    stage: 'unspecified'
  });

  const save = useAction(() =>
    api.createLevel({
      code: form.values.code.trim(),
      school_code: school,
      name_en: form.values.name_en.trim(),
      name_ar: form.values.name_ar.trim(),
      display_order: Number(form.values.display_order) || 0,
      stage: form.values.stage
    })
  );

  return (
    <form
      className="vstack gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        save
          .run()
          .then((level) => {
            Store.invalidate('levels:');
            Store.invalidate('years:');
            Store.toast('ok', `Rung ${level.code} saved`, school);
            form.reset();
            if (onSaved) onSaved(level);
          })
          .catch(() => {});
      }}
    >
      <div className="row g-3">
        <Field
          className="col-12 col-sm-6 col-lg-3"
          label={t('Rung code')}
          required
          hint={t('Unique within this school; other branches may use the same one.')}
          error={form.errorFor(save.error, 'code')}
        >
          <Input
            className="sis-code"
            value={form.values.code}
            placeholder="Y1"
            onInput={form.set('code')}
          />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-3"
          label={t('Division')}
          error={form.errorFor(save.error, 'stage')}
        >
          <Select
            value={form.values.stage}
            options={STAGES.map((stage) => ({ value: stage.key, label: stage.label }))}
            onChange={form.set('stage')}
          />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-3" label={t('Name (English)')}>
          <Input value={form.values.name_en} onInput={form.set('name_en')} />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-3" label={t('Name (Arabic)')}>
          <Input className="sis-name-ar" value={form.values.name_ar} onInput={form.set('name_ar')} />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-3" label={t('Order within the division')}>
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
          {t('Add rung')}
        </Button>
      </div>
    </form>
  );
}

/* -- One rung -------------------------------------------------------------------- */

function Rung({ level, school, year, lang, classCount }) {
  const [dialog, ask] = useConfirm();
  const [stage, setStage] = useState(level.stage);

  return (
    <div className="card mb-2 sis-row-open">
      {dialog}
      {/*
        * The whole rung opens, not the words on it.
        *
        * This used to be an anchor wrapped around the code and the name, so the target was the
        * text and everything to the right of it — most of a very wide row — was dead. The
        * stretched anchor covers the card instead, and the picker and the Remove button beside
        * it are lifted clear by `sis-row-actions` so they keep their own clicks.
        */}
      <a
        className="sis-row-target"
        href={Router.href('level', { school, code: level.code, year })}
        aria-label={`Open ${level.code} ${pickName(level, lang)}`}
      />
      <div className="card-body d-flex flex-column flex-md-row align-items-md-center gap-2 gap-md-3">
        <span className="d-flex align-items-center gap-3 flex-grow-1">
          <span className="sis-rung-code">{level.code}</span>
          <span className={lang === 'ar' ? 'sis-name-ar flex-grow-1' : 'sis-name-en flex-grow-1'}>
            {pickName(level, lang)}
          </span>
          <span className="small text-body-tertiary text-nowrap">
            {classCount === null || classCount === undefined ? '' : t('{0} class(es)', [classCount])}
          </span>
        </span>

        <div className="d-flex align-items-center gap-2 sis-row-actions">
          <Select
            size="sm"
            value={stage}
            options={STAGES.map((item) => ({ value: item.key, label: t(item.label) }))}
            onChange={(next) => {
              setStage(next);
              /* An ordinary edit: a stage is a label, so moving a rung between divisions
                 changes which heading it sits under and nothing else. No dialog. */
              api
                .createLevel({
                  code: level.code,
                  school_code: school,
                  name_en: level.name_en,
                  name_ar: level.name_ar,
                  display_order: level.display_order,
                  stage: next
                })
                .then(() => {
                  Store.invalidate('levels:');
                  Store.toast('ok', `${level.code} moved to ${next}`);
                })
                .catch((error) => {
                  setStage(level.stage);
                  Store.toast('bad', `Could not move ${level.code}`, error.message);
                });
            }}
          />

          <Button
            size="sm"
            variant="quiet"
            onClick={() =>
              ask({
                title: `Remove rung ${level.code}?`,
                tone: 'bad',
                confirmLabel: 'Remove it',
                body: (
                  <>
                    <p>
                      {t("A rung can only be removed while nothing points at it. This school's classes on")} <span className="sis-code">{level.code}</span> {t('would each have to be removed first, and the service will refuse while any of them exist.')}
                    </p>
                    <p className="small text-body-tertiary mb-0">
                      {t("That refusal is the database's, not this screen's — which is why it holds even when two registrars click at once.")}
                    </p>
                  </>
                ),
                run: () =>
                  /*
                   * There is no delete route, deliberately: nothing in this service deletes
                   * structure, because a rung with a class under it carries marks and registers
                   * with it. The honest thing is to say so rather than offer a button that
                   * cannot work.
                   */
                  Promise.reject(
                    ApiError(
                      'client',
                      0,
                      'not_supported',
                      'Rungs are not deleted by this service. Nothing that has ever held a class ' +
                        'can be removed without taking its marks and registers with it — leave it ' +
                        'in place, or move it to a division you do not use.',
                      'code'
                    )
                  )
              })
            }
          >
            {t('Remove')}
          </Button>
        </div>
      </div>
    </div>
  );
}

/* -- Screen ---------------------------------------------------------------------- */

export function School({ params = {} }) {
  const state = useStore();
  const [addingSchool, setAddingSchool] = useState(false);
  const [addingLevel, setAddingLevel] = useState(false);
  const [addingYear, setAddingYear] = useState(false);

  const schools = useResource(Store.keys.schools(false), () => api.schools(false));
  const schoolList = schools.value || [];

  /* The URL wins over the remembered school, so a pasted link lands where it says. */
  const code = params.code || state.school;
  useEffect(() => {
    if (params.code && params.code !== state.school) Store.setSchool(params.code);
  }, [params.code]);

  const levels = useResource(Store.keys.levels(code), () => api.schoolLevels(code), !!code);
  const years = useResource(Store.keys.years(code), () => api.years(code), !!code);

  /* The remembered year, but only once it is known to be one of *this* school's.
     `Store.setSchool` drops the year when the school changes, and says why. It runs in the
     effect above, though, which is after the first render -- so arriving at another school
     by URL renders once with the previous school's year still selected. That render fetched
     `classes` for the school just left and counted them per rung below, and because rung
     codes are unique per school rather than globally ("Y1" exists at every branch) the
     counts landed on this school's cards under matching codes. A newly created school, with
     no classes at all, showed the class counts of the school the registrar came from.
     Deriving the year from the list this school actually returned closes that: until the
     years arrive, no year is active, and `null` renders as "pick a year" rather than as a
     number belonging to somebody else. */
  const yearBelongsToSchool = ((years.value && years.value.academic_years) || []).some(
    function (row) { return row.code === state.year; }
  );
  const activeYear = yearBelongsToSchool ? state.year : null;

  const classes = useResource(
    Store.keys.classes(activeYear),
    () => api.classes(activeYear),
    !!activeYear
  );

  if (schools.ready && !schoolList.length) {
    return (
      <>
        <PageHead
          title={t('Schools')}
          lede={t('Nothing exists yet. A school comes first: every year, rung, class and mark below it belongs to one.')}
        />
        <Card title={t('Create the first school')}>
          <SchoolForm />
        </Card>
      </>
    );
  }

  const school = schoolList.find((item) => item.code === code);
  const yearList = (years.value && years.value.academic_years) || [];
  const levelList = levels.value || [];

  /* Classes per rung, for the count on each card. Counted from the selected year's classes: a
     rung's class count is a statement about a year, not about the rung. */
  const perLevel = {};
  (classes.value || []).forEach((section) => {
    perLevel[section.year_level_code] = (perLevel[section.year_level_code] || 0) + 1;
  });

  const grouped = STAGES.map((stage) => ({
    stage,
    levels: levelList.filter((level) => (level.stage || 'unspecified') === stage.key)
  })).filter((group) => group.levels.length > 0);

  return (
    <>
      <PageHead
        title={school ? pickName(school, state.lang) || code : code || 'School'}
        lede={
          school
            ? t('Its academic years, and its ladder grouped by division. Open a rung to see its classes.')
            : t('This school is not on file.')
        }
        actions={
          <>
            <Button onClick={() => setAddingSchool(!addingSchool)}>
              {addingSchool ? t('Close') : t('Add school')}
            </Button>
            <Button disabled={!code} onClick={() => setAddingLevel(!addingLevel)}>
              {addingLevel ? t('Close') : t('Add rung')}
            </Button>
            <Button variant="primary" disabled={!code} onClick={() => setAddingYear(!addingYear)}>
              {addingYear ? t('Close') : t('Add academic year')}
            </Button>
          </>
        }
      />

      <div className="vstack gap-4">
        {addingSchool ? (
          <Card className="sis-rise" title={t('New school')}>
            <SchoolForm onSaved={() => setAddingSchool(false)} />
          </Card>
        ) : null}

        {addingYear && code ? (
          <Card className="sis-rise" title={`New academic year in ${code}`}>
            <YearForm school={code} onSaved={() => setAddingYear(false)} />
          </Card>
        ) : null}

        {addingLevel && code ? (
          <Card className="sis-rise" title={`New rung in ${code}`}>
            <LevelForm
              school={code}
              count={levelList.length}
              onSaved={() => setAddingLevel(false)}
            />
          </Card>
        ) : null}

        <div className="row g-3">
          <div className="col-12 col-xl-6">
            <Card
              title={t('Academic years')}
              subtitle={t('{0} in this school', [yearList.length])}
              tight
              className="h-100"
            >
              <Table
                loading={years.loading}
                rows={yearList}
                rowKey={(row) => row.code}
                rowHref={(row) => Router.href('year', { code: row.code })}
                rowLabel={(row) => `Open academic year ${row.code}`}
                empty={
                  <Empty
                    title={t('No academic years in this school')}
                    action={
                      <Button variant="primary" onClick={() => setAddingYear(true)}>
                        {t('Add the first year')}
                      </Button>
                    }
                  >
                    {t('Classes are generated into a year and every upload names one, so a year comes before anything else.')}
                  </Empty>
                }
                columns={[
                  {
                    key: 'code',
                    header: t('Year'),
                    className: 'sis-code',
                    cell: (row) => (
                      <>
                        <a className="sis-plain" href={Router.href('year', { code: row.code })}>
                        {row.code}
                      </a>
                        {row.is_current ? (
                          <>
                            {' '}
                            <Badge tone="info">{t('current')}</Badge>
                          </>
                        ) : null}
                      </>
                    )
                  },
                  {
                    key: 'span',
                    header: t('Runs'),
                    className: 'sis-num',
                    hide: 'md',
                    cell: (row) => (
                      <span className="font-monospace small">
                        {dateText(row.starts_on)} → {dateText(row.ends_on)}
                      </span>
                    )
                  },
                  {
                    key: 'pick',
                    header: '',
                    cell: (row) =>
                      row.code === state.year ? (
                        <span className="small text-body-tertiary">{t('selected')}</span>
                      ) : (
                        <Button size="sm" onClick={() => Store.setYear(row.code)}>
                          {t('Work in this year')}
                        </Button>
                      )
                  }
                ]}
              />
            </Card>
          </div>

          <div className="col-12 col-xl-6">
            <Card title={t('This school')} className="h-100">
              <div className="row row-cols-2 row-cols-lg-3 g-3">
                <Tile
                  label={t('Rungs')}
                  value={levelList.length}
                  loading={levels.loading && !levels.ready}
                  note={t('{0} division(s) in use', [grouped.length])}
                />
                <Tile
                  label={t('Years')}
                  value={yearList.length}
                  loading={years.loading && !years.ready}
                  note={t('Academic years on file')}
                />
                <Tile
                  label={t('Classes')}
                  value={(classes.value || []).length}
                  loading={classes.loading && !classes.ready}
                  note={activeYear ? t('In {0}', [activeYear]) : t('Pick a year')}
                />
              </div>
            </Card>
          </div>
        </div>

        <div>
          <h2 className="h5 mb-1">{t('The ladder')}</h2>
          <p className="text-body-secondary" style={{ maxWidth: '62ch' }}>
            {t('Grouped by division, youngest first. A division is a label for reading a long ladder — no rule anywhere depends on it, so moving a rung between divisions changes which heading it sits under and nothing else.')}
          </p>

          {levels.loading && !levels.ready ? (
            <Card>
              <Skeleton rows={5} />
            </Card>
          ) : null}

          {levels.ready && !levelList.length ? (
            <Card>
              <Empty title={t('This school has no rungs yet')}>
                {t('Add one above, or open an academic year and use the generator to build the whole ladder at once.')}
              </Empty>
            </Card>
          ) : null}

          {grouped.map((group) => (
            <div className="mb-4 sis-fade" key={group.stage.key}>
              <div className="sis-stage-head">
                <span className="sis-stage-name">{t(group.stage.label)}</span>
                <span className="sis-stage-rule" />
                <span className="small text-body-tertiary">{t('{0} rung(s)', [group.levels.length])}</span>
              </div>
              {group.levels.map((level) => (
                <Rung
                  key={level.code}
                  level={level}
                  school={code}
                  year={activeYear}
                  lang={state.lang}
                  classCount={activeYear ? perLevel[level.code] || 0 : null}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
