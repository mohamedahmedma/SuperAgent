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
import { SCHOOL_LEVELS, STAGES, byStage } from '../structure.js';

/* The week, Saturday first, as an Egyptian school reads it. The value is what the service
   stores. The labels are built by a call rather than held in a constant because `t` has to run
   after the language is known, and again on every switch — a module-level table would freeze
   whichever language happened to be current at import. */
const WORKING_DAYS = ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday'];
const dayLabels = () => ({
  saturday: t('Saturday'),
  sunday: t('Sunday'),
  monday: t('Monday'),
  tuesday: t('Tuesday'),
  wednesday: t('Wednesday'),
  thursday: t('Thursday'),
  friday: t('Friday')
});

/* -- Add a school ---------------------------------------------------------------- */

function SchoolForm({ onSaved }) {
  const form = useForm({
    code: '', name_en: '', name_ar: '', language_type: '',
    kg_grade_count: 0, primary_grade_count: 0,
    preparatory_grade_count: 0, secondary_grade_count: 0,
    term_count: '', working_days: []
  });
  const levelSelected = SCHOOL_LEVELS.some((level) => Number(form.values[level.column]) > 0);
  const ready = levelSelected && !!form.values.language_type && !!form.values.term_count &&
    form.values.working_days.length > 0 && !!form.values.name_en.trim() && !!form.values.name_ar.trim();
  const dayLabel = dayLabels();
  const save = useAction(() =>
    api.createSchool({
      code: form.values.code.trim(),
      name_en: form.values.name_en.trim(),
      name_ar: form.values.name_ar.trim(),
      language_type: form.values.language_type,
      kg_grade_count: Number(form.values.kg_grade_count),
      primary_grade_count: Number(form.values.primary_grade_count),
      preparatory_grade_count: Number(form.values.preparatory_grade_count),
      secondary_grade_count: Number(form.values.secondary_grade_count),
      term_count: Number(form.values.term_count),
      working_days: form.values.working_days,
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
            Store.toast('ok', t('School {0} saved', [school.code]));
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
        <Field className="col-12 col-sm-4" label={t('Name (English)')} required>
          <Input value={form.values.name_en} required onInput={form.set('name_en')} />
        </Field>
        <Field className="col-12 col-sm-4" label={t('Name (Arabic)')} required>
          <Input className="sis-name-ar" value={form.values.name_ar} required onInput={form.set('name_ar')} />
        </Field>
      </div>
      <fieldset>
        <legend className="form-label fs-6">{t('School language type')}</legend>
        <div className="d-flex flex-wrap gap-3">
          {[
            ['arabic', t('Arabic')],
            ['languages', t('Languages')],
            ['both', t('Arabic and Languages')]
          ].map(([value, label]) => (
            <label className="form-check" key={value}>
              <input className="form-check-input" type="radio" name="school-language" required
                checked={form.values.language_type === value} onChange={() => form.set('language_type')(value)} />
              <span className="form-check-label">{label}</span>
            </label>
          ))}
        </div>
      </fieldset>
      <fieldset>
        <legend className="form-label fs-6">{t('Educational levels')}</legend>
        <div className="row g-3">
          {SCHOOL_LEVELS.map((level) => {
            const selected = Number(form.values[level.column]) > 0;
            return <div className="col-12 col-sm-6 col-lg-3" key={level.key}>
              <label className="form-check mb-2">
                <input className="form-check-input" type="checkbox" checked={selected}
                  onChange={(event) => form.set(level.column)(event.target.checked ? 1 : 0)} />
                <span className="form-check-label">{t(level.label)}</span>
              </label>
              {selected ? <Select value={String(form.values[level.column])}
                options={Array.from({ length: level.max }, (_, index) => ({ value: String(index + 1), label: t('{0} grade(s)', [index + 1]) }))}
                onChange={(value) => form.set(level.column)(Number(value))} /> : null}
            </div>;
          })}
        </div>
        {!levelSelected ? <div className="small text-danger mt-2">{t('Select at least one educational level.')}</div> : null}
      </fieldset>
      <fieldset>
        <legend className="form-label fs-6">{t('Number of academic terms')}</legend>
        <div className="d-flex gap-3">
          {[1, 2, 3].map((count) => <label className="form-check" key={count}>
            <input className="form-check-input" type="radio" name="term-count" required
              checked={Number(form.values.term_count) === count} onChange={() => form.set('term_count')(count)} />
            <span className="form-check-label">{t('{0} term(s)', [count])}</span>
          </label>)}
        </div>
      </fieldset>
      <fieldset>
        <legend className="form-label fs-6">{t('School working days')}</legend>
        <div className="d-flex flex-wrap gap-3">
          {WORKING_DAYS.map((day) => <label className="form-check" key={day}>
            <input className="form-check-input" type="checkbox" checked={form.values.working_days.includes(day)}
              onChange={(event) => form.set('working_days')(event.target.checked
                ? WORKING_DAYS.filter((item) => item === day || form.values.working_days.includes(item))
                : form.values.working_days.filter((item) => item !== day))} />
            <span className="form-check-label">{dayLabel[day]}</span>
          </label>)}
        </div>
      </fieldset>
      <p className="small text-body-tertiary mb-0">
        {t("Year codes are unique across the whole service, so give this school's years a code of their own —")} <span className="sis-code">{t('NC-2025-2026')}</span> rather than{' '}
        <span className="sis-code">2025-2026</span> — if another branch already uses the plain one.
      </p>
      <ErrorNote error={save.error} />
      <div className="d-grid d-sm-block">
        <Button type="submit" variant="primary" disabled={!ready} pending={save.pending} pendingLabel={t('Saving…')}>
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
            Store.toast('ok', t('Academic year {0} saved', [year.code]), school);
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

function LevelForm({ school, schoolConfig, track, count, onSaved }) {
  /* Only the stages this school switched on at creation. A rung the school does not run is
     not an option here, and the service refuses it anyway. */
  const enabledStages = SCHOOL_LEVELS.filter(
    (level) => Number(schoolConfig && schoolConfig[level.column]) > 0
  );
  const form = useForm({
    code: '',
    name_en: '',
    name_ar: '',
    display_order: String((count || 0) + 1),
    stage: enabledStages[0] ? enabledStages[0].key : 'unspecified',
    track_code: track
  });

  const save = useAction(() =>
    api.createLevel({
      code: form.values.code.trim(),
      school_code: school,
      track_code: track,
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
            Store.toast('ok', t('Rung {0} saved', [level.code]), school);
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
            options={enabledStages.map((stage) => ({ value: stage.key, label: t(stage.label) }))}
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
        aria-label={t('Open {0} {1}', [level.code, pickName(level, lang)]).join('')}
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
                  Store.toast('ok', t('{0} moved to {1}', [level.code, t(STAGES.find((item) => item.key === next)?.label || next)]));
                })
                .catch((error) => {
                  setStage(level.stage);
                  Store.toast('bad', t('Could not move {0}', [level.code]), error.message);
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
  const [activeTrack, setActiveTrack] = useState('');

  const schools = useResource(Store.keys.schools(false), () => api.schools(false));
  const schoolList = schools.value || [];

  /* The URL wins over the remembered school, so a pasted link lands where it says. */
  const code = params.code || state.school;
  useEffect(() => {
    if (params.code && params.code !== state.school) Store.setSchool(params.code);
  }, [params.code]);

  const levels = useResource(Store.keys.levels(code), () => api.schoolLevels(code), !!code);
  const tracks = useResource(Store.keys.tracks(code), () => api.schoolTracks(code), !!code);
  const years = useResource(Store.keys.years(code), () => api.years(code), !!code);
  const trackList = tracks.value || [];
  const selectedTrack = trackList.some((track) => track.code === activeTrack)
    ? activeTrack
    : (trackList[0] && trackList[0].code) || '';

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
  /* Rungs of the selected track, plus any that belong to no track at all.
     An untracked rung is one the school had before it declared its sections — revision 0009
     could only place those where the school runs a single section. Hiding it under every
     track would make it unreachable from the only screen that lists rungs, which is a worse
     answer than showing it in both. */
  const levelList = (levels.value || []).filter(
    (level) => !selectedTrack || !level.track_code || level.track_code === selectedTrack
  );

  /* Classes per rung, for the count on each card. Counted from the selected year's classes: a
     rung's class count is a statement about a year, not about the rung. */
  const perLevel = {};
  (classes.value || []).forEach((section) => {
    perLevel[section.year_level_code] = (perLevel[section.year_level_code] || 0) + 1;
  });

  const grouped = byStage(levelList);

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
        {trackList.length ? (
          <Card title={t('Academic track')} subtitle={t('You are managing this structure independently.')}>
            <div className="btn-group" role="group" aria-label={t('Academic track')}>
              {trackList.map((track) => (
                <Button
                  key={track.code}
                  variant={track.code === selectedTrack ? 'primary' : 'secondary'}
                  onClick={() => setActiveTrack(track.code)}
                >
                  {pickName(track, state.lang)}
                </Button>
              ))}
            </div>
          </Card>
        ) : null}
        {addingSchool ? (
          <Card className="sis-rise" title={t('New school')}>
            <SchoolForm onSaved={() => setAddingSchool(false)} />
          </Card>
        ) : null}

        {addingYear && code ? (
          <Card className="sis-rise" title={t('New academic year in {0}', [code])}>
            <YearForm school={code} onSaved={() => setAddingYear(false)} />
          </Card>
        ) : null}

        {addingLevel && code ? (
          <Card className="sis-rise" title={t('New rung in {0}', [code])}>
            <LevelForm
              school={code}
              schoolConfig={school}
              track={selectedTrack}
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
                rowLabel={(row) => t('Open academic year {0}', [row.code]).join('')}
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
