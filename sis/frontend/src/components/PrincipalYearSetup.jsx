import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { Store } from '../store.js';
import { pickName, useStore } from '../hooks.js';
import { t } from '../i18n.js';
import { Alert, Badge, Button, Card, Empty, ErrorNote, Field, Input, Select } from './Ui.jsx';
import { SubjectBoard } from '../views/Year.jsx';

let draftSequence = 1;

function blankYear(school) {
  return {
    id: draftSequence++,
    code: '',
    name_en: '',
    name_ar: '',
    starts_on: '',
    ends_on: '',
    is_current: true,
    school_code: school || ''
  };
}

function shiftIsoYear(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value || '')) return '';
  const [year, month, day] = value.split('-').map(Number);
  const shifted = new Date(Date.UTC(year + 1, month - 1, day));
  return [shifted.getUTCFullYear(), String(shifted.getUTCMonth() + 1).padStart(2, '0'), String(shifted.getUTCDate()).padStart(2, '0')].join('-');
}

function nextCode(value, school) {
  const match = String(value || '').match(/^(.*?)(\d{4})-(\d{4})$/);
  if (match) return `${match[1]}${Number(match[2]) + 1}-${Number(match[3]) + 1}`;
  return `${school || 'SCHOOL'}-`;
}

function nextYear(previous, school) {
  return {
    ...blankYear(school),
    code: nextCode(previous && previous.code, school),
    starts_on: shiftIsoYear(previous && previous.starts_on),
    ends_on: shiftIsoYear(previous && previous.ends_on),
    is_current: false
  };
}

function errorMessage(reason) {
  if (!reason) return '';
  return reason.message || (reason.detail && reason.detail.message) || String(reason);
}

function cairoTodayIso() {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Africa/Cairo', year: 'numeric', month: '2-digit', day: '2-digit'
  }).formatToParts(new Date());
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function isLockedYear(row) {
  return !!(row && row.is_current && row.starts_on && row.starts_on <= cairoTodayIso());
}

function TrackClassPlan({ track, grades, plan, onChange, lang }) {
  const rows = grades || [];
  const custom = plan.mode === 'custom';
  return (
    <div className="border rounded-3 p-3 vstack gap-3">
      <div className="d-flex flex-wrap align-items-center justify-content-between gap-2">
        <div>
          <strong>{pickName(track, lang) || track.code}</strong>
          <div className="small text-body-tertiary">{t('{0} grade(s)', [rows.length])}</div>
        </div>
        <Badge>{track.code}</Badge>
      </div>
      <div className="row g-3">
        <Field className="col-12 col-md-4" label={t('Class distribution')}>
          <Select value={plan.mode} options={[
            { value: 'same', label: t('Same number for every grade') },
            { value: 'custom', label: t('Different number per grade') }
          ]} onChange={(value) => onChange({ ...plan, mode: value })} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Class naming')}>
          <Select value={plan.sequence} options={[
            { value: 'numeric', label: t('Numeric sections') },
            { value: 'alphabetic', label: t('Alphabetic sections') }
          ]} onChange={(value) => onChange({ ...plan, sequence: value })} />
        </Field>
        {!custom ? (
          <Field className="col-12 col-md-4" label={t('Classes per grade')}>
            <Input type="number" min="0" max="60" inputMode="numeric" value={String(plan.sameCount)}
              onInput={(value) => onChange({ ...plan, sameCount: Math.max(0, Number(value) || 0) })} />
          </Field>
        ) : null}
      </div>
      {custom ? (
        <div className="row g-2">
          {rows.map((level) => (
            <Field className="col-6 col-md-4 col-xl-3" key={level.code} label={pickName(level, lang) || level.code}>
              <Input type="number" min="0" max="60" inputMode="numeric"
                value={String(plan.byGrade[level.code] ?? 1)}
                onInput={(value) => onChange({
                  ...plan,
                  byGrade: { ...plan.byGrade, [level.code]: Math.max(0, Number(value) || 0) }
                })} />
            </Field>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function SubjectCreator({ year, onSaved, disabled = false }) {
  const [form, setForm] = useState({ code: '', name_en: '', name_ar: '' });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const ready = !disabled && !!year && !!form.code.trim() && (!!form.name_en.trim() || !!form.name_ar.trim());

  const save = async () => {
    if (!ready || saving) return;
    setSaving(true); setError(null);
    try {
      await api.createSubject({
        code: form.code.trim(), academic_year_code: year,
        name_en: form.name_en.trim(), name_ar: form.name_ar.trim(),
        display_order: 0, is_active: true
      });
      setForm({ code: '', name_en: '', name_ar: '' });
      Store.invalidate('subjects:');
      if (onSaved) onSaved();
      Store.toast('ok', t('Subject added'), year);
    } catch (reason) { setError(reason); }
    finally { setSaving(false); }
  };

  return <div className="vstack gap-3">
    <div className="row g-3">
      <Field className="col-12 col-md-3" label={t('Subject code')} required>
        <Input className="sis-code" disabled={disabled} value={form.code} placeholder="MATH"
          onInput={(value) => setForm((old) => ({ ...old, code: value }))} />
      </Field>
      <Field className="col-12 col-md-4" label={t('Name (English)')}>
        <Input className="sis-name-en" disabled={disabled} value={form.name_en} onInput={(value) => setForm((old) => ({ ...old, name_en: value }))} />
      </Field>
      <Field className="col-12 col-md-4" label={t('Name (Arabic)')}>
        <Input className="sis-name-ar" disabled={disabled} value={form.name_ar} onInput={(value) => setForm((old) => ({ ...old, name_ar: value }))} />
      </Field>
      <div className="col-12 col-md-1 d-grid align-items-end">
        <Button variant="primary" disabled={!ready} pending={saving} onClick={save}>+</Button>
      </div>
    </div>
    {error ? <ErrorNote error={error} /> : null}
  </div>;
}

export function PrincipalYearSetup({ school, schoolConfig, tracks = [], levels = [], years = [], onSaved }) {
  const state = useStore();
  const [drafts, setDrafts] = useState(() => [blankYear(school)]);
  const [classPlans, setClassPlans] = useState({});
  const [configuredByTrack, setConfiguredByTrack] = useState({});
  const [configuredLoading, setConfiguredLoading] = useState(false);
  const [configuredError, setConfiguredError] = useState(null);
  const [legacy, setLegacy] = useState({ year_count: 6, classes_per_year: 2, suffixes: 'A,B,C,D' });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState([]);
  const [configureYear, setConfigureYear] = useState('');
  const [subjects, setSubjects] = useState([]);
  const [subjectError, setSubjectError] = useState(null);
  const [subjectLoading, setSubjectLoading] = useState(false);

  useEffect(() => {
    setDrafts([blankYear(school)]);
    setResult([]);
  }, [school]);

  useEffect(() => {
    const initial = {};
    tracks.forEach((track) => {
      initial[track.code] = { mode: 'same', sameCount: 1, sequence: 'numeric', byGrade: {} };
    });
    setClassPlans((old) => ({ ...initial, ...old }));
  }, [tracks.map((row) => row.code).join('|')]);

  useEffect(() => {
    let alive = true;
    if (!school || !tracks.length) {
      setConfiguredByTrack({});
      setConfiguredError(null);
      return () => { alive = false; };
    }
    setConfiguredLoading(true);
    setConfiguredError(null);
    Promise.all(tracks.map(async (track) => [track.code, await api.configuredGrades(school, track.code)]))
      .then((pairs) => { if (alive) setConfiguredByTrack(Object.fromEntries(pairs)); })
      .catch((reason) => { if (alive) setConfiguredError(reason); })
      .finally(() => { if (alive) setConfiguredLoading(false); });
    return () => { alive = false; };
  }, [school, tracks.map((row) => row.code).join('|')]);

  const yearOptions = useMemo(() => {
    const combined = [...years, ...result];
    const seen = new Set();
    return combined.filter((row) => {
      if (!row || !row.code || seen.has(row.code)) return false;
      seen.add(row.code); return true;
    }).map((row) => ({ value: row.code, label: pickName(row, state.lang) || row.code }));
  }, [years, result, state.lang]);

  const configureRow = [...years, ...result].find((row) => row && row.code === configureYear);
  const configureLocked = isLockedYear(configureRow);

  useEffect(() => {
    if (!configureYear && yearOptions.length) setConfigureYear(yearOptions[yearOptions.length - 1].value);
  }, [yearOptions.length, configureYear]);

  const loadSubjects = async () => {
    if (!configureYear) { setSubjects([]); return; }
    setSubjectLoading(true); setSubjectError(null);
    try { setSubjects(await api.subjects(configureYear, true)); }
    catch (reason) { setSubjectError(reason); }
    finally { setSubjectLoading(false); }
  };
  useEffect(() => { loadSubjects(); }, [configureYear]);

  const updateDraft = (id, patch) => setDrafts((rows) => rows.map((row) => row.id === id ? { ...row, ...patch } : row));
  const setCurrent = (id) => setDrafts((rows) => rows.map((row) => ({ ...row, is_current: row.id === id })));
  const removeDraft = (id) => setDrafts((rows) => rows.length === 1 ? rows : rows.filter((row) => row.id !== id));

  const draftCodes = drafts.map((row) => row.code.trim()).filter(Boolean);
  const existingCodes = new Set(years.map((row) => String(row.code)));
  const codesAreUnique = new Set(draftCodes).size === draftCodes.length;
  const codesAreNew = draftCodes.every((code) => !existingCodes.has(code));
  const namesPresent = drafts.every((row) => !!row.name_en.trim() || !!row.name_ar.trim());
  const currentYearsStartInFuture = drafts.every((row) => !row.is_current || !row.starts_on || row.starts_on > cairoTodayIso());
  const valid = drafts.length > 0 && drafts.every((row) =>
    row.code.trim() && row.starts_on && row.ends_on && row.ends_on >= row.starts_on &&
    (!!row.name_en.trim() || !!row.name_ar.trim())
  ) && codesAreUnique && codesAreNew && currentYearsStartInFuture && !configuredLoading && !configuredError;

  const saveAll = async () => {
    if (!valid || saving) return;
    setSaving(true); setError(null); setResult([]);
    const created = [];
    try {
      for (const draft of drafts) {
        const year = await api.createAcademicYear({
          code: draft.code.trim(), school_code: school,
          name_en: draft.name_en.trim(), name_ar: draft.name_ar.trim(),
          starts_on: draft.starts_on, ends_on: draft.ends_on,
          is_current: !!draft.is_current
        });
        created.push(year);

        if (tracks.length) {
          for (const track of tracks) {
            const plan = classPlans[track.code] || { mode: 'same', sameCount: 1, sequence: 'numeric', byGrade: {} };
            const body = {
              academic_year_code: year.code,
              track_code: track.code,
              mode: plan.mode,
              sequence: plan.sequence
            };
            if (plan.mode === 'custom') {
              const gradeRows = configuredByTrack[track.code] || [];
              body.classes_by_grade = Object.fromEntries(gradeRows.map((level) => [level.code, Number(plan.byGrade[level.code] ?? 1)]));
            } else {
              body.class_count = Number(plan.sameCount) || 0;
            }
            await api.createConfiguredClasses(body);
          }
        } else {
          const suffixes = String(legacy.suffixes || '').split(',').map((item) => item.trim()).filter(Boolean);
          await api.generateStructure({
            academic_year_code: year.code,
            year_count: Number(legacy.year_count) || 1,
            classes_per_year: Number(legacy.classes_per_year) || 0,
            class_suffixes: suffixes.length ? suffixes : null
          });
        }
      }

      Store.invalidate('years:'); Store.invalidate('levels:'); Store.invalidate('classes:'); Store.invalidate('terms:');
      const current = created.find((row) => row.is_current) || created[created.length - 1];
      if (current) Store.setYear(current.code);
      setResult(created);
      if (created.length) setConfigureYear(created[created.length - 1].code);
      Store.toast('ok', t('{0} academic year(s) created', [created.length]), t('Classes were created from the school plan.'));
      if (onSaved) onSaved(created);
    } catch (reason) {
      setError(reason);
    } finally { setSaving(false); }
  };

  return <div className="vstack gap-4">
    <Alert tone="info" title={t('Academic year setup')}>
      {t('Create one or several academic years, build their classes, then configure subjects by drag and drop. Existing years and data are never deleted by this setup.')}
    </Alert>

    <Card title={t('1. Academic years')} subtitle={t('Add as many years as you need before saving.') }>
      <div className="vstack gap-3">
        {drafts.map((row, index) => <div className="border rounded-3 p-3" key={row.id}>
          <div className="d-flex align-items-center justify-content-between gap-2 mb-3">
            <strong>{t('Academic year {0}', [index + 1])}</strong>
            {drafts.length > 1 ? <Button size="sm" variant="danger" onClick={() => removeDraft(row.id)}>{t('Remove')}</Button> : null}
          </div>
          <div className="row g-3">
            <Field className="col-12 col-md-4" label={t('Year code')} required>
              <Input className="sis-code" value={row.code} placeholder={`${school}-2026-2027`}
                onInput={(value) => updateDraft(row.id, { code: value })} />
            </Field>
            <Field className="col-12 col-md-4" label={t('Name (English)')}>
              <Input className="sis-name-en" value={row.name_en} onInput={(value) => updateDraft(row.id, { name_en: value })} />
            </Field>
            <Field className="col-12 col-md-4" label={t('Name (Arabic)')}>
              <Input className="sis-name-ar" value={row.name_ar} onInput={(value) => updateDraft(row.id, { name_ar: value })} />
            </Field>
            {!row.name_en.trim() && !row.name_ar.trim() ? (
              <div className="col-12 small text-danger">{t('Enter the academic year name in English, Arabic, or both.')}</div>
            ) : null}
            <Field className="col-12 col-md-4" label={t('First day')} required>
              <Input type="date" value={row.starts_on} onInput={(value) => updateDraft(row.id, { starts_on: value })} />
            </Field>
            <Field className="col-12 col-md-4" label={t('Last day')} required>
              <Input type="date" value={row.ends_on} onInput={(value) => updateDraft(row.id, { ends_on: value })} />
            </Field>
            <div className="col-12 col-md-4 d-flex align-items-end">
              <label className="form-check mb-2">
                <input className="form-check-input" type="radio" name="principal-current-year" checked={!!row.is_current}
                  onChange={() => setCurrent(row.id)} />
                <span className="form-check-label">{t('Make current academic year')}</span>
              </label>
            </div>
          </div>
        </div>)}
        <div className="d-flex flex-wrap gap-2">
          <Button onClick={() => setDrafts((rows) => [...rows, nextYear(rows[rows.length - 1], school)])}>{t('Add another academic year')}</Button>
        </div>
      </div>
    </Card>

    <Card title={t('2. Classes and sections')} subtitle={t('The same class plan is applied to every year in this batch.') }>
      {tracks.length ? <div className="vstack gap-3">
        {configuredError ? <ErrorNote error={configuredError} /> : null}
        {configuredLoading ? <div className="small text-body-tertiary">{t('Loading…')}</div> : null}
        {tracks.map((track) => <TrackClassPlan key={track.code} track={track} grades={configuredByTrack[track.code] || []}
          plan={classPlans[track.code] || { mode: 'same', sameCount: 1, sequence: 'numeric', byGrade: {} }}
          onChange={(next) => setClassPlans((old) => ({ ...old, [track.code]: next }))} lang={state.lang} />)}
      </div> : <div className="row g-3">
        <Field className="col-12 col-md-4" label={t('Year levels')}>
          <Input type="number" min="1" value={String(legacy.year_count)} onInput={(value) => setLegacy((old) => ({ ...old, year_count: Number(value) || 1 }))} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Sections per level')}>
          <Input type="number" min="0" value={String(legacy.classes_per_year)} onInput={(value) => setLegacy((old) => ({ ...old, classes_per_year: Number(value) || 0 }))} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Section suffixes')}>
          <Input className="sis-code" value={legacy.suffixes} onInput={(value) => setLegacy((old) => ({ ...old, suffixes: value }))} />
        </Field>
      </div>}
      <div className="mt-3 small text-body-tertiary">
        {t('The school is configured for {0} term(s). Term sections are created automatically with every academic year.', [schoolConfig ? schoolConfig.term_count : '—'])}
      </div>
    </Card>

    {error ? <ErrorNote error={error} /> : null}
    <div className="d-flex flex-wrap align-items-center gap-3">
      <Button variant="primary" disabled={!valid} pending={saving} pendingLabel={t('Creating…')} onClick={saveAll}>
        {t('Create academic year setup')}
      </Button>
      {!valid ? <span className="small text-body-tertiary">
        {!codesAreUnique || !codesAreNew
          ? t('Year codes must be new and unique within this batch.')
          : !namesPresent
            ? t('Each academic year needs at least one name: English, Arabic, or both.')
            : !currentYearsStartInFuture
              ? t('A current academic year must be fully configured before its first day.')
              : t('Every year needs a code, first day and last day.')}
      </span> : null}
    </div>

    {result.length ? <Alert tone="ok" title={t('Academic years created')}>
      <div className="d-flex flex-wrap gap-2">{result.map((row) => <Badge key={row.code}>{row.code}</Badge>)}</div>
    </Alert> : null}

    <Card title={t('3. Subjects and grade assignment')} subtitle={t('Select a year, add subjects, then drag each subject onto the grades that teach it.') }>
      {yearOptions.length ? <div className="vstack gap-3">
        <Field label={t('Academic year')}>
          <Select value={configureYear} options={yearOptions} onChange={setConfigureYear} />
        </Field>
        {configureLocked ? (
          <Alert tone="warn" title={t('Academic year locked')}>
            {t('This is the current academic year and its first day has arrived. Subjects, grade assignments and classes are now read-only.')}
          </Alert>
        ) : null}
        <SubjectCreator year={configureYear} onSaved={loadSubjects} disabled={configureLocked} />
        {subjectError ? <ErrorNote error={subjectError} onRetry={loadSubjects} /> : null}
        {subjectLoading ? <div className="small text-body-tertiary">{t('Loading…')}</div> : null}
        {!subjectLoading && configureYear ? <SubjectBoard year={configureYear} school={school} levels={levels} subjects={subjects} lang={state.lang} readOnly={configureLocked} /> : null}
      </div> : <Empty title={t('Create an academic year first')}>
        {t('Subject configuration becomes available as soon as an academic year exists.')}
      </Empty>}
    </Card>
  </div>;
}
