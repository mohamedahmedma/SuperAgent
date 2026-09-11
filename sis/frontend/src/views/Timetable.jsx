import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api.js';
import { Store } from '../store.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { Badge, Button, Card, Empty, ErrorNote, PageHead, Select, Skeleton } from '../components/Ui.jsx';
import { t } from '../i18n.js';

const DAY_LABELS = {
  sunday: 'Sunday', monday: 'Monday', tuesday: 'Tuesday', wednesday: 'Wednesday',
  thursday: 'Thursday', friday: 'Friday', saturday: 'Saturday'
};

const slotKey = (day, period) => `${day}:${period}`;
const copyEntries = (entries) => (entries || []).map((entry) => ({ ...entry }));

function changedSlots(before, after) {
  const oldBySlot = new Map(before.map((entry) => [
    slotKey(entry.day_of_week, entry.period_number), entry
  ]));
  const nextBySlot = new Map(after.map((entry) => [
    slotKey(entry.day_of_week, entry.period_number), entry
  ]));
  const keys = new Set([...oldBySlot.keys(), ...nextBySlot.keys()]);
  return [...keys].filter((key) => {
    const oldEntry = oldBySlot.get(key);
    const nextEntry = nextBySlot.get(key);
    return !oldEntry || !nextEntry || oldEntry.subject_code !== nextEntry.subject_code;
  });
}

export function Timetable() {
  const state = useStore();
  const [klass, setKlass] = useState('');
  const [grade, setGrade] = useState('');
  const [term, setTerm] = useState('');
  const [dragged, setDragged] = useState(null);
  const [savedEntries, setSavedEntries] = useState([]);
  const [draftEntries, setDraftEntries] = useState([]);
  const [saving, setSaving] = useState(false);
  const [breakPeriod, setBreakPeriod] = useState('');
  const [periodCount, setPeriodCount] = useState('');
  const [dayStartsAt, setDayStartsAt] = useState('');
  const [dayEndsAt, setDayEndsAt] = useState('');
  const [breakDuration, setBreakDuration] = useState('20');
  const [savingDayLayout, setSavingDayLayout] = useState(false);
  const [copying, setCopying] = useState(false);
  const [saved, setSaved] = useState(false);
  const [actionError, setActionError] = useState(null);
  const automaticCopyAttempt = useRef('');

  const supervisedGrades = useMemo(() => [...new Set(
    ((state.profile && state.profile.grants) || [])
      .filter((grant) => grant.permission === 'timetable.read' && grant.scope_type === 'year_level')
      .map((grant) => grant.scope_code).filter(Boolean)
  )], [state.profile]);
  const mayReadEveryClass = Store.canIn('timetable.read', { school: state.school }) &&
    Store.canIn('structure.read', { school: state.school });

  const options = useQuery(async () => {
    const teaching = await api.teachingAssignments(state.year);
    const teacherClasses = (teaching.assignments || []).map((row) => ({
      code: row.class_code,
      name_en: row.class_name_en,
      name_ar: row.class_name_ar,
      year_level_code: row.year_level_code,
      year_level_name_en: row.year_level_name_en,
      year_level_name_ar: row.year_level_name_ar
    }));
    const broadClasses = mayReadEveryClass ? await api.classes(state.year) : [];
    const supervisedClasses = (await Promise.all(
      supervisedGrades.map((grade) => api.classes(state.year, grade))
    )).flat();
    const classes = [...broadClasses, ...supervisedClasses, ...teacherClasses].filter(
      (row, index, rows) => rows.findIndex((item) => item.code === row.code) === index
    );
    const terms = await api.terms(state.year);
    return { classes, terms, teachingAssignments: teaching.assignments || [] };
  }, [state.year, mayReadEveryClass, supervisedGrades.join('|')], !!state.year);

  const classes = (options.value && options.value.classes) || [];
  const terms = (options.value && options.value.terms) || [];
  const isSchoolLeader = (state.profile?.roles || []).some((role) =>
    role.role_code === 'school_owner' || role.role_code === 'school_manager'
  );
  const grades = useMemo(() => [...new Map(classes.map((row) => [row.year_level_code, {
    code: row.year_level_code,
    name_en: row.year_level_name_en,
    name_ar: row.year_level_name_ar
  }])).values()].filter((row) => row.code), [classes]);
  const visibleClasses = isSchoolLeader && grade
    ? classes.filter((row) => row.year_level_code === grade)
    : classes;

  useEffect(() => {
    if (isSchoolLeader && !grades.some((row) => row.code === grade)) setGrade(grades[0]?.code || '');
  }, [grades, grade, isSchoolLeader]);
  useEffect(() => {
    if (!visibleClasses.some((row) => row.code === klass)) setKlass(visibleClasses[0]?.code || '');
  }, [visibleClasses, klass]);
  useEffect(() => {
    if (!terms.some((row) => row.code === term)) setTerm(terms[0]?.code || '');
  }, [options.value, state.year]);

  const chosen = classes.find((row) => row.code === klass);
  const mayEdit = !!chosen && Store.canIn('timetable.write', {
    school: state.school, yearLevel: chosen.year_level_code, classSection: chosen.code
  });
  const mayConfigureSchoolDay = (state.profile?.roles || []).some(
    (role) => role.role_code === 'school_manager'
  );
  const schoolPeriods = useQuery(
    () => api.timetablePeriods(state.school),
    [state.school],
    !!state.school && mayConfigureSchoolDay
  );
  const subjects = useQuery(
    () => api.subjects(state.year, false, chosen.year_level_code),
    [state.year, chosen?.year_level_code],
    !!state.year && !!chosen && mayEdit
  );
  const week = useQuery(
    () => api.timetableWeek(state.year, klass, term),
    [state.year, klass, term],
    !!state.year && !!klass && !!term
  );
  const plan = week.value;
  const sourceTerm = useMemo(() => terms
    .filter((item) => item.code !== term && Number(item.sequence || 0) < Number(terms.find((row) => row.code === term)?.sequence || 0))
    .sort((left, right) => Number(right.sequence || 0) - Number(left.sequence || 0))[0], [terms, term]);
  const sourceWeek = useQuery(
    () => api.timetableWeek(state.year, klass, sourceTerm.code),
    [state.year, klass, sourceTerm?.code],
    !!state.year && !!klass && !!sourceTerm && mayEdit
  );

  useEffect(() => {
    if (!plan) return;
    const loaded = copyEntries(plan.entries);
    setSavedEntries(loaded);
    setDraftEntries(copyEntries(loaded));
    setActionError(null);
    setSaved(false);
    setDragged(null);
    const breakSlot = plan.periods.find((period) => !period.is_teaching);
    setBreakPeriod(String(breakSlot?.period_number || ''));
    if (breakSlot?.starts_at && breakSlot?.ends_at) {
      const [startHour, startMinute] = String(breakSlot.starts_at).slice(0, 5).split(':').map(Number);
      const [endHour, endMinute] = String(breakSlot.ends_at).slice(0, 5).split(':').map(Number);
      setBreakDuration(String(endHour * 60 + endMinute - startHour * 60 - startMinute));
    }
  }, [plan]);
  useEffect(() => {
    if (schoolPeriods.value) {
      const grid = schoolPeriods.value;
      setPeriodCount(String(grid.length));
      setDayStartsAt(grid[0]?.starts_at ? String(grid[0].starts_at).slice(0, 5) : '');
      setDayEndsAt(grid.at(-1)?.ends_at ? String(grid.at(-1).ends_at).slice(0, 5) : '');
    }
  }, [schoolPeriods.value]);

  /* Term 2 starts as a real copy, not a read-time illusion. It is written only once, only
     when a supervisor opens an empty later term, and never overwrites work already saved there. */
  useEffect(() => {
    const key = `${state.year}:${klass}:${sourceTerm?.code || ''}:${term}`;
    if (!mayEdit || !plan || !sourceTerm || !term || plan.entries.length ||
      !sourceWeek.value?.entries?.length || automaticCopyAttempt.current === key) return;
    automaticCopyAttempt.current = key;
    setCopying(true);
    setActionError(null);
    api.copyTimetableTerm(state.year, klass, sourceTerm.code, term)
      .then(() => week.reload())
      .catch((error) => {
        // Another supervisor may have initialized it first; reload and show only real failures.
        if (error?.status !== 409) setActionError(error);
        return week.reload();
      })
      .finally(() => setCopying(false));
  }, [state.year, klass, term, mayEdit, plan, sourceTerm?.code, sourceWeek.value, week]);

  const changed = useMemo(
    () => changedSlots(savedEntries, draftEntries),
    [savedEntries, draftEntries]
  );
  const hasChanges = changed.length > 0;
  const editable = mayEdit && !saving;
  const entryAt = (day, period) => draftEntries.find(
    (entry) => entry.day_of_week === day && entry.period_number === period
  );
  const visibleSubjects = [
    ...(subjects.value || []),
    ...(((options.value && options.value.teachingAssignments) || [])
      .filter((item) => item.class_code === klass)
      .map((item) => ({ code: item.subject_code, name_en: item.subject_name_en, name_ar: item.subject_name_ar })))
  ];
  const subjectMap = new Map(visibleSubjects.map((item) => [item.code, item]));

  const setEntry = (entries, day, period, subjectCode) => {
    const withoutSlot = entries.filter(
      (entry) => entry.day_of_week !== day || entry.period_number !== period
    );
    if (subjectCode === undefined) return withoutSlot;
    return [...withoutSlot, {
      academic_year_code: state.year,
      class_code: klass,
      term_code: term,
      day_of_week: day,
      period_number: period,
      subject_code: subjectCode,
      teacher_staff_number: null
    }];
  };

  const place = (day, period, payload) => {
    if (!editable || !payload) return;
    setDraftEntries((current) => {
      if (payload.kind === 'subject') {
        return setEntry(current, day, period, payload.subject);
      }
      if (payload.kind !== 'slot' || (payload.day === day && payload.period === period)) {
        return current;
      }
      const target = current.find(
        (entry) => entry.day_of_week === day && entry.period_number === period
      );
      let next = setEntry(current, day, period, payload.subject);
      next = setEntry(next, payload.day, payload.period, target?.subject_code);
      return next;
    });
    setActionError(null);
    setSaved(false);
    setDragged(null);
  };

  const clear = (day, period) => {
    if (!editable) return;
    setDraftEntries((current) => setEntry(current, day, period, undefined));
    setActionError(null);
    setSaved(false);
    setDragged(null);
  };

  const discard = () => {
    setDraftEntries(copyEntries(savedEntries));
    setActionError(null);
    setSaved(false);
    setDragged(null);
  };

  const save = async () => {
    if (!mayEdit || !hasChanges || saving) return;
    const draft = copyEntries(draftEntries);
    const oldBySlot = new Map(savedEntries.map((entry) => [
      slotKey(entry.day_of_week, entry.period_number), entry
    ]));
    const nextBySlot = new Map(draft.map((entry) => [
      slotKey(entry.day_of_week, entry.period_number), entry
    ]));
    const entries = changed.filter((key) => nextBySlot.has(key)).map((key) => {
      const entry = nextBySlot.get(key);
      return {
        class_code: klass,
        term_code: term,
        day_of_week: entry.day_of_week,
        period_number: entry.period_number,
        subject_code: entry.subject_code
      };
    });
    const clearSlots = changed.filter(
      (key) => oldBySlot.has(key) && !nextBySlot.has(key)
    ).map((key) => {
      const entry = oldBySlot.get(key);
      return {
        class_code: klass,
        term_code: term,
        day_of_week: entry.day_of_week,
        period_number: entry.period_number
      };
    });

    setSaving(true);
    setActionError(null);
    setSaved(false);
    try {
      await api.saveTimetableChanges(state.year, entries, clearSlots);
      setSavedEntries(copyEntries(draft));
      setSaved(true);
    } catch (error) {
      setActionError(error);
    } finally {
      setSaving(false);
    }
  };

  const saveDayLayout = async () => {
    if (!mayEdit || !plan || savingDayLayout) return;
    const requestedCount = Number(periodCount);
    if (mayConfigureSchoolDay && (!Number.isInteger(requestedCount) || requestedCount < 1 || requestedCount > 20)) {
      setActionError(new Error(t('Enter a number from 1 to 20.')));
      return;
    }
    if (mayConfigureSchoolDay && ((dayStartsAt && !dayEndsAt) || (!dayStartsAt && dayEndsAt) ||
      (dayStartsAt && dayEndsAt && dayStartsAt >= dayEndsAt))) {
      setActionError(new Error(t('Enter both day times, with the end after the start.')));
      return;
    }
    const breakMinutes = Number(breakDuration || 0);
    if (!Number.isInteger(breakMinutes) || breakMinutes < 0 || breakMinutes > 180) {
      setActionError(new Error(t('Enter a break duration from 0 to 180 minutes.')));
      return;
    }
    setSavingDayLayout(true);
    setActionError(null);
    try {
      const grid = schoolPeriods.value || [];
      const selectedBreak = breakPeriod && Number(breakPeriod) <= requestedCount
        ? Number(breakPeriod) : null;
      const timeChanged = dayStartsAt && dayEndsAt && (
        String(grid[0]?.starts_at || '').slice(0, 5) !== dayStartsAt ||
        String(grid.at(-1)?.ends_at || '').slice(0, 5) !== dayEndsAt
      );
      if (mayConfigureSchoolDay && (requestedCount !== grid.length || timeChanged)) {
        const periods = Array.from({ length: requestedCount }, (_, index) => {
          const existing = grid[index];
          return existing || {
            period_number: index + 1, name_en: '', name_ar: '',
            starts_at: null, ends_at: null, is_teaching: true
          };
        });
        if (dayStartsAt && dayEndsAt) {
          const [startHour, startMinute] = dayStartsAt.split(':').map(Number);
          const [endHour, endMinute] = dayEndsAt.split(':').map(Number);
          const start = startHour * 60 + startMinute;
          const end = endHour * 60 + endMinute;
          const toTime = (minutes) => `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${String(minutes % 60).padStart(2, '0')}:00`;
          const breakIndex = selectedBreak ? selectedBreak - 1 : -1;
          const lessonCount = periods.length - (breakIndex >= 0 ? 1 : 0);
          const lessonMinutes = (end - start - (breakIndex >= 0 ? breakMinutes : 0)) / lessonCount;
          if (lessonMinutes < 1) throw new Error(t('The day is too short for this number of periods and break duration.'));
          let cursor = start;
          for (let index = 0; index < periods.length; index += 1) {
            const duration = index === breakIndex ? breakMinutes : lessonMinutes;
            periods[index] = {
              ...periods[index],
              starts_at: toTime(Math.round(cursor)),
              ends_at: toTime(Math.round(cursor + duration))
            };
            cursor += duration;
          }
        }
        await api.setTimetablePeriods(state.school, periods);
        await schoolPeriods.reload();
      }
      await api.setGradeBreak(state.year, chosen.year_level_code, selectedBreak, breakMinutes);
      await week.reload();
    } catch (error) {
      setActionError(error);
    } finally {
      setSavingDayLayout(false);
    }
  };

  return <>
    <PageHead title={t('Timetable')}
      lede={t('Choose a class to view its weekly timetable. Supervisors can drag subjects into lessons and swap existing lessons.')} />
    <div className="vstack gap-3">
      <Card title={t('Class and term')}>
        <div className="row g-3">
          {isSchoolLeader ? <div className="col-12 col-md-4"><label className="form-label">{t('Grade')}</label>
            <Select value={grade} onChange={setGrade} disabled={!grades.length || saving || hasChanges}
              options={grades.map((row) => ({ value: row.code, label: pickName(row, state.lang) || row.code }))} />
          </div> : null}
          <div className={`col-12 ${isSchoolLeader ? 'col-md-4' : 'col-md-6'}`}><label className="form-label">{t('Class')}</label>
            <Select value={klass} onChange={setKlass}
              disabled={!visibleClasses.length || saving || hasChanges}
              options={visibleClasses.map((row) => ({ value: row.code,
                label: `${pickName(row, state.lang) || row.code} · ${row.code}` }))} />
          </div>
          <div className={`col-12 ${isSchoolLeader ? 'col-md-4' : 'col-md-6'}`}><label className="form-label">{t('Term')}</label>
            <Select value={term} onChange={setTerm}
              disabled={!terms.length || saving || hasChanges}
              options={terms.map((row) => ({ value: row.code, label: pickName(row, state.lang) || row.code }))} />
          </div>
        </div>
        {chosen ? <div className="mt-3 d-flex flex-wrap gap-2 align-items-center"><Badge tone={mayEdit ? 'info' : undefined}>
          {mayEdit ? t('Editable timetable') : t('View only')}
        </Badge>{copying ? <span className="small text-body-secondary">{t('Copying...')}</span> : null}</div> : null}
      </Card>

      {options.loading && !options.ready ? <Card><Skeleton rows={4} /></Card> : null}
      {options.error ? <ErrorNote error={options.error} onRetry={options.reload} /> : null}
      {options.ready && !classes.length ? <Card><Empty title={t('No classes are assigned to this account.')}>
        {t('Teachers only see classes assigned to them. Supervisors see classes in their managed grade.')}
      </Empty></Card> : null}

      {chosen && mayEdit && plan ? <Card title={t('Day layout')}
        subtitle={t('Choose the break period once. It applies only to this grade’s classes; lessons move with it automatically.')}>
        <div className="row g-3 align-items-start">
          {mayConfigureSchoolDay ? <div className="col-12 col-lg"><label className="form-label" htmlFor="timetable-period-count">{t('Periods per day')}</label>
            <input id="timetable-period-count" className="form-control" type="number" min="1" max="20" value={periodCount}
              onChange={(event) => setPeriodCount(event.target.value)} disabled={savingDayLayout || saving || hasChanges}
              aria-describedby="timetable-period-count-help" />
          </div> : null}
          <div className={`col-12 ${mayConfigureSchoolDay ? 'col-lg' : 'col-md-6'}`}><label className="form-label">{t('Break period')}</label>
            <Select value={breakPeriod} onChange={setBreakPeriod} disabled={savingDayLayout || saving || hasChanges}
              options={[{ value: '', label: t('No break') }, ...plan.periods.map((period) => ({
                value: String(period.period_number),
                label: `${t('Period')} ${period.period_number}`
              }))]} />
          </div>
          <div className={`col-12 ${mayConfigureSchoolDay ? 'col-lg' : 'col-md-6'}`}><label className="form-label" htmlFor="timetable-break-duration">{t('Break duration (minutes)')}</label>
            <input id="timetable-break-duration" className="form-control" type="number" min="0" max="180" value={breakDuration}
              onChange={(event) => setBreakDuration(event.target.value)} disabled={savingDayLayout || saving || hasChanges || !breakPeriod} />
          </div>
          {mayConfigureSchoolDay ? <><div className="col-12 col-lg"><label className="form-label" htmlFor="timetable-day-start">{t('Day starts at')}</label>
            <input id="timetable-day-start" className="form-control" type="time" value={dayStartsAt}
              onChange={(event) => setDayStartsAt(event.target.value)} disabled={savingDayLayout || saving || hasChanges} />
          </div>
          <div className="col-12 col-lg"><label className="form-label" htmlFor="timetable-day-end">{t('Day ends at')}</label>
            <input id="timetable-day-end" className="form-control" type="time" value={dayEndsAt}
              onChange={(event) => setDayEndsAt(event.target.value)} disabled={savingDayLayout || saving || hasChanges} />
          </div></> : null}
          {mayConfigureSchoolDay ? <div className="col-12"><div id="timetable-period-count-help" className="form-text">{t('Reducing the number deletes lessons in the removed periods.')} {t('Times are divided equally across the day.')}</div></div> : null}
          <div className="col-12"><Button variant="secondary" pending={savingDayLayout}
            pendingLabel={t('Saving…')} disabled={saving || hasChanges || savingDayLayout}
            onClick={saveDayLayout}>{t('Save day layout')}</Button></div>
        </div>
      </Card> : null}

      {chosen && mayEdit ? <Card title={t('Subjects')}
        subtitle={t('Drag a subject onto any lesson. Drag one lesson onto another to swap them.')}>
        {subjects.loading ? <Skeleton rows={2} /> : <div className="sis-subject-tray">
          {(subjects.value || []).map((subject) => <button type="button" key={subject.code}
            className={`sis-subject-chip ${dragged?.kind === 'subject' && dragged.subject === subject.code ? 'active' : ''}`}
            draggable={editable} disabled={saving}
            onClick={() => setDragged({ kind: 'subject', subject: subject.code })}
            onDragStart={() => setDragged({ kind: 'subject', subject: subject.code })}>
            <span>{pickName(subject, state.lang) || subject.code}</span><small>{subject.code}</small>
          </button>)}
        </div>}
      </Card> : null}

      {chosen && term ? <Card title={t('Weekly timetable')} tight footer={mayEdit ? <>
        <div className="sis-pull small">
          {hasChanges ? t('{0} timetable change(s) not yet saved.', [changed.length]) :
            saved ? t('Timetable saved to the database.') : t('No unsaved timetable changes.')}
        </div>
        <Button variant="quiet" disabled={!hasChanges || saving} onClick={discard}>
          {t('Discard changes')}
        </Button>
        <Button variant="primary" pending={saving} pendingLabel={t('Saving…')}
          disabled={!hasChanges} onClick={save}>{t('Save timetable')}</Button>
      </> : null}>
        {week.loading && !week.ready ? <div className="p-3"><Skeleton rows={7} /></div> : null}
        {week.error ? <div className="p-3"><ErrorNote error={week.error} onRetry={week.reload} /></div> : null}
        {actionError ? <div className="p-3 pb-0"><div className="alert alert-danger mb-0" role="alert">
          {actionError.message || t('Could not save the timetable settings.')}
        </div></div> : null}
        {plan ? <div className="sis-timetable-scroll"><table className="sis-timetable-grid">
          <thead><tr><th>{t('Period')}</th>{plan.days.map((day) => <th key={day}>{t(DAY_LABELS[day] || day)}</th>)}</tr></thead>
          <tbody>{plan.periods.map((period) => <tr key={period.period_number}>
            <th><strong>{pickName(period, state.lang) || `${t('Period')} ${period.period_number}`}</strong>
              {period.is_timed ? <small>{String(period.starts_at).slice(0, 5)}–{String(period.ends_at).slice(0, 5)}</small> : null}</th>
            {plan.days.map((day) => {
              const entry = entryAt(day, period.period_number);
              const subject = entry && subjectMap.get(entry.subject_code);
              if (!period.is_teaching) return <td className="sis-timetable-break" key={day}>{pickName(period, state.lang) || t('Break')}</td>;
              return <td key={day} className={`sis-timetable-slot ${dragged && editable ? 'is-drop-ready' : ''}`}
                onClick={() => dragged?.kind === 'subject' && place(day, period.period_number, dragged)}
                onDragOver={(event) => editable && event.preventDefault()}
                onDrop={() => place(day, period.period_number, dragged)}>
                {entry?.subject_code ? <div className="sis-lesson" draggable={editable}
                  onDragStart={() => setDragged({ kind: 'slot', day, period: period.period_number, subject: entry.subject_code })}>
                  <strong>{pickName(subject, state.lang) || entry.subject_code}</strong><small>{entry.subject_code}</small>
                  {mayEdit ? <Button size="sm" variant="quiet" title={t('Clear lesson')} disabled={saving}
                    onClick={(event) => { event.stopPropagation(); clear(day, period.period_number); }}>×</Button> : null}
                </div> : <span className="sis-empty-slot">{mayEdit ? t('Drop subject here') : '—'}</span>}
              </td>;
            })}
          </tr>)}</tbody>
        </table></div> : null}
      </Card> : null}
    </div>
  </>;
}
