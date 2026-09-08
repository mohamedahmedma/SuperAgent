import { useEffect, useMemo, useState } from 'react';
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
  const [term, setTerm] = useState('');
  const [dragged, setDragged] = useState(null);
  const [savedEntries, setSavedEntries] = useState([]);
  const [draftEntries, setDraftEntries] = useState([]);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [actionError, setActionError] = useState(null);

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

  useEffect(() => {
    if (!classes.some((row) => row.code === klass)) setKlass(classes[0]?.code || '');
  }, [options.value, state.year]);
  useEffect(() => {
    if (!terms.some((row) => row.code === term)) setTerm(terms[0]?.code || '');
  }, [options.value, state.year]);

  const chosen = classes.find((row) => row.code === klass);
  const mayEdit = !!chosen && Store.canIn('timetable.write', {
    school: state.school, yearLevel: chosen.year_level_code, classSection: chosen.code
  });
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

  useEffect(() => {
    if (!plan) return;
    const loaded = copyEntries(plan.entries);
    setSavedEntries(loaded);
    setDraftEntries(copyEntries(loaded));
    setActionError(null);
    setSaved(false);
    setDragged(null);
  }, [plan]);

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

  return <>
    <PageHead title={t('Timetable')}
      lede={t('Choose a class to view its weekly timetable. Supervisors can drag subjects into lessons and swap existing lessons.')} />
    <div className="vstack gap-3">
      <Card title={t('Class and term')}>
        <div className="row g-3">
          <div className="col-12 col-md-6"><label className="form-label">{t('Class')}</label>
            <Select value={klass} onChange={setKlass}
              disabled={!classes.length || saving || hasChanges}
              options={classes.map((row) => ({ value: row.code,
                label: `${pickName(row, state.lang) || row.code} · ${row.code}` }))} />
          </div>
          <div className="col-12 col-md-6"><label className="form-label">{t('Term')}</label>
            <Select value={term} onChange={setTerm}
              disabled={!terms.length || saving || hasChanges}
              options={terms.map((row) => ({ value: row.code, label: pickName(row, state.lang) || row.code }))} />
          </div>
        </div>
        {chosen ? <div className="mt-3"><Badge tone={mayEdit ? 'info' : undefined}>
          {mayEdit ? t('Editable timetable') : t('View only')}
        </Badge></div> : null}
      </Card>

      {options.loading && !options.ready ? <Card><Skeleton rows={4} /></Card> : null}
      {options.error ? <ErrorNote error={options.error} onRetry={options.reload} /> : null}
      {options.ready && !classes.length ? <Card><Empty title={t('No classes are assigned to this account.')}>
        {t('Teachers only see classes assigned to them. Supervisors see classes in their managed grade.')}
      </Empty></Card> : null}

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
        {actionError ? <div className="p-3 pb-0"><ErrorNote error={actionError} /></div> : null}
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
