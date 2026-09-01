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

export function Timetable() {
  const state = useStore();
  const [klass, setKlass] = useState('');
  const [term, setTerm] = useState('');
  const [dragged, setDragged] = useState(null);
  const [busySlot, setBusySlot] = useState('');
  const [actionError, setActionError] = useState(null);

  const roles = Store.roles();
  const isSupervisor = roles.includes('year_supervisor');
  const supervisedGrades = useMemo(() => [...new Set(
    ((state.profile && state.profile.grants) || [])
      .filter((grant) => grant.permission === 'timetable.read' && grant.scope_type === 'year_level')
      .map((grant) => grant.scope_code).filter(Boolean)
  )], [state.profile]);

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
    const supervised = isSupervisor
      ? (await Promise.all(supervisedGrades.map((grade) => api.classes(state.year, grade)))).flat()
      : [];
    const classes = [...supervised, ...teacherClasses].filter(
      (row, index, rows) => rows.findIndex((item) => item.code === row.code) === index
    );
    const terms = await api.terms(state.year);
    return { classes, terms, teachingAssignments: teaching.assignments || [] };
  }, [state.year, isSupervisor, supervisedGrades.join('|')], !!state.year);

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
  const entries = (plan && plan.entries) || [];
  const entryAt = (day, period) => entries.find(
    (entry) => entry.day_of_week === day && entry.period_number === period
  );
  const visibleSubjects = [
    ...(subjects.value || []),
    ...(((options.value && options.value.teachingAssignments) || [])
      .filter((item) => item.class_code === klass)
      .map((item) => ({ code: item.subject_code, name_en: item.subject_name_en, name_ar: item.subject_name_ar })))
  ];
  const subjectMap = new Map(visibleSubjects.map((item) => [item.code, item]));

  const place = async (day, period, payload) => {
    if (!mayEdit || !payload) return;
    const target = entryAt(day, period);
    const changes = [];
    if (payload.kind === 'subject') {
      changes.push({ class_code: klass, term_code: term, day_of_week: day,
        period_number: period, subject_code: payload.subject });
    } else if (payload.kind === 'slot') {
      if (payload.day === day && payload.period === period) return;
      changes.push({ class_code: klass, term_code: term, day_of_week: day,
        period_number: period, subject_code: payload.subject });
      changes.push({ class_code: klass, term_code: term, day_of_week: payload.day,
        period_number: payload.period, subject_code: target?.subject_code || null });
    }
    setBusySlot(slotKey(day, period)); setActionError(null);
    try {
      await api.placeTimetableLessons(state.year, changes);
      week.reload();
    } catch (error) { setActionError(error); }
    finally { setBusySlot(''); setDragged(null); }
  };

  const clear = async (day, period) => {
    setBusySlot(slotKey(day, period)); setActionError(null);
    try {
      await api.clearTimetableSlots(state.year, [{ class_code: klass, term_code: term,
        day_of_week: day, period_number: period }]);
      week.reload();
    } catch (error) { setActionError(error); }
    finally { setBusySlot(''); }
  };

  return <>
    <PageHead title={t('Timetable')}
      lede={t('Choose a class to view its weekly timetable. Supervisors can drag subjects into lessons and swap existing lessons.')} />
    <div className="vstack gap-3">
      <Card title={t('Class and term')}>
        <div className="row g-3">
          <div className="col-12 col-md-6"><label className="form-label">{t('Class')}</label>
            <Select value={klass} onChange={setKlass} disabled={!classes.length}
              options={classes.map((row) => ({ value: row.code,
                label: `${pickName(row, state.lang) || row.code} · ${row.code}` }))} />
          </div>
          <div className="col-12 col-md-6"><label className="form-label">{t('Term')}</label>
            <Select value={term} onChange={setTerm} disabled={!terms.length}
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
            className={`sis-subject-chip ${dragged?.kind === 'subject' && dragged.subject === subject.code ? 'active' : ''}`} draggable
            onClick={() => setDragged({ kind: 'subject', subject: subject.code })}
            onDragStart={() => setDragged({ kind: 'subject', subject: subject.code })}>
            <span>{pickName(subject, state.lang) || subject.code}</span><small>{subject.code}</small>
          </button>)}
        </div>}
      </Card> : null}

      {chosen && term ? <Card title={t('Weekly timetable')} tight>
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
              const key = slotKey(day, period.period_number);
              if (!period.is_teaching) return <td className="sis-timetable-break" key={day}>{pickName(period, state.lang) || t('Break')}</td>;
              return <td key={day} className={`sis-timetable-slot ${dragged && mayEdit ? 'is-drop-ready' : ''}`}
                onClick={() => dragged?.kind === 'subject' && place(day, period.period_number, dragged)}
                onDragOver={(event) => mayEdit && event.preventDefault()}
                onDrop={() => place(day, period.period_number, dragged)}>
                {busySlot === key ? <span className="spinner-border spinner-border-sm" /> : entry?.subject_code ?
                  <div className="sis-lesson" draggable={mayEdit}
                    onDragStart={() => setDragged({ kind: 'slot', day, period: period.period_number, subject: entry.subject_code })}>
                    <strong>{pickName(subject, state.lang) || entry.subject_code}</strong><small>{entry.subject_code}</small>
                    {mayEdit ? <Button size="sm" variant="quiet" title={t('Clear lesson')} onClick={() => clear(day, period.period_number)}>×</Button> : null}
                  </div> : <span className="sis-empty-slot">{mayEdit ? t('Drop subject here') : '—'}</span>}
              </td>;
            })}
          </tr>)}</tbody>
        </table></div> : null}
      </Card> : null}
    </div>
  </>;
}
