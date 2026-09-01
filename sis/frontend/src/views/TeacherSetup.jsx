import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { Store } from '../store.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { Button, Card, Empty, ErrorNote, Field, Input, NoYearNotice, PageHead, Select, Skeleton, Table } from '../components/Ui.jsx';
import { t } from '../i18n.js';

const blank = () => ({
  staff_number: '', full_name_en: '', full_name_ar: '', email: '', phone: '',
  username: '', password: '', is_active: true, assignments: []
});

export function TeacherSetup() {
  const state = useStore();
  const [form, setForm] = useState(blank());
  const [pair, setPair] = useState('');
  const [chosenSubject, setChosenSubject] = useState('');
  const [chosenStage, setChosenStage] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const data = useQuery(
    () => Promise.all([api.teachers(state.school), api.subjectAssignments(state.year)]),
    [state.school, state.year],
    !!state.school && !!state.year
  );
  const [teachers = [], board = []] = data.value || [];
  const levels = useQuery(() => api.schoolLevels(state.school), [state.school], !!state.school);
  const eligible = useMemo(() => board.flatMap((grade) => grade.subjects.map((subject) => ({
    key: `${grade.year_level_code}\u0000${subject.code}`,
    academic_year_code: state.year,
    year_level_code: grade.year_level_code,
    track_code: grade.track_code,
    subject_code: subject.code,
    subject_name_en: subject.name_en,
    subject_name_ar: subject.name_ar
  }))), [board, state.year]);
  const levelByCode = Object.fromEntries((levels.value || []).map((row) => [row.code, row]));
  const subjects = [...new Map(eligible.map((row) => [row.subject_code, row])).values()];
  const stages = [...new Set(eligible.filter((row) => row.subject_code === chosenSubject)
    .map((row) => (levelByCode[row.year_level_code] || {}).stage).filter(Boolean))];
  const filteredGrades = eligible.filter((row) => row.subject_code === chosenSubject &&
    (levelByCode[row.year_level_code] || {}).stage === chosenStage);
  const duplicateStaffNumber = teachers.some((row) =>
    row.staff_number.toLowerCase() === form.staff_number.trim().toLowerCase()
  );

  useEffect(() => { setForm(blank()); setPair(''); }, [state.school, state.year]);

  const addAssignment = () => {
    const candidate = eligible.find((item) => item.key === pair);
    if (!candidate) return;
    const exists = form.assignments.some((item) =>
      item.year_level_code === candidate.year_level_code &&
      item.subject_code === candidate.subject_code
    );
    if (!exists) setForm((old) => ({
      ...old,
      assignments: [...old.assignments, { ...candidate, class_codes: [] }]
    }));
    setPair('');
  };

  const save = async () => {
    setSaving(true); setError(null);
    try {
      await api.saveTeacher(state.school, form.staff_number.trim(), {
        full_name_en: form.full_name_en,
        full_name_ar: form.full_name_ar,
        email: form.email,
        phone: form.phone,
        is_active: form.is_active,
        username: form.username.trim() || null,
        password: form.password || null,
        assignments: form.assignments.map((item) => ({
          academic_year_code: item.academic_year_code,
          subject_code: item.subject_code,
          year_level_code: item.year_level_code,
          // Existing class work survives a manager editing eligibility. New eligibility
          // starts empty; the grade supervisor owns the class decision.
          class_codes: item.class_codes || []
        }))
      });
      Store.toast(t('Teacher configuration saved.'), 'success');
      data.reload();
    } catch (reason) { setError(reason); }
    finally { setSaving(false); }
  };

  if (!state.year) return <><PageHead title={t('Teacher setup')} /><NoYearNotice /></>;
  if (data.loading && !data.ready) return <><PageHead title={t('Teacher setup')} /><Card><Skeleton rows={6} /></Card></>;

  return <>
    <PageHead title={t('Create teacher')}
      lede={t('Define the teacher account, subjects, eligible grades, and track scope. Grade supervisors assign classes afterward.')} />
    {data.error ? <ErrorNote error={data.error} onRetry={data.reload} /> : null}
    <div className="vstack gap-3">
      <Card title={t('New teacher account')}>
        <div className="row g-3 mt-1">
          <Field className="col-12 col-md-4" label={t('Staff number')} required
            hint={duplicateStaffNumber ? t('This staff number already belongs to another teacher.') : null}>
            <Input value={form.staff_number}
              onInput={(value) => setForm((old) => ({ ...old, staff_number: value }))} />
          </Field>
          <Field className="col-12 col-md-4" label={t('English name')}>
            <Input value={form.full_name_en} onInput={(value) => setForm((old) => ({ ...old, full_name_en: value }))} />
          </Field>
          <Field className="col-12 col-md-4" label={t('Arabic name')}>
            <Input value={form.full_name_ar} onInput={(value) => setForm((old) => ({ ...old, full_name_ar: value }))} />
          </Field>
          <Field className="col-12 col-md-6" label={t('Username')}>
            <Input value={form.username} onInput={(value) => setForm((old) => ({ ...old, username: value }))} />
          </Field>
          <Field className="col-12 col-md-6" label={t('Password')}
            hint={t('Required for a new account; leave blank to keep an existing password.')}>
            <Input type="password" value={form.password} onInput={(value) => setForm((old) => ({ ...old, password: value }))} />
          </Field>
        </div>
      </Card>

      <Card title={t('Subject, grade, and track eligibility')}>
        <div className="d-flex flex-column flex-md-row gap-2">
          <Select value={chosenSubject}
            options={[{ value: '', label: t('Choose subject') }, ...subjects.map((item) => ({
              value: item.subject_code,
              label: (state.lang === 'ar' ? item.subject_name_ar : item.subject_name_en) || item.subject_code
            }))]} onChange={(value) => { setChosenSubject(value); setChosenStage(''); setPair(''); }} />
          <Select value={chosenStage} disabled={!chosenSubject}
            options={[{ value: '', label: t('Choose stage') }, ...stages.map((stage) => ({ value: stage, label: stage }))]}
            onChange={(value) => { setChosenStage(value); setPair(''); }} />
          <Select className="flex-grow-1" value={pair} disabled={!chosenStage}
            options={[{ value: '', label: t('Choose grade') }, ...filteredGrades.map((item) => ({
              value: item.key,
              label: pickName(levelByCode[item.year_level_code] || {}, state.lang) || item.year_level_code
            }))]} onChange={setPair} />
          <Button onClick={addAssignment} disabled={!pair}>{t('Add eligibility')}</Button>
        </div>
        <Table rows={form.assignments} rowKey={(row) => `${row.year_level_code}:${row.subject_code}`}
          empty={<Empty title={t('No eligible grades yet')} />}
          columns={[
            { key: 'subject', header: t('Subject'), cell: (row) => row.subject_code },
            { key: 'grade', header: t('Grade'), cell: (row) => row.year_level_code },
            { key: 'track', header: t('Track'), cell: (row) => row.track_code || '—' },
            { key: 'remove', header: '', cell: (row) => <Button size="sm" variant="danger"
              onClick={() => setForm((old) => ({ ...old, assignments: old.assignments.filter((item) =>
                item.year_level_code !== row.year_level_code || item.subject_code !== row.subject_code) }))}>
              {t('Remove')}</Button> }
          ]} />
      </Card>
      {error ? <ErrorNote error={error} /> : null}
      <div><Button variant="primary" pending={saving}
        disabled={duplicateStaffNumber || !form.staff_number.trim() || (!form.full_name_en.trim() && !form.full_name_ar.trim())}
        onClick={save}>{t('Save teacher configuration')}</Button></div>
    </div>
  </>;
}
