import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { Button, Card, ErrorNote, PageHead, Select, Skeleton, Table } from '../components/Ui.jsx';
import { t } from '../i18n.js';

export function GradeAssignments() {
  const state = useStore();
  const grants = ((state.profile && state.profile.grants) || []).filter(
    (grant) => grant.permission === 'teachers.assign_classes' && grant.scope_type === 'year_level'
  );
  const grades = [...new Set(grants.map((grant) => grant.scope_code).filter(Boolean))];
  const [grade, setGrade] = useState(grades[0] || '');
  const [subject, setSubject] = useState('');
  const [teacher, setTeacher] = useState('');
  const [classes, setClasses] = useState([]);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const options = useQuery(
    () => api.gradeAssignmentOptions(state.school, grade, state.year, subject),
    [state.school, state.year, grade, subject],
    !!state.school && !!state.year && !!grade
  );
  const value = options.value || { subjects: [], classes: [], eligible_teachers: [] };
  const offeredClasses = subject
    ? (value.available_classes || []).concat(
        value.classes.filter((row) => classes.includes(row.code))
      ).filter((row, index, rows) => rows.findIndex((item) => item.code === row.code) === index)
    : [];
  /* The grade's whole teaching staff, which is a different question from the eligible
     teachers above: that list is one subject's candidates, this one is who works on the
     grade at all. It is read separately because it does not change when the subject does. */
  const staff = useQuery(
    () => api.gradeTeachers(state.school, grade),
    [state.school, grade],
    !!state.school && !!grade
  );

  useEffect(() => { setSubject(''); setTeacher(''); setClasses([]); }, [grade, state.year]);
  useEffect(() => {
    const row = value.eligible_teachers.find((item) => item.staff_number === teacher);
    setClasses(row ? row.assigned_class_codes : []);
  }, [teacher, options.value]);

  const assign = () => {
    setSaving(true); setSaveError(null);
    api.assignTeacherClasses(state.school, grade, {
      academic_year_code: state.year, subject_code: subject,
      staff_number: teacher, class_codes: classes
    }).then(() => options.reload(), setSaveError).finally(() => setSaving(false));
  };

  return <>
    <PageHead title={t('Class assignments')}
      lede={t('Choose a managed grade, subject, eligible teacher, and one or more classes.')} />
    {!grades.length ? <Card>{t('No managed grades are assigned to this account.')}</Card> :
      <div className="vstack gap-3">
        <Card title={t('1. Grade')}>
          <Select value={grade} onChange={setGrade}
            options={grades.map((code) => ({ value: code, label: code }))} />
        </Card>
        {options.loading && !options.ready ? <Card><Skeleton rows={4} /></Card> : null}
        {options.error ? <ErrorNote error={options.error} onRetry={options.reload} /> : null}
        {options.ready ? <>
          <Card title={t('2. Subject')}>
            <Select value={subject} onChange={(next) => { setSubject(next); setTeacher(''); }}
              options={[{ value: '', label: t('Choose…') }, ...value.subjects.map((row) => ({
                value: row.code, label: pickName(row, state.lang) || row.code
              }))]} />
          </Card>
          <Card title={t('3. Eligible teacher')}>
            <Select value={teacher} disabled={!subject} onChange={setTeacher}
              options={[{ value: '', label: t('Choose…') }, ...value.eligible_teachers.map((row) => ({
                value: row.staff_number, label: pickName(row, state.lang) || row.staff_number
              }))]} />
          </Card>
          <Card title={t('4. Classes')}>
            <div className="d-flex flex-wrap gap-3">
              {offeredClasses.map((row) => <label className="form-check" key={row.code}>
                <input className="form-check-input" type="checkbox" checked={classes.includes(row.code)}
                  disabled={!teacher} onChange={(event) => setClasses(event.target.checked
                    ? [...classes, row.code] : classes.filter((code) => code !== row.code))} />
                <span className="form-check-label">{pickName(row, state.lang) || row.code}</span>
              </label>)}
            </div>
            {saveError ? <ErrorNote error={saveError} /> : null}
            <Button className="mt-3" variant="primary" pending={saving} disabled={!teacher || !classes.length}
              onClick={assign}>{t('Assign teacher')}</Button>
          </Card>
        </> : null}
        <Card title={t('Teaching staff on this grade')}
          subtitle={t('Read-only. Each teacher is shown as they stand on this grade alone.')}>
          {staff.error ? <ErrorNote error={staff.error} onRetry={staff.reload} /> :
            <Table loading={staff.loading} rows={staff.value || []}
              rowKey={(row) => row.staff_number}
              columns={[
                { key: 'name', header: t('Teacher'),
                  cell: (row) => pickName(row, state.lang) || row.staff_number },
                { key: 'staff_number', header: t('Staff number'),
                  cell: (row) => row.staff_number },
                { key: 'subjects', header: t('Subjects'),
                  cell: (row) => row.assignments.map((item) => item.subject_code).join(', ') },
                { key: 'classes', header: t('Assigned classes'),
                  cell: (row) => row.assignments.flatMap((item) => item.class_codes).join(', ') }
              ]} />}
        </Card>
      </div>}
  </>;
}
