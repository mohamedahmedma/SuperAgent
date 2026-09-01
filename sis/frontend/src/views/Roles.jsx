import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { Badge, Card, Empty, ErrorNote, PageHead, Select, Skeleton } from '../components/Ui.jsx';
import { t } from '../i18n.js';

export function Roles() {
  const state = useStore();
  const [grade, setGrade] = useState('');
  const [subject, setSubject] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState(null);
  const data = useQuery(() => Promise.all([
    api.teachers(state.school),
    api.rbacUsers(),
    api.rbacYearLevels(state.school),
    api.subjectAssignments(state.year)
  ]), [state.school, state.year], !!state.school && !!state.year);

  const [teachers = [], users = [], yearLevels = [], subjectBoard = []] = data.value || [];
  const usersById = Object.fromEntries(users.map((user) => [user.id, user]));
  const selectedLevel = yearLevels.find((row) => String(row.id) === grade);
  const subjects = useMemo(() => {
    if (!selectedLevel) return [];
    const row = subjectBoard.find((item) => item.year_level_code === selectedLevel.code);
    return (row && row.subjects) || [];
  }, [grade, subjectBoard]);

  useEffect(() => {
    if (!grade && yearLevels.length) setGrade(String(yearLevels[0].id));
  }, [yearLevels.length]);
  useEffect(() => {
    if (subject && !subjects.some((row) => row.code === subject)) setSubject('');
  }, [grade, subjects.length]);

  const matchingTeachers = teachers.filter((teacher) => teacher.assignments.some((assignment) =>
    assignment.academic_year_code === state.year &&
    assignment.year_level_code === selectedLevel?.code &&
    (!subject || assignment.subject_code === subject)
  ));

  const toggleSupervisor = async (user, roleCode, checked) => {
    if (!grade) return;
    const grant = { role_code: roleCode, scope_type: 'year_level', scope_id: Number(grade) };
    setBusy(`${user.id}:${roleCode}`); setError(null);
    try {
      if (checked) await api.addUserRole(user.id, grant);
      else await api.removeUserRole(user.id, grant);
      data.reload();
    } catch (reason) { setError(reason); }
    finally { setBusy(''); }
  };

  if (data.loading && !data.ready) return <><PageHead title={t('Staff roles')} /><Card><Skeleton rows={7} /></Card></>;
  if (data.error) return <><PageHead title={t('Staff roles')} /><ErrorNote error={data.error} onRetry={data.reload} /></>;

  return <>
    <PageHead title={t('Staff roles')}
      lede={t('Choose a grade and subject to review its teachers. Supervisors are managed separately and may also be teachers.')} />
    {error ? <ErrorNote error={error} /> : null}
    <div className="vstack gap-3">
      <Card title={t('Teachers')} subtitle={t('Teachers come from their subject and class assignments.')}>
        <div className="row g-3 mb-3">
          <div className="col-12 col-md-6">
            <label className="form-label">{t('Grade')}</label>
            <Select value={grade} options={yearLevels.map((row) => ({
              value: String(row.id), label: pickName(row, state.lang) || row.code
            }))} onChange={setGrade} />
          </div>
          <div className="col-12 col-md-6">
            <label className="form-label">{t('Teacher subject')}</label>
            <Select value={subject} options={[
              { value: '', label: t('All subjects') },
              ...subjects.map((row) => ({ value: row.code, label: pickName(row, state.lang) || row.code }))
            ]} onChange={setSubject} />
          </div>
        </div>
        {matchingTeachers.length ? <div className="vstack gap-2">{matchingTeachers.map((teacher) => {
          const assignment = teacher.assignments.filter((row) =>
            row.academic_year_code === state.year && row.year_level_code === selectedLevel?.code &&
            (!subject || row.subject_code === subject)
          );
          return <div className="sis-row-open border rounded p-3" key={teacher.staff_number}>
            <div className="d-flex flex-wrap justify-content-between gap-2">
              <div><strong>{pickName(teacher, state.lang) || teacher.staff_number}</strong>
                <div className="small text-body-tertiary">{teacher.username || t('No login account')}</div></div>
              <div className="d-flex flex-wrap gap-1">{assignment.flatMap((row) => row.class_codes || []).map((code) =>
                <Badge key={code}>{code}</Badge>)}</div>
            </div>
          </div>;
        })}</div> : <Empty title={t('No teachers match this grade and subject.')} />}
      </Card>

      <Card title={t('Supervisors')}
        subtitle={t('A class supervisor may also remain an ordinary teacher.') }>
        <p className="small text-body-tertiary">{t('Supervisor scope')}: <strong>{pickName(selectedLevel, state.lang) || selectedLevel?.code}</strong></p>
        <div className="vstack gap-2">{teachers.map((teacher) => {
          const user = usersById[teacher.user_id];
          if (!user) return null;
          const holds = (role) => (user.roles || []).some((row) =>
            row.role_code === role && row.scope_type === 'year_level' && String(row.scope_id) === grade
          );
          return <div className="d-flex flex-column flex-sm-row align-items-sm-center gap-2 border rounded p-3" key={teacher.staff_number}>
            <div className="flex-grow-1"><strong>{pickName(teacher, state.lang) || teacher.staff_number}</strong>
              <div className="small text-body-tertiary">{user.username}</div></div>
            {[
              ['year_supervisor', 'Class Supervisor']
            ].map(([role, label]) => <label className="form-check form-switch mb-0" key={role}>
              <input className="form-check-input" type="checkbox" checked={holds(role)} disabled={!!busy || !grade}
                onChange={(event) => toggleSupervisor(user, role, event.target.checked)} />
              <span className="form-check-label">{t(label)}</span>
            </label>)}
          </div>;
        })}</div>
      </Card>
    </div>
  </>;
}
