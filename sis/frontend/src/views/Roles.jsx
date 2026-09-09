import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { Store } from '../store.js';
import { Badge, Button, Card, Empty, ErrorNote, PageHead, Select, Skeleton } from '../components/Ui.jsx';
import { t } from '../i18n.js';

const permissionResourceLabel = (resource) => resource
  .replace(/[._]/g, ' ')
  .replace(/\b\w/g, (character) => character.toUpperCase());

export function Roles() {
  const state = useStore();
  const [grade, setGrade] = useState('');
  const [subject, setSubject] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState(null);
  const [permissionRole, setPermissionRole] = useState('school_manager');
  const [matrix, setMatrix] = useState([]);
  const [overrideUserId, setOverrideUserId] = useState('');
  const [overrideRows, setOverrideRows] = useState([]);
  const isAdmin = Store.roles().includes('admin');
  const data = useQuery(() => Promise.all([
    api.teachers(state.school),
    api.rbacUsers(),
    api.rbacYearLevels(state.school),
    api.subjectAssignments(state.year)
  ]), [state.school, state.year], !!state.school && !!state.year);

  const [teachers = [], users = [], yearLevels = [], subjectBoard = []] = data.value || [];
  const permissions = useQuery(() => api.rolePermissionMatrix(permissionRole), [permissionRole], isAdmin);
  const overrideUsers = users.filter((user) => !(user.roles || []).some((role) => role.role_code === 'admin'));
  const userOverrides = useQuery(
    () => api.userPermissionOverrides(overrideUserId),
    [overrideUserId],
    isAdmin && !!overrideUserId
  );
  useEffect(() => { if (permissions.value) setMatrix(permissions.value.resources || []); }, [permissions.value]);
  useEffect(() => {
    if (!overrideUserId && overrideUsers.length) setOverrideUserId(String(overrideUsers[0].id));
  }, [overrideUserId, overrideUsers.length]);
  useEffect(() => { if (userOverrides.value) setOverrideRows(userOverrides.value.permissions || []); }, [userOverrides.value]);
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
      {isAdmin ? <Card title={t('Role permissions')} subtitle={t('Admin-only access policy. Admin permissions are permanent.') }>
        <div className="row g-3 align-items-end mb-3"><div className="col-12 col-md-5"><label className="form-label">{t('Role')}</label><Select value={permissionRole} options={[
          ['school_owner', t('School Owner')], ['school_manager', t('School Manager')], ['floor_supervisor', t('Floor Supervisor')], ['attendance_supervisor', t('Attendance Supervisor')], ['teacher', t('Teacher')]
        ].map(([value, label]) => ({ value, label }))} onChange={setPermissionRole} /></div><div className="col-12 col-md-7 small text-body-tertiary">{t('Choose No Access, Read Only, or Read + Write for each resource.')}</div></div>
        {permissions.error ? <ErrorNote error={permissions.error} onRetry={permissions.reload} /> : null}
        <div className="vstack gap-2">{matrix.map((row) => <div className="d-flex flex-wrap align-items-center justify-content-between gap-2 border rounded p-2" key={row.resource}><strong>{permissionResourceLabel(row.resource)}</strong><Select className="w-auto" value={row.mode} options={[
          { value: 'no_access', label: t('No Access') }, { value: 'read_only', label: t('Read Only') }, { value: 'read_write', label: t('Read + Write') }
        ]} onChange={(mode) => setMatrix((rows) => rows.map((item) => item.resource === row.resource ? { ...item, mode } : item))} /></div>)}</div>
        <div className="mt-3 d-flex justify-content-end"><Button pending={busy === 'permissions'} disabled={!!busy || !matrix.length} onClick={async () => { setBusy('permissions'); setError(null); try { const saved = await api.saveRolePermissionMatrix(permissionRole, matrix); setMatrix(saved.resources || []); Store.invalidate('roles:'); } catch (reason) { setError(reason); } finally { setBusy(''); } }}>{t('Save permissions')}</Button></div>
      </Card> : null}
      {isAdmin ? <Card title={t('Individual user overrides')} subtitle={t('Overrides apply only to this staff user. Default returns to inherited role access.')}>
        <div className="row g-3 align-items-end mb-3"><div className="col-12 col-md-6"><label className="form-label">{t('Staff user')}</label><Select value={overrideUserId} options={overrideUsers.map((user) => ({ value: String(user.id), label: user.full_name_en || user.full_name_ar || user.username }))} onChange={setOverrideUserId} /></div><div className="col-12 col-md-6 small text-body-tertiary">{userOverrides.value?.roles?.length ? <>{t('Roles')}: {userOverrides.value.roles.join(', ')}</> : t('No roles assigned')}</div></div>
        {userOverrides.error ? <ErrorNote error={userOverrides.error} onRetry={userOverrides.reload} /> : null}
        {overrideRows.length ? <div className="table-responsive"><table className="table table-sm align-middle mb-0"><thead><tr><th>{t('Permission')}</th><th>{t('Inherited')}</th><th>{t('Override')}</th><th>{t('Effective')}</th></tr></thead><tbody>{overrideRows.map((row) => <tr key={row.permission}><td>{permissionResourceLabel(row.permission)}</td><td>{row.inherited ? t('Allowed') : t('Not allowed')}</td><td><Select className="w-auto" value={row.override || ''} options={[{ value: '', label: t('Default (inherit role)') }, { value: 'allow', label: t('Allow') }, { value: 'deny', label: t('Deny') }]} onChange={(override) => setOverrideRows((rows) => rows.map((item) => item.permission === row.permission ? { ...item, override: override || null, effective: override === 'allow' ? true : override === 'deny' ? false : item.inherited } : item))} /></td><td>{row.effective ? t('Allowed') : t('Not allowed')}</td></tr>)}</tbody></table></div> : <Empty title={t('No staff users available for overrides.')} />}
        <div className="mt-3 d-flex justify-content-end"><Button pending={busy === 'overrides'} disabled={!!busy || !overrideUserId} onClick={async () => { setBusy('overrides'); setError(null); try { const saved = await api.saveUserPermissionOverrides(overrideUserId, { overrides: overrideRows.filter((row) => row.override).map((row) => ({ permission: row.permission, effect: row.override })) }); setOverrideRows(saved.permissions || []); } catch (reason) { setError(reason); } finally { setBusy(''); } }}>{t('Save overrides')}</Button></div>
      </Card> : null}
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
              ['floor_supervisor', 'Floor Supervisor']
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
