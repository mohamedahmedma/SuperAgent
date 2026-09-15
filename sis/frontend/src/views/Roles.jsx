import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { Store } from '../store.js';
import { Badge, Button, Card, Empty, ErrorNote, PageHead, Select, Skeleton } from '../components/Ui.jsx';
import { t } from '../i18n.js';

const PERMISSION_GROUPS = [
  ['system', 'System administration', ['system.manage', 'system.status.write', 'audit.read']],
  ['schools', 'Schools', ['schools.read', 'schools.write']],
  ['accounts', 'Accounts and roles', ['users.read', 'users.write', 'roles.assign']],
  ['structure', 'Academic structure', ['structure.read', 'structure.write', 'timetable.read', 'timetable.write']],
  ['students', 'Students and guardians', ['students.read', 'students.create', 'students.write', 'guardians.read', 'guardians.write', 'documents.read', 'documents.write']],
  ['teaching', 'Teachers and assignments', ['teachers.read', 'teachers.write', 'teachers.assign_subjects', 'teachers.assign_classes', 'teacher_attendance.read', 'teacher_attendance.write']],
  ['learning', 'Attendance and grades', ['attendance.read', 'attendance.write', 'grades.read', 'grades.write', 'reports.read', 'imports.run']],
  ['communication', 'Communication', ['chat.read', 'chat.write']]
];

const PERMISSION_LABELS = {
  'system.manage': 'Manage system settings', 'system.status.write': 'Change system status', 'audit.read': 'View audit log',
  'schools.read': 'View schools', 'schools.write': 'Manage schools',
  'users.read': 'View staff accounts', 'users.write': 'Manage staff accounts', 'roles.assign': 'Assign roles and scopes',
  'structure.read': 'View academic structure', 'structure.write': 'Manage years, grades, classes and subjects',
  'timetable.read': 'View timetables', 'timetable.write': 'Create and edit timetables',
  'students.read': 'View student records', 'students.create': 'Create students and initial enrolment', 'students.write': 'Edit, transfer and promote students',
  'guardians.read': 'View guardians', 'guardians.write': 'Manage guardians', 'documents.read': 'View student documents', 'documents.write': 'Manage student documents',
  'teachers.read': 'View teachers', 'teachers.write': 'Manage teacher records', 'teachers.assign_subjects': 'Assign subjects to teachers',
  'teachers.assign_classes': 'Assign classes to teachers', 'teacher_attendance.read': 'View staff attendance', 'teacher_attendance.write': 'Record staff attendance',
  'attendance.read': 'View student attendance', 'attendance.write': 'Record student attendance',
  'grades.read': 'View grades', 'grades.write': 'Record and edit grades', 'reports.read': 'View reports', 'imports.run': 'Upload and commit import files',
  'chat.read': 'Read staff chats', 'chat.write': 'Send and edit staff messages'
};

const permissionResourceLabel = (permission) => t(PERMISSION_LABELS[permission] || permission);

export function Roles() {
  const state = useStore();
  const [grade, setGrade] = useState('');
  const [subject, setSubject] = useState('');
  const [busy, setBusy] = useState('');
  const [error, setError] = useState(null);
  const [permissionRole, setPermissionRole] = useState('school_manager');
  const [selectedPermissions, setSelectedPermissions] = useState([]);
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
  const permissions = useQuery(() => api.rolePermissions(permissionRole), [permissionRole], isAdmin);
  const overrideUsers = users.filter((user) => !(user.roles || []).some((role) => role.role_code === 'admin'));
  const userOverrides = useQuery(
    () => api.userPermissionOverrides(overrideUserId),
    [overrideUserId],
    isAdmin && !!overrideUserId
  );
  useEffect(() => { if (permissions.value) setSelectedPermissions(permissions.value.permissions || []); }, [permissions.value]);
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
      {isAdmin ? <Card title={t('Role permissions')} subtitle={t('Set every feature permission independently. Changes affect every user who has this role.') }>
        <div className="sis-permission-toolbar"><div><label className="form-label">{t('Role')}</label><Select value={permissionRole} options={[
          ['school_owner', t('School Owner')], ['school_manager', t('School Manager')], ['floor_supervisor', t('Floor Supervisor')], ['attendance_supervisor', t('Attendance Supervisor')], ['teacher', t('Teacher')]
        ].map(([value, label]) => ({ value, label }))} onChange={setPermissionRole} /></div><Badge tone="info">{t('{0} permissions selected.', [selectedPermissions.length])}</Badge></div>
        {permissions.error ? <ErrorNote error={permissions.error} onRetry={permissions.reload} /> : null}
        <div className="sis-permission-grid">{PERMISSION_GROUPS.map(([group, label, codes]) => {
          const available = permissionRole === 'school_owner' ? codes.filter((code) => code.endsWith('.read')) : codes;
          const allChecked = available.length > 0 && available.every((code) => selectedPermissions.includes(code));
          const enabled = available.filter((code) => selectedPermissions.includes(code)).length;
          return <section className="sis-permission-group" key={group} aria-labelledby={`permission-${group}`}>
            <header className="sis-permission-group-head"><div><h3 id={`permission-${group}`}>{t(label)}</h3><small>{t('{0} of {1} enabled', [enabled, available.length])}</small></div>
              <Button size="sm" variant="outline" onClick={() => setSelectedPermissions((current) => allChecked ? current.filter((code) => !available.includes(code)) : [...new Set([...current, ...available])])}>{allChecked ? t('Clear group') : t('Allow group')}</Button></header>
            <div className="sis-permission-list">{available.map((code) => <label className="sis-permission-option" key={code}>
              <span className="sis-permission-copy">{permissionResourceLabel(code)}</span>
              <input type="checkbox" role="switch" aria-label={permissionResourceLabel(code)} checked={selectedPermissions.includes(code)} onChange={(event) => setSelectedPermissions((current) => event.target.checked ? [...current, code] : current.filter((item) => item !== code))} />
            </label>)}</div>
          </section>;
        })}</div>
        <div className="sis-permission-save"><Button pending={busy === 'permissions'} disabled={!!busy || permissions.loading} onClick={async () => { setBusy('permissions'); setError(null); try { const saved = await api.saveRolePermissions(permissionRole, selectedPermissions); setSelectedPermissions(saved.permissions || []); Store.invalidate('roles:'); Store.toast(t('Permissions saved successfully.'), 'success'); } catch (reason) { setError(reason); } finally { setBusy(''); } }}>{t('Save permissions')}</Button></div>
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
