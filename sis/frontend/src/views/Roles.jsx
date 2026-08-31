import { useState } from 'react';
import { api } from '../api.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { Badge, Card, ErrorNote, PageHead, Skeleton } from '../components/Ui.jsx';
import { t } from '../i18n.js';

const ASSIGNABLE = ['teacher', 'year_supervisor', 'attendance_supervisor'];
const FALLBACK = {
  teacher: 'Teacher',
  year_supervisor: 'Grade Supervisor',
  attendance_supervisor: 'Attendance Supervisor'
};

export function Roles() {
  const state = useStore();
  const school = state.school;
  const data = useQuery(
    () => Promise.all([api.teachers(school), api.rbacUsers(), api.rbacRoles(), api.teacherAttendance(school)]),
    [school],
    !!school
  );
  const [busy, setBusy] = useState('');
  const [error, setError] = useState(null);

  if (data.loading && !data.ready) return <><PageHead title={t('Teacher roles')} /><Card><Skeleton rows={6} /></Card></>;
  if (data.error) return <><PageHead title={t('Teacher roles')} /><ErrorNote error={data.error} onRetry={data.reload} /></>;

  const [teachers = [], users = [], roles = [], attendance = []] = data.value || [];
  const usersById = Object.fromEntries(users.map((user) => [user.id, user]));
  const roleNames = Object.fromEntries(roles.map((role) => [role.code, role]));

  const toggle = (user, code, checked) => {
    const grant = { role_code: code, scope_type: 'school', scope_id: user.school_id };
    const key = `${user.id}:${code}`;
    setBusy(key); setError(null);
    const action = checked ? api.addUserRole(user.id, grant) : api.removeUserRole(user.id, grant);
    action.then(() => data.reload(), setError).finally(() => setBusy(''));
  };

  return (
    <>
      <PageHead title={t('Teacher roles')} lede={t('Roles are additive. Selecting a supervisor role keeps the Teacher role active.')} />
      {error ? <ErrorNote error={error} /> : null}
      <div className="vstack gap-3">
        {teachers.map((teacher) => {
          const user = usersById[teacher.user_id];
          const held = new Set((user && user.roles || []).map((role) => role.role_code));
          return (
            <Card key={teacher.staff_number} title={pickName(teacher, state.lang) || teacher.staff_number}
              subtitle={user ? user.username : t('No login account')}>
              {user ? <div className="d-flex flex-wrap gap-3" role="group" aria-label={t('Active roles')}>
                {ASSIGNABLE.map((code) => {
                  const checked = held.has(code);
                  const label = pickName(roleNames[code] || {}, state.lang) || t(FALLBACK[code]);
                  return <label className="form-check form-switch" key={code}>
                    <input className="form-check-input" type="checkbox" checked={checked}
                      disabled={!!busy} onChange={(event) => toggle(user, code, event.target.checked)} />
                    <span className="form-check-label">{label} {checked ? <Badge tone="ok">{t('Active')}</Badge> : null}</span>
                  </label>;
                })}
              </div> : <p className="text-body-tertiary mb-0">{t('Create or link a login account before assigning roles.')}</p>}
            </Card>
          );
        })}
        <Card title={t('Teacher attendance')} subtitle={t('{0} record(s)', [attendance.length])} tight>
          <div className="table-responsive"><table className="table mb-0"><thead><tr>
            <th>{t('Date')}</th><th>{t('Teacher')}</th><th>{t('Status')}</th><th>{t('Note')}</th>
          </tr></thead><tbody>{attendance.map((row) => <tr key={`${row.staff_number}:${row.on_date}`}>
            <td>{row.on_date}</td><td>{pickName(row, state.lang) || row.staff_number}</td>
            <td><Badge tone={row.state === 'present' ? 'ok' : 'warn'}>{t(row.state)}</Badge></td><td>{row.note || '—'}</td>
          </tr>)}</tbody></table></div>
        </Card>
      </div>
    </>
  );
}
