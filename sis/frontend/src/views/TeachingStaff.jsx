import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { Store } from '../store.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { t } from '../i18n.js';
import { Alert, Badge, Button, Card, Empty, ErrorNote, NoYearNotice, PageHead, Select, Skeleton, useConfirm } from '../components/Ui.jsx';

function SupervisorCard({ title, rows, users, roleCode, gradeId, lang, busy, onAssign, onRemove }) {
  const [userId, setUserId] = useState('');
  useEffect(() => setUserId(''), [gradeId]);
  const candidates = users.filter((user) => user.is_active !== false && !rows.some((row) => row.user.id === user.id));
  return <Card title={title} className="h-100">
    <div className="vstack gap-3">
      {rows.map(({ user, scopes, grants }) => <div className="d-flex flex-wrap align-items-center justify-content-between gap-3 border rounded-3 p-3" key={user.id}>
        <div>
          <strong>{pickName(user, lang) || user.username}</strong>
          <div className="small text-body-tertiary">{user.username}</div>
          <div className="d-flex flex-wrap gap-1 mt-2">{scopes.map((scope) => <Badge key={scope}>{scope}</Badge>)}</div>
        </div>
        <Button size="sm" variant="danger" disabled={busy} onClick={() => onRemove(user, grants)}>{t('Remove supervisor')}</Button>
      </div>)}
      {!rows.length ? <Empty title={t('No supervisor assigned')} /> : null}
      <div className="d-flex flex-wrap gap-2">
        <Select className="flex-grow-1" value={userId} disabled={busy} options={[
          { value: '', label: t('Choose an existing account') },
          ...candidates.map((user) => ({ value: String(user.id), label: `${pickName(user, lang) || user.username} (${user.username})` }))
        ]} onChange={setUserId} />
        <Button disabled={busy || !candidates.some((user) => String(user.id) === userId)} onClick={async () => {
          if (await onAssign(userId, roleCode)) setUserId('');
        }}>{t('Assign supervisor')}</Button>
      </div>
    </div>
  </Card>;
}

export function TeachingStaff() {
  const state = useStore();
  const [gradeId, setGradeId] = useState('');
  const [removing, setRemoving] = useState('');
  const [removeError, setRemoveError] = useState(null);
  const [dialog, ask] = useConfirm();

  const data = useQuery(() => Promise.all([
    api.teachers(state.school),
    api.archivedTeachers(state.school),
    api.rbacUsers(),
    api.rbacYearLevels(state.school),
    api.subjectAssignments(state.year),
    api.classes(state.year)
  ]), [state.school, state.year], !!state.school && !!state.year);

  const [teachers = [], archivedTeachers = [], users = [], grades = [], subjectBoard = [], classes = []] = data.value || [];
  const selectedGrade = grades.find((row) => String(row.id) === gradeId);

  useEffect(() => {
    if (!gradeId && grades.length) setGradeId(String(grades[0].id));
    if (gradeId && grades.length && !grades.some((row) => String(row.id) === gradeId)) {
      setGradeId(String(grades[0].id));
    }
  }, [grades.length, gradeId]);

  const gradeClasses = useMemo(() => {
    if (!selectedGrade) return [];
    return classes.filter((row) => row.year_level_code === selectedGrade.code);
  }, [classes, selectedGrade && selectedGrade.code]);
  const gradeClassCodes = new Set(gradeClasses.map((row) => String(row.code)));

  const supervisors = useMemo(() => {
    if (!selectedGrade) return { year: [], attendance: [] };
    const year = [];
    const attendance = [];
    users.filter((user) => user.is_active !== false).forEach((user) => {
      (user.roles || []).forEach((role) => {
        if (role.role_code === 'floor_supervisor' && role.scope_type === 'year_level' && String(role.scope_id) === gradeId) {
          year.push({ user, scope: selectedGrade.code, grant: role });
        }
        if (role.role_code === 'attendance_supervisor') {
          const gradeWide = role.scope_type === 'year_level' && String(role.scope_id) === gradeId;
          const classScoped = role.scope_type === 'class_section' && gradeClassCodes.has(String(role.scope_code || ''));
          if (gradeWide || classScoped) attendance.push({ user, scope: gradeWide ? selectedGrade.code : role.scope_code, grant: role });
        }
      });
    });
    const unique = (rows) => {
      const map = new Map();
      rows.forEach((row) => {
        const current = map.get(row.user.id) || { user: row.user, scopes: [], grants: [] };
        current.grants.push(row.grant);
        if (row.scope && !current.scopes.includes(row.scope)) current.scopes.push(row.scope);
        map.set(row.user.id, current);
      });
      return [...map.values()];
    };
    return { year: unique(year), attendance: unique(attendance) };
  }, [users, gradeId, selectedGrade && selectedGrade.code, gradeClasses.map((row) => row.code).join('|')]);

  const subjects = useMemo(() => {
    if (!selectedGrade) return [];
    const row = subjectBoard.find((item) => item.year_level_code === selectedGrade.code);
    return (row && row.subjects) || [];
  }, [subjectBoard, selectedGrade && selectedGrade.code]);

  const teacherRowsFor = (subjectCode) => teachers.filter((teacher) =>
    teacher.is_active && (teacher.assignments || []).some((assignment) =>
      assignment.academic_year_code === state.year &&
      assignment.year_level_code === selectedGrade?.code &&
      assignment.subject_code === subjectCode
    )
  );

  const removeTeacher = async (teacher) => {
    setRemoving(teacher.staff_number); setRemoveError(null);
    try {
      const outcome = await api.removeTeacher(state.school, teacher.staff_number);
      Store.invalidate('teachers:');
      Store.invalidate('roles:');
      Store.toast('ok', t('Teacher removed from active staff'), pickName(teacher, state.lang) || teacher.staff_number);
      data.reload();
      return outcome;
    } catch (reason) {
      setRemoveError(reason);
      throw reason;
    } finally { setRemoving(''); }
  };

  const restoreTeacher = async (teacher) => {
    setRemoving(teacher.staff_number); setRemoveError(null);
    try {
      await api.restoreTeacher(state.school, teacher.staff_number);
      Store.invalidate('teachers:');
      Store.invalidate('roles:');
      Store.toast('ok', t('Teacher restored to active staff'), pickName(teacher, state.lang) || teacher.staff_number);
      data.reload();
    } catch (reason) { setRemoveError(reason); }
    finally { setRemoving(''); }
  };

  const assignSupervisor = async (userId, roleCode) => {
    setRemoving(`supervisor:${userId}`); setRemoveError(null);
    try {
      await api.addUserRole(userId, { role_code: roleCode, scope_type: 'year_level', scope_id: Number(gradeId) });
      Store.invalidate('roles:');
      data.reload();
      return true;
    } catch (reason) { setRemoveError(reason); return false; }
    finally { setRemoving(''); }
  };

  const removeSupervisor = (user, grants) => ask({
    title: t('Remove supervisor assignment?'), tone: 'bad', confirmLabel: t('Remove supervisor'),
    body: t('This removes supervision permissions for this grade. The account and its other assignments are preserved.'),
    run: async () => {
      setRemoving(`supervisor:${user.id}`); setRemoveError(null);
      try {
        for (const grant of grants) {
          await api.removeUserRole(user.id, { role_code: grant.role_code, scope_type: grant.scope_type, scope_id: grant.scope_id });
        }
        Store.invalidate('roles:');
      } catch (reason) { setRemoveError(reason); throw reason; }
      finally { data.reload(); setRemoving(''); }
    }
  });

  if (!state.year) return <><PageHead title={t('Teaching staff')} /><NoYearNotice /></>;
  if (data.loading && !data.ready) return <><PageHead title={t('Teaching staff')} /><Card><Skeleton rows={8} /></Card></>;
  if (data.error) return <><PageHead title={t('Teaching staff')} /><ErrorNote error={data.error} onRetry={data.reload} /></>;

  return <>
    {dialog}
    <PageHead title={t('Teaching staff')} />
    {removeError ? <ErrorNote error={removeError} /> : null}
    <div className="vstack gap-3">
      <Card title={t('Grade')} subtitle={t('Choose a grade to see its supervisors and subject teachers.') }>
        <Select value={gradeId} options={grades.map((row) => ({
          value: String(row.id), label: pickName(row, state.lang) || row.code
        }))} onChange={setGradeId} />
      </Card>

      {selectedGrade ? <>
        <div className="row g-3">
          <div className="col-12 col-lg-6">
            <SupervisorCard title={t('Class supervisor')} rows={supervisors.year} users={users}
              roleCode="floor_supervisor" gradeId={gradeId} lang={state.lang} busy={!!removing}
              onAssign={assignSupervisor} onRemove={removeSupervisor} />
          </div>
          <div className="col-12 col-lg-6">
            <SupervisorCard title={t('Attendance supervisor')} rows={supervisors.attendance} users={users}
              roleCode="attendance_supervisor" gradeId={gradeId} lang={state.lang} busy={!!removing}
              onAssign={assignSupervisor} onRemove={removeSupervisor} />
          </div>
        </div>

        <Card title={t('Subject teachers')} subtitle={pickName(selectedGrade, state.lang) || selectedGrade.code}>
          {subjects.length ? <div className="vstack gap-3">
            {subjects.map((subject) => {
              const rows = teacherRowsFor(subject.code);
              return <div className="border rounded-3 p-3" key={subject.code}>
                <div className="d-flex flex-wrap align-items-center gap-2 mb-3">
                  <strong>{pickName(subject, state.lang) || subject.code}</strong>
                  <Badge>{subject.code}</Badge>
                </div>
                {rows.length ? <div className="vstack gap-2">{rows.map((teacher) => {
                  const assignments = (teacher.assignments || []).filter((assignment) =>
                    assignment.academic_year_code === state.year &&
                    assignment.year_level_code === selectedGrade.code &&
                    assignment.subject_code === subject.code
                  );
                  const classCodes = [...new Set(assignments.flatMap((assignment) => assignment.class_codes || []))];
                  return <div className="d-flex flex-column flex-lg-row align-items-lg-center justify-content-between gap-3 border rounded-3 p-3" key={teacher.staff_number}>
                    <div className="flex-grow-1">
                      <strong>{pickName(teacher, state.lang) || teacher.staff_number}</strong>
                      <div className="small text-body-tertiary">{teacher.staff_number}{teacher.username ? ` · ${teacher.username}` : ''}</div>
                      <div className="d-flex flex-wrap gap-1 mt-2">
                        {classCodes.length ? classCodes.map((code) => <Badge key={code}>{code}</Badge>) : <span className="small text-body-tertiary">{t('No class assigned yet')}</span>}
                      </div>
                    </div>
                    <Button size="sm" variant="danger" disabled={!!removing} pending={removing === teacher.staff_number}
                      onClick={() => ask({
                        title: t('Remove teacher from the system?'),
                        tone: 'bad',
                        confirmLabel: t('Remove teacher'),
                        changes: [
                          { label: t('Teacher'), was: pickName(teacher, state.lang) || teacher.staff_number, now: t('Removed from active staff') },
                          { label: t('Login account'), was: teacher.username || t('No login account'), now: t('Deactivated') }
                        ],
                        body: <Alert tone="warn">
                          {t('The teacher and login account will be deactivated. Their roles, teaching assignments, historical attendance, and recorded academic data remain preserved and can be restored by the school manager.')}
                        </Alert>,
                        run: () => removeTeacher(teacher)
                      })}>
                      {t('Remove teacher')}
                    </Button>
                  </div>;
                })}</div> : <Empty title={t('No teacher assigned to this subject')} />}
              </div>;
            })}
          </div> : <Empty title={t('No subjects configured for this grade')} />}
        </Card>
      </> : <Empty title={t('No grades configured')} />}
      {archivedTeachers.length ? <Card title={t('Archived teachers')} subtitle={t('These staff records and their teaching history are retained.')}>
        <div className="vstack gap-2">{archivedTeachers.map((teacher) => <div className="d-flex flex-wrap align-items-center justify-content-between gap-3 border rounded-3 p-3" key={teacher.staff_number}>
          <div><strong>{pickName(teacher, state.lang) || teacher.staff_number}</strong><div className="small text-body-tertiary">{teacher.staff_number}{teacher.username ? ` · ${teacher.username}` : ''}</div></div>
          <Button size="sm" variant="secondary" disabled={!!removing} pending={removing === teacher.staff_number} onClick={() => ask({
            title: t('Restore teacher?'), confirmLabel: t('Restore teacher'),
            body: t('This reactivates the teacher and their login account. Existing assignments and role scopes are retained.'),
            run: () => restoreTeacher(teacher)
          })}>{t('Restore teacher')}</Button>
        </div>)}</div>
      </Card> : null}
    </div>
  </>;
}
