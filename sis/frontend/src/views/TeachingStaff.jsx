import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { Store } from '../store.js';
import { pickName, useQuery, useStore } from '../hooks.js';
import { t } from '../i18n.js';
import { Alert, Badge, Button, Card, Empty, ErrorNote, NoYearNotice, PageHead, Select, Skeleton, useConfirm } from '../components/Ui.jsx';

export function TeachingStaff() {
  const state = useStore();
  const [gradeId, setGradeId] = useState('');
  const [removing, setRemoving] = useState('');
  const [removeError, setRemoveError] = useState(null);
  const [dialog, ask] = useConfirm();

  const data = useQuery(() => Promise.all([
    api.teachers(state.school),
    api.rbacUsers(),
    api.rbacYearLevels(state.school),
    api.subjectAssignments(state.year),
    api.classes(state.year)
  ]), [state.school, state.year], !!state.school && !!state.year);

  const [teachers = [], users = [], grades = [], subjectBoard = [], classes = []] = data.value || [];
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
        if (role.role_code === 'year_supervisor' && role.scope_type === 'year_level' && String(role.scope_id) === gradeId) {
          year.push({ user, scope: selectedGrade.code });
        }
        if (role.role_code === 'attendance_supervisor') {
          const gradeWide = role.scope_type === 'year_level' && String(role.scope_id) === gradeId;
          const classScoped = role.scope_type === 'class_section' && gradeClassCodes.has(String(role.scope_code || ''));
          if (gradeWide || classScoped) attendance.push({ user, scope: gradeWide ? selectedGrade.code : role.scope_code });
        }
      });
    });
    const unique = (rows) => {
      const map = new Map();
      rows.forEach((row) => {
        const current = map.get(row.user.id) || { user: row.user, scopes: [] };
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
            <Card title={t('Class supervisor')} className="h-100">
              {supervisors.year.length ? <div className="vstack gap-2">
                {supervisors.year.map(({ user }) => <div className="border rounded-3 p-3" key={user.id}>
                  <strong>{pickName(user, state.lang) || user.username}</strong>
                  <div className="small text-body-tertiary">{user.username}</div>
                </div>)}
              </div> : <Empty title={t('No class supervisor assigned')} />}
            </Card>
          </div>
          <div className="col-12 col-lg-6">
            <Card title={t('Attendance supervisor')} className="h-100">
              {supervisors.attendance.length ? <div className="vstack gap-2">
                {supervisors.attendance.map(({ user, scopes }) => <div className="border rounded-3 p-3" key={user.id}>
                  <strong>{pickName(user, state.lang) || user.username}</strong>
                  <div className="small text-body-tertiary">{user.username}</div>
                  {scopes.length ? <div className="d-flex flex-wrap gap-1 mt-2">{scopes.map((scope) => <Badge key={scope}>{scope}</Badge>)}</div> : null}
                </div>)}
              </div> : <Empty title={t('No attendance supervisor assigned')} />}
            </Card>
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
                          { label: t('Login account'), was: teacher.username || t('No login account'), now: t('Deleted') }
                        ],
                        body: <Alert tone="warn">
                          {t('The teacher account, active roles, and current teaching assignments will be removed. Historical attendance and recorded academic data remain preserved.')}
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
    </div>
  </>;
}
