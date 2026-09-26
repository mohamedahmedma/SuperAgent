import { useMemo, useState } from 'react';
import { api } from '../api.js';
import { useQuery } from '../hooks.js';
import { Button, Card, Empty, ErrorNote, Field, Input, PageHead, Skeleton, Table } from '../components/Ui.jsx';

const PAGE_SIZE = 50;
const ENTITY = { Student: 'بيانات طالب', ClassEnrolment: 'تسجيل طالب', Guardian: 'ولي أمر', GuardianPhone: 'بيانات تواصل ولي الأمر', StudentGuardian: 'صلة ولي الأمر بالطالب', Attendance: 'حضور وغياب', SubjectGrade: 'درجة طالب', StudentDocument: 'مستند طالب', Homework: 'واجب دراسي', Teacher: 'بيانات معلم', User: 'حساب مستخدم', RoleAssignment: 'تعيين دور', TimetableLesson: 'حصة دراسية', Assessment: 'تقييم دراسي', AssessmentMark: 'درجة تقييم' };
const FIELD = { class_code: 'الفصل', subject_code: 'المادة', term_code: 'الفصل الدراسي', academic_year_code: 'العام الدراسي', name_ar: 'الاسم العربي', name_en: 'الاسم الإنجليزي', full_name_ar: 'الاسم', full_name_en: 'الاسم', phone: 'رقم الهاتف', address: 'العنوان', percentage: 'الدرجة', points: 'الدرجة', max_points: 'الدرجة الكلية', starts_on: 'تاريخ البداية', ends_on: 'تاريخ النهاية', state: 'الحالة', day_of_week: 'اليوم', period_number: 'رقم الحصة', title: 'عنوان الواجب', details: 'تفاصيل الواجب', original_filename: 'الملف المرفق', student_number: 'رقم الطالب', staff_number: 'رقم الموظف' };
const SUBJECT = { AR: 'اللغة العربية', MATH: 'الرياضيات', EN: 'اللغة الإنجليزية', CS: 'الكمبيوتر', ART: 'الرسم', PE: 'التربية الرياضية', RE: 'التربية الدينية', SCI: 'العلوم', SS: 'الدراسات الاجتماعية', BIO: 'الأحياء', PHY: 'الفيزياء', CHEM: 'الكيمياء', HIST: 'التاريخ', GEO: 'الجغرافيا' };

function yearLabel(value) { const found = String(value || '').match(/20\d{2}[-/]20\d{2}/); return found ? found[0].replace('-', ' / ') : (value || '—'); }
function classLabel(value) { const found = String(value || '').match(/^(KG|PREP|P|SEC)(\d+)-(\d+)$/i); if (!found) return value || '—'; const stage = { KG: 'رياض الأطفال', P: 'الابتدائي', PREP: 'الإعدادي', SEC: 'الثانوي' }[found[1].toUpperCase()]; return `${stage} ${found[2]} — فصل ${Number(found[3])}`; }
function readableValue(key, value) {
  if (value === null || value === undefined || value === '') return '';
  if (key === 'class_code') return classLabel(value);
  if (key === 'subject_code') return SUBJECT[value] || 'مادة دراسية';
  if (key === 'academic_year_code' || key === 'term_code') return yearLabel(value);
  if (key === 'state') return ({ present: 'حاضر', absent: 'غائب', late: 'متأخر', excused: 'بعذر' })[value] || value;
  if (typeof value === 'boolean') return value ? 'نعم' : 'لا';
  return String(value);
}
const hasValue = (value) => value !== null && value !== undefined && value !== '';
function changes(entry) {
  const before = entry.old_values || {}; const after = entry.new_values || {};
  return [...new Set([...Object.keys(before), ...Object.keys(after)])]
    .filter((key) => FIELD[key] && before[key] !== after[key] && (hasValue(before[key]) || hasValue(after[key])))
    .map((key) => ({ label: FIELD[key], before: readableValue(key, before[key]), after: readableValue(key, after[key]), hadBefore: hasValue(before[key]), hasAfter: hasValue(after[key]) }));
}
function summary(entry) {
  const changed = changes(entry); const item = ENTITY[entry.entity_type] || 'سجل';
  if (entry.action === 'homework_uploaded') return `رفع واجب: ${(entry.context && entry.context.title) || ''}`;
  if (entry.action === 'homework_deleted') return `حذف واجب: ${(entry.context && entry.context.title) || ''}`;
  if (entry.action === 'homework_restored') return `استعادة واجب: ${(entry.context && entry.context.title) || ''}`;
  if (entry.entity_type === 'ClassEnrolment' && changed.some((row) => row.label === 'الفصل')) { const row = changed.find((candidate) => candidate.label === 'الفصل'); return `نقل طالب من ${row.before} إلى ${row.after}`; }
  if (entry.entity_type === 'TimetableLesson') return 'تم تعديل حصة في الجدول الدراسي';
  if (entry.entity_type === 'SubjectGrade' || entry.entity_type === 'AssessmentMark') return 'تم تعديل درجة طالب';
  if (entry.action === 'soft_delete') return `تم إلغاء ${item}`;
  if (entry.action === 'restore') return `تم استعادة ${item}`;
  if (entry.action.includes('create')) return `تمت إضافة ${item}`;
  return changed.length ? `تم تعديل ${item}` : `تم تحديث ${item}`;
}
function Details({ entry }) {
  const rows = changes(entry);
  return (
    <details className="sis-audit-details">
      <summary>التفاصيل</summary>
      {rows.length ? (
        <ul>
          {rows.map((row, index) => (
            <li key={`${row.label}-${index}`}>
              <strong>{row.label}:</strong>{' '}
              {row.hadBefore && row.hasAfter ? (
                <>{row.before} ← {row.after}</>
              ) : row.hasAfter ? row.after : (
                <>{row.before} <small className="text-body-tertiary">(تم حذفه)</small></>
              )}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mb-0 text-body-tertiary">لا توجد تفاصيل إضافية متاحة لهذا الإجراء.</p>
      )}
    </details>
  );
}

function AuditMobileCard({ entry }) {
  const initial = String(entry.actor_name || '؟').trim().charAt(0);
  const homeworkScope = entry.entity_type === 'Homework' && entry.context;
  return (
    <article className="sis-audit-mobile-card">
      <header>
        <div className="sis-audit-mobile-actor">
          <span className="sis-audit-mobile-avatar" aria-hidden="true">{initial}</span>
          <div>
            <strong>{entry.actor_name}</strong>
            <small>{entry.actor_role || 'النظام'}</small>
          </div>
        </div>
        <time dateTime={entry.created_at}>{String(entry.created_at).slice(0, 16).replace('T', ' ')}</time>
      </header>
      <div className="sis-audit-mobile-summary">
        <span>الإجراء</span>
        <strong>{summary(entry)}</strong>
      </div>
      {entry.academic_year ? (
        <div className="sis-audit-mobile-meta">
          <span>العام الدراسي</span>
          <b>{yearLabel(entry.academic_year)}</b>
        </div>
      ) : null}
      {homeworkScope ? (
        <div className="sis-audit-mobile-meta">
          <span>النطاق</span>
          <b>{SUBJECT[entry.context.subject_code] || entry.context.subject_code} · {classLabel(entry.context.class_code)}</b>
        </div>
      ) : null}
      <Details entry={entry} />
    </article>
  );
}

export function AuditLog() {
  const [entityType, setEntityType] = useState('');
  const [action, setAction] = useState('');
  const [page, setPage] = useState(0);
  const filters = useMemo(() => ({
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
    entity_type: entityType || null,
    action: action || null
  }), [entityType, action, page]);
  const log = useQuery(() => api.auditLog(filters), [filters]);
  const rows = log.value || [];
  const reset = () => {
    setEntityType('');
    setAction('');
    setPage(0);
  };
  return <div className="sis-audit-page"><PageHead title="سجل التدقيق" subtitle="ملخص واضح لكل تغيير مهم داخل النظام." /><Card title="سجل التدقيق" subtitle="اسم من نفّذ الإجراء، دوره، العام الدراسي، ثم ملخص التعديل.">
    <div className="row g-3 align-items-end mb-3 sis-audit-filters"><Field className="col-12 col-md-5" label="نوع السجل"><Input value={entityType} onInput={(value) => { setEntityType(value); setPage(0); }} placeholder="مثال: طالب" /></Field><Field className="col-12 col-md-5" label="الإجراء"><Input value={action} onInput={(value) => { setAction(value); setPage(0); }} placeholder="مثال: تعديل" /></Field><div className="col-12 col-md-2 d-grid"><Button variant="outline" onClick={reset} disabled={!entityType && !action}>مسح عوامل التصفية</Button></div></div>
    {log.error ? <ErrorNote error={log.error} onRetry={log.reload} /> : log.loading && !log.value ? <Skeleton rows={6} /> : <><div className="d-none d-md-block"><Table loading={log.loading} rows={rows} rowKey={(entry) => entry.id} empty={<Empty title="لا توجد سجلات تدقيق">ستظهر التغييرات المهمة هنا فور تسجيلها.</Empty>} columns={[
      { key: 'when', header: 'الوقت', className: 'sis-code text-nowrap', cell: (entry) => String(entry.created_at).slice(0, 16).replace('T', ' ') },
      { key: 'actor', header: 'من نفّذ التعديل', cell: (entry) => <div><strong>{entry.actor_name}</strong>{entry.entity_type === 'Homework' && entry.context ? <small className="d-block text-body-tertiary">{SUBJECT[entry.context.subject_code] || entry.context.subject_code} · {classLabel(entry.context.class_code)}</small> : null}</div> },
      { key: 'role', header: 'الدور', cell: (entry) => entry.actor_role || 'النظام' },
      { key: 'year', header: 'العام الدراسي', hide: 'lg', cell: (entry) => entry.academic_year ? yearLabel(entry.academic_year) : '—' },
      { key: 'summary', header: 'التعديل', cell: (entry) => <strong>{summary(entry)}</strong> },
      { key: 'details', header: 'التفاصيل', hide: 'md', cell: (entry) => <Details entry={entry} /> }
    ]} /></div><div className="sis-audit-mobile-list d-md-none">{rows.length ? rows.map((entry) => <AuditMobileCard entry={entry} key={entry.id} />) : <Empty title="لا توجد سجلات تدقيق">ستظهر التغييرات المهمة هنا فور تسجيلها.</Empty>}</div></>}
    <div className="card-footer sis-audit-pagination"><span className="small text-body-tertiary">الصفحة {page + 1}</span><div className="btn-group" role="group" aria-label="صفحات سجل التدقيق"><Button size="sm" variant="outline" disabled={page === 0 || log.loading} onClick={() => setPage((value) => value - 1)}>السابق</Button><Button size="sm" variant="outline" disabled={rows.length < PAGE_SIZE || log.loading} onClick={() => setPage((value) => value + 1)}>التالي</Button></div></div>
  </Card></div>;
}
