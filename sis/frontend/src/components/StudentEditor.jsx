/*
 * Editing one child's record — the same form wherever it is reached from.
 *
 * It lives here rather than on either screen because the class register and the child's own
 * record both need it, and two copies of a form that PATCHes a person is how a field ends up
 * editable in one place and read-only in the other.
 *
 * Two decisions the form makes on the registrar's behalf:
 *
 * **Only what changed is sent.** `form.changed()` produces the diff and that diff is the whole
 * request body. Sending every field would overwrite a value another registrar edited between
 * this screen loading and this save — a phone number typed by the front desk, quietly replaced
 * by the stale one the form loaded ten minutes ago.
 *
 * **The confirmation shows was → now.** Structural and destructive actions are confirmed; this
 * is neither, but a name correction is the one edit a registrar makes fastest and regrets most,
 * so the diff is on screen before it is saved rather than in a toast afterwards.
 *
 * `StudentEditorFor` is the same form for a screen that has a student *number* and not a record
 * — the class register, whose rows carry names and a start date and nothing else. It fetches
 * the record first, so the "was" column states what is actually on file rather than what the
 * row it came from happened to include.
 */
import { api } from '../api.js';
import { Store } from '../store.js';
import { useAction, useForm, useResource } from '../hooks.js';
import { Button, ErrorNote, Field, Input, Skeleton, useConfirm } from './Ui.jsx';
import { t } from '../i18n.js';

const FIELDS = [
  { key: 'full_name_en', label: 'Name (English)' },
  { key: 'full_name_ar', label: 'Name (Arabic)', className: 'sis-name-ar' },
  { key: 'date_of_birth', label: 'Date of birth', type: 'date' },
  { key: 'contact_phone', label: 'Contact phone', type: 'tel', inputMode: 'tel' },
  { key: 'contact_email', label: 'Contact email', type: 'email', inputMode: 'email' },
  { key: 'address', label: 'Address', wide: true }
];

export function StudentEditor({ student, onDone }) {
  const initial = {};
  FIELDS.forEach((field) => {
    initial[field.key] = student[field.key] || '';
  });

  const form = useForm(initial);
  const [dialog, ask] = useConfirm();

  const changed = form.changed();
  const diff = Object.keys(changed).map((key) => ({
    label: (FIELDS.find((field) => field.key === key) || {}).label || key.replace(/_/g, ' '),
    was: student[key],
    now: changed[key]
  }));

  return (
    <>
      {dialog}
      <form
        className="vstack gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (!diff.length) {
            Store.toast('info', t('Nothing changed'));
            if (onDone) onDone();
            return;
          }
          ask({
            title: `Change ${student.student_number}?`,
            confirmLabel: 'Save the change',
            changes: diff,
            body: (
              <p className="mb-0">
                {t('This is her record, so the change applies in every year rather than only this one. Her marks, her attendance and her placements are untouched.')}
              </p>
            ),
            run: () =>
              api.updateStudent(student.student_number, changed).then(() => {
                Store.invalidate('roster:');
                Store.invalidate('student:');
                Store.toast('ok', t('{0} updated', [student.student_number]));
                if (onDone) onDone();
              })
          });
        }}
      >
        <div className="row g-3">
          {FIELDS.map((field) => (
            <Field
              key={field.key}
              className={field.wide ? 'col-12' : 'col-12 col-sm-6 col-lg-4'}
              label={field.label}
            >
              <Input
                className={field.className}
                type={field.type}
                inputMode={field.inputMode}
                value={form.values[field.key]}
                onInput={form.set(field.key)}
              />
            </Field>
          ))}
        </div>
        <div className="d-grid gap-2 d-sm-flex">
          <Button type="submit" variant="primary" disabled={!diff.length}>
            {diff.length ? `Save ${diff.length} change(s)` : 'Nothing to save'}
          </Button>
          {onDone ? (
            <Button variant="quiet" onClick={onDone}>
              {t('Cancel')}
            </Button>
          ) : null}
        </div>
      </form>
    </>
  );
}

export function StudentEditorFor({ studentNumber, academicYear, onDone }) {
  /* The same cache key the record screen uses, so opening the form from a class register and
     opening her record show one answer rather than two that can disagree.

     The year rides along because the read is scope-checked against it: a registrar who holds
     students.read over one year level is narrowed with `academic_year_code=""` when it is
     missing, no scope matches, and the form the Edit button opens is a 403 rather than her
     record. */
  const record = useResource(
    `${Store.keys.student(studentNumber)}:${academicYear || ''}`,
    () => api.student(studentNumber, academicYear),
    !!studentNumber
  );

  if (record.loading && !record.value) return <Skeleton rows={3} />;
  if (record.error) return <ErrorNote error={record.error} onRetry={record.reload} />;
  if (!record.value) return null;

  return <StudentEditor student={record.value} onDone={onDone} />;
}
