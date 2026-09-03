/* Roster — enrol children into classes, and read the register back. */
import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import { DASH, dateText, labelOf, pickName, useQuery, useResource, useStore } from '../hooks.js';
import { Badge, Button, Card, Empty, ErrorNote, Field, Input, NoYearNotice, PageHead, Select, Table } from '../components/Ui.jsx';
import { ImportFlow } from '../components/ImportFlow.jsx';
import { t } from '../i18n.js';

const TEMPLATE = {
  name: 'family-roster-template.csv',
  header: t('student_number,full_name_ar,full_name_en,class_code,guardian_name_ar,guardian_name_en,guardian_phone,relationship_type,is_primary_contact,can_view_records')
};

const emptyAdmission = () => ({
  full_name_ar: '', full_name_en: '', gender: '',
  date_of_birth: '', contact_email: '', address: '',
  guardian_full_name_ar: '', guardian_full_name_en: '', guardian_phone: '',
  relationship_type: '', relationship_label: '', grade_code: '', class_code: ''
});

function NewAdmission({ year, classes, onCreated }) {
  const state = useStore();
  const [form, setForm] = useState(emptyAdmission());
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const set = (key) => (value) => setForm((old) => ({ ...old, [key]: value }));
  const requiredFields = Object.keys(form).filter((key) => key !== 'contact_email');
  const required = requiredFields.every((key) => String(form[key]).trim());
  const grades = [...new Map(classes.map((row) => [row.year_level_code, {
    code: row.year_level_code,
    name_en: row.year_level_name_en,
    name_ar: row.year_level_name_ar
  }])).values()].filter((row) => row.code);
  const gradeClasses = classes.filter((row) => row.year_level_code === form.grade_code);

  const save = async () => {
    setSaving(true); setError(null);
    try {
      const result = await api.admitStudent({ ...form, academic_year_code: year });
      Store.invalidate('roster:');
      Store.invalidate('student:');
      Store.toast(t('Student admitted successfully.'), 'success');
      setForm(emptyAdmission());
      onCreated(result);
    } catch (reason) { setError(reason); }
    finally { setSaving(false); }
  };

  return (
    <Card
      title={t('Admit a new student')}
    >
      <h3 className="h6 mb-3">{t('Student information')}</h3>
      <div className="row g-3">
        <Field className="col-12 col-md-6" label={t('Arabic name')} required>
          <Input className="sis-name-ar" value={form.full_name_ar} onInput={set('full_name_ar')} />
        </Field>
        <Field className="col-12 col-md-6" label={t('English name')} required>
          <Input className="sis-name-en" value={form.full_name_en} onInput={set('full_name_en')} />
        </Field>
        <Field className="col-12 col-md-3" label={t('Gender')} required>
          <Select value={form.gender} onChange={set('gender')} options={[
            { value: '', label: t('Choose…') }, { value: 'male', label: t('Male') },
            { value: 'female', label: t('Female') }
          ]} />
        </Field>
        <Field className="col-12 col-md-3" label={t('Date of birth')} required>
          <Input type="date" value={form.date_of_birth} onInput={set('date_of_birth')} />
        </Field>
        <Field className="col-12 col-md-6" label={t('Student email')}>
          <Input type="email" placeholder={t('Optional')} value={form.contact_email} onInput={set('contact_email')} />
        </Field>
        <Field className="col-12" label={t('Address')} required>
          <Input value={form.address} onInput={set('address')} />
        </Field>
      </div>

      <h3 className="h6 mt-4 mb-3">{t('Guardian information')}</h3>
      <div className="row g-3">
        <Field className="col-12 col-md-4" label={t('Guardian Arabic name')} required>
          <Input className="sis-name-ar" value={form.guardian_full_name_ar} onInput={set('guardian_full_name_ar')} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Guardian English name')} required>
          <Input className="sis-name-en" value={form.guardian_full_name_en} onInput={set('guardian_full_name_en')} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Guardian phone')} required>
          <Input inputMode="tel" value={form.guardian_phone} onInput={set('guardian_phone')} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Relationship')} required>
          <Select value={form.relationship_type} onChange={set('relationship_type')} options={[
            { value: '', label: t('Choose…') }, { value: 'father', label: t('Father') },
            { value: 'mother', label: t('Mother') }, { value: 'guardian', label: t('Guardian') },
            { value: 'sibling', label: t('Sibling') }, { value: 'grandparent', label: t('Grandparent') },
            { value: 'other', label: t('Other') }
          ]} />
        </Field>
        <Field className="col-12 col-md-8" label={t('Relationship description')} required>
          <Input value={form.relationship_label} onInput={set('relationship_label')} />
        </Field>
      </div>

      <h3 className="h6 mt-4 mb-3">{t('Placement')}</h3>
      <div className="row g-3">
        <Field className="col-12 col-md-4" label={t('Academic year')} required>
          <Input className="sis-code" value={year} disabled />
        </Field>
        <Field className="col-12 col-md-4" label={t('Grade')} required>
          <Select value={form.grade_code}
            onChange={(value) => setForm((old) => ({ ...old, grade_code: value, class_code: '' }))}
            options={[{ value: '', label: t('Choose…') }, ...grades.map((row) => ({
              value: row.code, label: pickName(row, state.lang) || row.code
            }))]} />
        </Field>
        <Field className="col-12 col-md-4" label={t('Class')} required>
          <Select value={form.class_code} disabled={!form.grade_code} onChange={set('class_code')}
            options={[{ value: '', label: t('Choose…') }, ...gradeClasses.map((row) => ({
              value: row.code,
              label: pickName(row, state.lang) || row.code
            }))]} />
        </Field>
      </div>
      {error ? <div className="mt-3"><ErrorNote error={error} /></div> : null}
      <div className="mt-4">
        <Button variant="primary" pending={saving} disabled={!required} onClick={save}>
          {t('Create student and guardian')}
        </Button>
      </div>
    </Card>
  );
}

export function StudentSetup() {
  const state = useStore();
  const classes = useResource(
    Store.keys.classes(state.year), () => api.classes(state.year), !!state.year
  );
  const [createdClass, setCreatedClass] = useState('');

  if (!state.year) {
    return <><PageHead title={t('Student setup')} /><NoYearNotice /></>;
  }
  return <>
    <PageHead
      title={t('Student setup')}
      lede={t('Create one complete student record with a guardian and first class placement.')}
    />
    <NewAdmission
      year={state.year}
      classes={classes.value || []}
      onCreated={(result) => setCreatedClass(result.placement.class_code)}
    />
    {createdClass ? (
      <div className="alert alert-success mt-3">
        {t('The student was added to class {0}.', [createdClass])}
      </div>
    ) : null}
  </>;
}

/* -- The register ---------------------------------------------------------------- */

function Register({ year, classCode }) {
  const state = useStore();
  const [picked, setPicked] = useState(classCode || '');
  const [gradeCode, setGradeCode] = useState('');

  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);
  const sections = classes.value || [];
  const grades = [...new Map(sections.map((row) => [row.year_level_code, {
    code: row.year_level_code,
    name_en: row.year_level_name_en,
    name_ar: row.year_level_name_ar
  }])).values()].filter((row) => row.code);
  const gradeClasses = sections.filter((row) => row.year_level_code === gradeCode);

  useEffect(() => {
    if (!classCode || !sections.length) return;
    const section = sections.find((row) => row.code === classCode);
    if (section) setGradeCode(section.year_level_code);
  }, [classCode, sections.length]);
  const register = useQuery(
    () => api.classRoster(picked, year),
    [picked, year],
    !!(picked && year)
  );

  const list = (register.value && register.value.students) || [];

  return (
    <Card
      title={t('The register')}
      subtitle={
        register.value
          ? `${register.value.count} on the register as of ${dateText(register.value.as_of)}`
          : t('Read a class back after uploading')
      }
      actions={
        picked ? (
          <Button size="sm" icon="refresh" onClick={register.reload}>
            {t('Reload')}
          </Button>
        ) : null
      }
      tight
    >
      <div className="card-body row g-3">
        <Field className="col-12 col-md-6" label={t('Grade')}>
          <Select
            value={gradeCode}
            options={[{ value: '', label: t('Choose grade') }, ...grades.map((grade) => ({
              value: grade.code,
              label: pickName(grade, state.lang) || grade.code
            }))]}
            onChange={(value) => { setGradeCode(value); setPicked(''); }}
          />
        </Field>
        <Field className="col-12 col-md-6" label={t('Class')} hint={t('Choose a grade first.')}>
          <Select
            className="sis-code"
            value={picked}
            disabled={!gradeCode}
            placeholder={t('— choose a class —')}
            options={gradeClasses.map((section) => ({
              value: section.code,
              label: labelOf(section, state.lang)
            }))}
            onChange={setPicked}
          />
        </Field>
      </div>

      <ErrorNote error={register.error} onRetry={register.reload} />

      {!picked ? (
        <Empty title={t('No class chosen')}>{t('Pick a class to see who is on its register today.')}</Empty>
      ) : (
        <Table
          loading={register.loading}
          rows={list}
          rowKey={(row) => row.student_number}
          rowHref={(row) => Router.href('student', { number: row.student_number })}
          rowLabel={(row) => t('Open {0}', [pickName(row, state.lang) || row.student_number]).join('')}
          empty={
            <Empty title={t('Nobody is placed in this class')}>
              {t('Upload a roster above, choosing this class, to enrol children into it.')}
            </Empty>
          }
          columns={[
            {
              key: 'number',
              header: t('Student no.'),
              className: 'sis-code',
              cell: (row) => (
                <a
                  className="sis-plain"
                  href={Router.href('student', { number: row.student_number })}
                >
                  {row.student_number}
                </a>
              )
            },
            {
              key: 'name',
              header: t('Name'),
              className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
              cell: (row) =>
                /* Empty when the student row could not be loaded. The placement is still
                   listed — a register quietly one child short is worse than one showing a
                   number without a name. */
                pickName(row, state.lang) || (
                  <span className="sis-ungraded">{DASH} name not on file</span>
                )
            },
            {
              key: 'from',
              header: t('From'),
              className: 'sis-num',
              hide: 'md',
              cell: (row) => <span className="font-monospace small">{dateText(row.starts_on)}</span>
            },
            {
              key: 'to',
              header: t('Until'),
              className: 'sis-num',
              hide: 'md',
              cell: (row) =>
                /* `ends_on` is her LAST DAY in the class, not the day after, and null means the
                   placement is open — she is in this class now. "Open" is the honest word; a
                   blank here would read as missing data. */
                row.ends_on ? (
                  <span className="font-monospace small">{dateText(row.ends_on)}</span>
                ) : (
                  <Badge tone="ok">{t('open')}</Badge>
                )
            },
            {
              key: 'links',
              header: '',
              cell: (row) => (
                <div className="sis-row-actions d-flex gap-1">
                  <a
                    className="btn btn-sm btn-quiet"
                    href={Router.href('guardians', { student: row.student_number })}
                  >
                    {t('Guardians')}
                  </a>
                  <a
                    className="btn btn-sm btn-quiet d-none d-lg-inline-flex"
                    href={Router.href('marks', { student: row.student_number })}
                  >
                    {t('Marks')}
                  </a>
                </div>
              )
            }
          ]}
        />
      )}
    </Card>
  );
}

/* -- Screen ---------------------------------------------------------------------- */

export function Roster({ params = {} }) {
  const state = useStore();
  const year = state.year;
  const [classCode, setClassCode] = useState('');
  const [gradeCode, setGradeCode] = useState('');

  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);
  const sections = classes.value || [];
  const grades = [...new Map(sections.map((row) => [row.year_level_code, {
    code: row.year_level_code,
    name_en: row.year_level_name_en,
    name_ar: row.year_level_name_ar
  }])).values()].filter((row) => row.code);
  const gradeClasses = sections.filter((row) => row.year_level_code === gradeCode);

  if (!year) {
    return (
      <>
        <PageHead title={t('Roster')} />
        <NoYearNotice />
      </>
    );
  }

  const fields = (
    <div className="row g-3">
      <Field
        className="col-12 col-sm-6 col-lg-4"
        label={t('Academic year')}
        hint={t('Chosen in the header, for every screen.')}
      >
        <Input className="sis-code" value={year} disabled />
      </Field>
      <Field
        className="col-12 col-sm-6 col-lg-4"
        label={t('Grade')}
      >
        <Select
          value={gradeCode}
          options={[{ value: '', label: t('Choose grade') }, ...grades.map((grade) => ({
            value: grade.code,
            label: pickName(grade, state.lang) || grade.code
          }))]}
          onChange={(value) => { setGradeCode(value); setClassCode(''); }}
        />
      </Field>
      <Field
        className="col-12 col-sm-6 col-lg-4"
        label={t('Class')}
        hint={t('Only classes in the selected grade are shown.')}
      >
        <Select
          className="sis-code"
          value={classCode}
          disabled={!gradeCode}
          placeholder={t('— from the file —')}
          options={gradeClasses.map((section) => ({
            value: section.code,
            label: labelOf(section, state.lang)
          }))}
          onChange={setClassCode}
        />
      </Field>
    </div>
  );

  return (
    <>
      <PageHead
        title={t('Roster')}
        lede={t('Enrol children and place them in classes. A placement is a dated membership: a child who moves from 3A to 3B in March is in 3A for Term 1 and 3B for Term 2, and both stay true.')}
      />

      <div className="vstack gap-4">
        <ImportFlow
          kind="roster"
          template={TEMPLATE}
          label={t('Choose one student and guardian roster sheet')}
          hint={t('One row per student. Include the guardian in the same row; alternate guardian fields remain optional.')}
          invalidate={['classes:']}
          fields={fields}
          onPreview={(file) => {
            const form = new FormData();
            form.append('file', file);
            form.append('academic_year_code', year);
            if (classCode) form.append('class_code', classCode);
            return api.previewRoster(form);
          }}
          onCommit={(batchId) => api.commitRoster(batchId)}
        />

        <Register year={year} classCode={classCode || params.class} />
      </div>
    </>
  );
}
