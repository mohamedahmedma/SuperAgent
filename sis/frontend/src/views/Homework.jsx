import { useEffect, useMemo, useState } from 'react';
import { api } from '../api.js';
import { Store } from '../store.js';
import { useQuery, useStore } from '../hooks.js';
import { Button, Card, Dropzone, Empty, ErrorNote, Field, Input, PageHead, Select, Table } from '../components/Ui.jsx';
import { t } from '../i18n.js';

function subjectLabel(row, lang) {
  return lang === 'ar' ? (row.subject_name_ar || row.subject_code) : (row.subject_name_en || row.subject_code);
}

export function Homework() {
  const state = useStore();
  const year = state.year;
  const [assignmentKey, setAssignmentKey] = useState('');
  const [title, setTitle] = useState('');
  const [details, setDetails] = useState('');
  const [file, setFile] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);

  const teaching = useQuery(() => api.teachingAssignments(year), [year], !!year);
  const homework = useQuery(() => api.teacherHomework(year), [year], !!year);
  const assignments = (teaching.value && teaching.value.assignments) || [];

  useEffect(() => {
    if (!assignmentKey && assignments.length) {
      setAssignmentKey(`${assignments[0].class_code}\u0000${assignments[0].subject_code}`);
    }
  }, [assignmentKey, assignments.length]);

  const selected = useMemo(() => assignments.find((row) =>
    `${row.class_code}\u0000${row.subject_code}` === assignmentKey
  ), [assignments, assignmentKey]);

  const canPublish = !!(selected && title.trim() && !saving);

  async function publish() {
    if (!canPublish) return;
    setSaving(true);
    setSaveError(null);
    try {
      await api.uploadHomework({
        academicYear: year,
        classCode: selected.class_code,
        subjectCode: selected.subject_code,
        title: title.trim(),
        details: details.trim(),
        file
      });
      Store.toast(t('Homework published.'), 'success');
      setTitle('');
      setDetails('');
      setFile(null);
      homework.reload();
    } catch (error) {
      setSaveError(error);
    } finally {
      setSaving(false);
    }
  }

  async function remove(row) {
    if (!window.confirm(t('Delete this homework item?'))) return;
    try {
      await api.deleteHomework(row.id);
      Store.toast(t('Homework deleted.'), 'success');
      homework.reload();
    } catch (error) {
      setSaveError(error);
    }
  }

  return <>
    <PageHead title={t('Homework')} lede={t('Publish today’s assignment for your own class and subject.')} />
    <div className="vstack gap-3">
      <Card title={t('Publish homework')} tight>
        <div className="card-body row g-3">
          <Field className="col-12 col-lg-6" label={t('Class and subject')} required>
            <Select value={assignmentKey} strict options={assignments.map((row) => ({
              value: `${row.class_code}\u0000${row.subject_code}`,
              label: `${row.class_name_ar || row.class_code} · ${subjectLabel(row, state.lang)} · ${row.year_level_code}`
            }))} onChange={setAssignmentKey} />
          </Field>
          <Field className="col-12 col-lg-6" label={t('Homework title')} required>
            <Input value={title} placeholder={t('Example: Solve exercises 1–10')} onInput={setTitle} />
          </Field>
          <Field className="col-12" label={t('Teacher notes')} hint={t('These notes are also available to the parent chatbot.')}>
            <textarea className="form-control" rows="3" value={details}
              onChange={(e) => setDetails(e.target.value)} />
          </Field>
          <Field className="col-12" label={t('Attachment')}>
            <Dropzone file={file}
              accept=".pdf,.png,.jpg,.jpeg,.webp,.doc,.docx"
              label={t('Choose homework file')}
              hint={t('PDF, image or Word document · maximum 20 MB')}
              onFile={setFile} />
          </Field>
        </div>
        <ErrorNote error={teaching.error || saveError} onRetry={() => { teaching.reload(); homework.reload(); }} />
        <div className="card-body pt-0 d-flex justify-content-end">
          <Button variant="primary" icon="upload" pending={saving}
            pendingLabel={t('Publishing…')} disabled={!canPublish} onClick={publish}>
            {t('Publish homework')}
          </Button>
        </div>
      </Card>

      <Card title={t('Published homework')} tight>
        <ErrorNote error={homework.error} onRetry={homework.reload} />
        <Table loading={homework.loading} rows={(homework.value && homework.value.assignments) || []}
          rowKey={(row) => row.id}
          empty={<Empty icon="upload" title={t('No homework published yet')}>
            {t('Homework you publish appears here with its upload date.')}
          </Empty>}
          columns={[
            { key: 'date', header: t('Date'), cell: (row) => row.uploaded_on },
            { key: 'class', header: t('Class'), cell: (row) => row.class_code },
            { key: 'subject', header: t('Subject'), cell: (row) => row.subject_code },
            { key: 'title', header: t('Homework'), cell: (row) => row.title },
            { key: 'file', header: t('File'), cell: (row) =>
              row.download_url && row.original_filename
                ? <a href={row.download_url} target="_blank" rel="noreferrer">{row.original_filename}</a>
                : <span className="text-body-secondary">?</span> },
            { key: 'actions', header: '', cell: (row) =>
              <Button size="sm" variant="quiet" onClick={() => remove(row)}>{t('Delete')}</Button> }
          ]} />
      </Card>
    </div>
  </>;
}
