import { useEffect, useMemo, useState } from 'react';

import { api } from '../api.js';
import { Store } from '../store.js';
import { labelOf, pickName, useResource, useStore } from '../hooks.js';
import { Alert, Badge, Button, Card, Empty, ErrorNote, Field, PageHead, Select, useConfirm } from '../components/Ui.jsx';
import { t } from '../i18n.js';

const ACTIONS = [
  { value: 'promote', label: 'Promote to next grade' },
  { value: 'repeat', label: 'Repeat the same grade' },
  { value: 'graduate', label: 'Graduate' },
  { value: 'exclude', label: 'Exclude from this run' }
];

function actionTone(action) {
  return { promote: 'ok', repeat: 'warn', graduate: 'info', exclude: 'neutral' }[action];
}

export function Promotions() {
  const state = useStore();
  const [sourceYear, setSourceYear] = useState('');
  const [targetYear, setTargetYear] = useState('');
  const [preview, setPreview] = useState(null);
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [makeCurrent, setMakeCurrent] = useState(true);
  const [confirmDialog, askConfirm] = useConfirm();
  const years = useResource(Store.keys.years(state.school), () => api.years(state.school), !!state.school);
  const yearList = ((years.value && years.value.academic_years) || []).slice()
    .sort((a, b) => String(a.starts_on).localeCompare(String(b.starts_on)));

  useEffect(() => {
    if (!sourceYear && state.year && yearList.some((row) => row.code === state.year)) setSourceYear(state.year);
  }, [state.year, yearList.length]);

  useEffect(() => {
    if (!sourceYear || targetYear) return;
    const source = yearList.find((row) => row.code === sourceYear);
    const next = yearList.find((row) => source && row.starts_on > source.ends_on);
    if (next) setTargetYear(next.code);
  }, [sourceYear, targetYear, yearList.length]);

  const resetPreview = (setter) => (value) => {
    setter(value); setPreview(null); setRows([]); setError(null);
  };

  const loadPreview = async () => {
    setLoading(true); setError(null);
    try {
      const result = await api.previewPromotions(sourceYear, targetYear);
      setPreview(result); setRows(result.rows || []);
    } catch (reason) { setError(reason); }
    finally { setLoading(false); }
  };

  const classOptions = (row, action = row.action) => {
    const level = action === 'repeat' ? row.source_year_level_code : row.target_year_level_code;
    return ((preview && preview.target_classes && preview.target_classes[level]) || []).map((item) => ({
      value: item.code, label: pickName(item, state.lang) || item.code
    }));
  };

  const updateAction = (studentNumber, action) => setRows((items) => items.map((row) => {
    if (row.student_number !== studentNumber) return row;
    const options = classOptions(row, action);
    return { ...row, action, target_class_code: ['promote', 'repeat'].includes(action) ? (options[0]?.value || '') : null };
  }));
  const updateClass = (studentNumber, targetClass) => setRows((items) => items.map((row) =>
    row.student_number === studentNumber ? { ...row, target_class_code: targetClass } : row
  ));

  const counts = useMemo(() => rows.reduce((out, row) => {
    out[row.action] = (out[row.action] || 0) + 1; return out;
  }, {}), [rows]);
  const invalid = rows.some((row) => ['promote', 'repeat'].includes(row.action) && !row.target_class_code);

  const commit = () => askConfirm({
    title: t('Confirm student promotion'),
    tone: 'bad',
    confirmLabel: t('Apply promotion'),
    body: <div className="vstack gap-2">
      <p className="mb-0">{t('This closes the old placements and opens the selected placements in the target year. Previous marks and attendance remain unchanged.')}</p>
      <strong>{t('{0} student(s) will be processed.', [rows.filter((row) => row.action !== 'exclude').length])}</strong>
    </div>,
    run: async () => {
      setSaving(true); setError(null);
      try {
        const result = await api.commitPromotions({
          source_year_code: sourceYear,
          target_year_code: targetYear,
          make_target_current: makeCurrent,
          rows: rows.map((row) => ({
            student_number: row.student_number,
            action: row.action,
            target_class_code: row.target_class_code || null
          }))
        });
        Store.invalidate('years:'); Store.invalidate('classes:'); Store.invalidate('roster:');
        if (makeCurrent) Store.setYear(targetYear);
        Store.toast(t('Student promotion completed.'), 'success');
        setPreview(null); setRows([]);
        return result;
      } catch (reason) { setError(reason); throw reason; }
      finally { setSaving(false); }
    }
  });

  return <>
    <PageHead title={t('Student promotion')} lede={t('Move the whole register into the next academic year with a review before anything is written.')} />
    <div className="vstack gap-4">
      <Card title={t('1. Choose academic years')} subtitle={t('Create the target year and its classes before starting.') }>
        <div className="row g-3">
          <Field className="col-12 col-md-5" label={t('Ending academic year')} required>
            <Select value={sourceYear} strict options={yearList.map((year) => ({ value: year.code, label: labelOf(year, state.lang) || year.code }))} onChange={resetPreview(setSourceYear)} />
          </Field>
          <Field className="col-12 col-md-5" label={t('New academic year')} required>
            <Select value={targetYear} strict options={yearList.filter((year) => year.code !== sourceYear).map((year) => ({ value: year.code, label: labelOf(year, state.lang) || year.code }))} onChange={resetPreview(setTargetYear)} />
          </Field>
          <div className="col-12 col-md-2 d-grid align-items-end">
            <Button variant="primary" pending={loading} disabled={!sourceYear || !targetYear || sourceYear === targetYear} onClick={loadPreview}>{t('Preview')}</Button>
          </div>
        </div>
        {error ? <div className="mt-3"><ErrorNote error={error} /></div> : null}
      </Card>

      {preview ? <Card title={t('2. Review every student')} subtitle={t('Change any suggestion before applying the promotion.')} tight>
        <div className="card-body d-flex flex-wrap gap-2" aria-live="polite">
          {ACTIONS.map((action) => <Badge key={action.value} tone={actionTone(action.value)}>{t(action.label)}: {counts[action.value] || 0}</Badge>)}
        </div>
        {!rows.length ? <Empty title={t('No students to promote')}>{t('No placement covered the last day of the selected year.')}</Empty> :
          <div className="table-responsive">
            <table className="table align-middle mb-0 sis-promotion-table">
              <thead><tr><th>{t('Student')}</th><th>{t('Current class')}</th><th>{t('Decision')}</th><th>{t('Target class')}</th></tr></thead>
              <tbody>{rows.map((row) => <tr key={row.student_number}>
                <td><strong>{pickName(row, state.lang) || row.student_number}</strong><small className="d-block text-body-secondary sis-code">{row.student_number}</small></td>
                <td><span className="sis-code">{row.source_class_code}</span><small className="d-block text-body-secondary">{row.source_year_level_code}</small></td>
                <td><Select value={row.action} strict options={ACTIONS.filter((action) =>
                  (action.value !== 'graduate' || !row.target_year_level_code) &&
                  (action.value !== 'promote' || !!row.target_year_level_code)
                ).map((action) => ({ value: action.value, label: t(action.label) }))} onChange={(value) => updateAction(row.student_number, value)} /></td>
                <td>{['promote', 'repeat'].includes(row.action) ? <Select value={row.target_class_code || ''} strict options={classOptions(row)} onChange={(value) => updateClass(row.student_number, value)} /> : <span className="text-body-secondary">—</span>}</td>
              </tr>)}</tbody>
            </table>
          </div>}
        <div className="card-body border-top d-flex flex-column flex-md-row gap-3 align-items-md-center justify-content-between">
          <label className="d-flex gap-2 align-items-center"><input type="checkbox" className="form-check-input m-0" checked={makeCurrent} onChange={(event) => setMakeCurrent(event.target.checked)} /><span>{t('Make the new year current after promotion')}</span></label>
          <Button variant="primary" pending={saving} disabled={!rows.length || invalid || saving} onClick={commit}>{t('Apply promotion')}</Button>
        </div>
      </Card> : null}
    </div>
    {confirmDialog}
  </>;
}
