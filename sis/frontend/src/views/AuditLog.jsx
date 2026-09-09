import { useMemo, useState } from 'react';
import { api } from '../api.js';
import { useQuery } from '../hooks.js';
import { Badge, Button, Card, Empty, ErrorNote, Field, Input, PageHead, Skeleton, Table } from '../components/Ui.jsx';
import { t } from '../i18n.js';

const PAGE_SIZE = 50;

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
  const reset = () => { setEntityType(''); setAction(''); setPage(0); };

  return <>
    <PageHead title={t('Audit Log')} subtitle={t('Append-only record of important changes.')} />
    <Card title={t('Audit Log')} subtitle={t('A read-only, newest-first record.')}>
      <div className="row g-3 align-items-end mb-3">
        <Field className="col-12 col-md-5" label={t('Resource type')}>
          <Input value={entityType} onInput={(value) => { setEntityType(value); setPage(0); }} placeholder={t('e.g. student')} />
        </Field>
        <Field className="col-12 col-md-5" label={t('Action')}>
          <Input value={action} onInput={(value) => { setAction(value); setPage(0); }} placeholder={t('e.g. updated')} />
        </Field>
        <div className="col-12 col-md-2 d-grid"><Button variant="outline" onClick={reset} disabled={!entityType && !action}>{t('Clear filters')}</Button></div>
      </div>
      {log.error ? <ErrorNote error={log.error} onRetry={log.reload} /> : null}
      {log.loading && !log.value ? <Skeleton rows={6} /> :
        <Table
          loading={log.loading}
          rows={rows}
          rowKey={(entry) => entry.id}
          empty={<Empty title={t('No audit entries')}>
            {entityType || action ? t('Try clearing one or both filters.') : t('Important changes will appear here as they are recorded.')}
          </Empty>}
          columns={[
            { key: 'when', header: t('When'), className: 'sis-code text-nowrap', cell: (entry) => String(entry.created_at).slice(0, 16).replace('T', ' ') },
            { key: 'actor', header: t('Actor'), cell: (entry) => entry.actor || <span className="sis-ungraded">—</span> },
            { key: 'action', header: t('Action'), cell: (entry) => <Badge tone="info">{entry.action.replaceAll('_', ' ')}</Badge> },
            { key: 'resource', header: t('Resource'), cell: (entry) => <><span>{entry.entity_type}</span><div className="sis-code sis-xs text-body-tertiary">{entry.entity_id}</div></> },
            { key: 'changes', header: t('Changes'), hide: 'md', cell: (entry) => <details><summary>{t('View')}</summary><pre className="small mb-0 mt-2">{JSON.stringify({ old: entry.old_values, new: entry.new_values }, null, 2)}</pre></details> }
          ]}
        />}
      <div className="card-footer d-flex flex-wrap align-items-center justify-content-between gap-2">
        <span className="small text-body-tertiary">{t('Page {0}', [page + 1])}</span>
        <div className="btn-group" role="group" aria-label={t('Audit log pages')}>
          <Button size="sm" variant="outline" disabled={page === 0 || log.loading} onClick={() => setPage((value) => value - 1)}>{t('Previous')}</Button>
          <Button size="sm" variant="outline" disabled={rows.length < PAGE_SIZE || log.loading} onClick={() => setPage((value) => value + 1)}>{t('Next')}</Button>
        </div>
      </div>
    </Card>
  </>;
}
