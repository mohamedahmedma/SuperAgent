import { api } from '../api.js';
import { useQuery } from '../hooks.js';
import { Card, Empty, ErrorNote, PageHead, Skeleton } from '../components/Ui.jsx';
import { t } from '../i18n.js';

export function AuditLog() {
  const log = useQuery(() => api.auditLog({ limit: 100 }), []);
  return <>
    <PageHead title={t('Audit Log')} subtitle={t('Append-only record of important changes.')} />
    <Card>
      {log.loading ? <Skeleton lines={6} /> : log.error ? <ErrorNote error={log.error} /> :
        !log.value?.length ? <Empty title={t('No audit entries')} /> :
        <div className="table-responsive"><table className="table align-middle mb-0">
          <thead><tr><th>{t('When')}</th><th>{t('Actor')}</th><th>{t('Action')}</th><th>{t('Resource')}</th><th>{t('Changes')}</th></tr></thead>
          <tbody>{log.value.map((entry) => <tr key={entry.id}>
            <td className="text-nowrap">{String(entry.created_at).slice(0, 16).replace('T', ' ')}</td>
            <td>{entry.actor}</td><td>{entry.action}</td><td>{entry.entity_type} #{entry.entity_id}</td>
            <td><details><summary>{t('View')}</summary><pre className="small mb-0">{JSON.stringify({ old: entry.old_values, new: entry.new_values }, null, 2)}</pre></details></td>
          </tr>)}</tbody>
        </table></div>}
    </Card>
  </>;
}
