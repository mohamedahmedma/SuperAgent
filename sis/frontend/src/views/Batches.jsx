/*
 * Batches — the audit screen: what an upload did, row by row, after the fact.
 *
 * Exists for the twelve-hundred-row file with fourteen rejected rows in it. The outcome filter
 * is a server-side query (`?outcome=rejected`), not a filter over a page already fetched, so
 * reading those fourteen costs one request instead of paging through twelve hundred — which is
 * the difference between a registrar reading them and committing without having read them.
 */
import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Router } from '../router.js';
import { DASH, useQuery } from '../hooks.js';
import {
  Alert,
  Badge,
  Button,
  Card,
  Chip,
  Empty,
  ErrorNote,
  Field,
  PageHead,
  Pagination,
  SearchField,
  Skeleton
} from '../components/Ui.jsx';
import { RowsTable, Totals, recent, remember } from '../components/ImportFlow.jsx';
import { t } from '../i18n.js';

const PAGE = 50;

const STATUS_TONE = { previewed: 'info', committed: 'ok', expired: 'warn' };

/* ISO, trimmed to the minute. Not localised, for the reason dates are not localised anywhere in
   this console: a d/m versus m/d guess on a school record is a real hazard. */
function stamp(value) {
  return value ? String(value).slice(0, 16).replace('T', ' ') : DASH;
}

export function Batches({ params = {} }) {
  const [typed, setTyped] = useState(params.batch || '');
  const [asked, setAsked] = useState(params.batch || '');
  const [filter, setFilter] = useState(null);
  const [offset, setOffset] = useState(0);

  /*
   * A route parameter arriving from elsewhere — the "Open in Batches" link on a preview — has to
   * move the input as well as the query, otherwise the field shows the previous id while the
   * table shows the new one.
   */
  useEffect(() => {
    if (params.batch && params.batch !== asked) {
      setTyped(params.batch);
      setAsked(params.batch);
      setFilter(null);
      setOffset(0);
    }
  }, [params.batch]);

  const report = useQuery(
    () =>
      api.importReport(asked, {
        limit: PAGE,
        offset,
        outcome: filter ? [filter] : null
      }),
    [asked, offset, filter],
    !!asked
  );

  const batch = report.value;
  const history = recent();

  function submit(event) {
    event.preventDefault();
    const value = typed.trim();
    setAsked(value);
    setFilter(null);
    setOffset(0);
    Router.setParams({ batch: value || null });
    if (value) remember(value);
  }

  return (
    <>
      <PageHead
        title={t('Batches')}
        lede={t('What an upload did, row by row. Filter by outcome to read the rejected rows of a large file without fetching the rest.')}
      />

      <div className="vstack gap-4">
        <Card title={t('Find a batch')}>
          <div className="vstack gap-3">
            <form className="row g-2 align-items-end" onSubmit={submit}>
              <Field
                className="col-12 col-sm-8 col-lg-6"
                label={t('Batch id')}
                hint={t('Shown after every preview and every commit.')}
              >
                <SearchField className="sis-code" value={typed} onInput={setTyped} />
              </Field>
              <div className="col-12 col-sm-4 col-lg-3 d-grid">
                <Button type="submit" variant="primary" icon="search" disabled={!typed.trim()}>
                  {t('Open')}
                </Button>
              </div>
            </form>

            {history.length ? (
              <div className="vstack gap-2">
                <span className="sis-tile-label">{t('Recent on this machine')}</span>
                <div className="d-flex flex-wrap gap-2">
                  {history.map((item) => (
                    <Chip
                      key={item.id}
                      active={item.id === asked}
                      onClick={() => {
                        setTyped(item.id);
                        setAsked(item.id);
                        setFilter(null);
                        setOffset(0);
                        Router.setParams({ batch: item.id });
                      }}
                    >
                      <span className="font-monospace sis-xs">{item.id}</span>
                    </Chip>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        </Card>

        {!asked ? (
          <Card>
            <Empty title={t('No batch open')}>
              {t('Paste a batch id above, or open one from a preview on the Roster, Guardians or Marks screen.')}
            </Empty>
          </Card>
        ) : (
          <Card
            className="sis-rise"
            title={`Batch ${asked}`}
            subtitle={
              batch ? `${batch.kind} — uploaded by ${batch.actor}` : report.loading ? 'Loading…' : null
            }
            actions={
              <Button size="sm" icon="refresh" onClick={report.reload}>
                {t('Reload')}
              </Button>
            }
            tight
          >
            <ErrorNote error={report.error} onRetry={report.reload} />

            {batch ? (
              <>
                <div className="card-body vstack gap-3">
                  <div className="row row-cols-2 row-cols-md-4 g-3">
                    <div className="col">
                      <div className="sis-tile-label">{t('Status')}</div>
                      <Badge tone={STATUS_TONE[batch.status]}>{batch.status}</Badge>
                    </div>
                    <div className="col">
                      <div className="sis-tile-label">{t('Uploaded')}</div>
                      <span className="font-monospace small">{stamp(batch.created_at)}</span>
                    </div>
                    <div className="col">
                      <div className="sis-tile-label">
                        {batch.committed_at ? 'Committed' : 'Expires'}
                      </div>
                      <span className="font-monospace small">
                        {stamp(batch.committed_at || batch.expires_at)}
                      </span>
                    </div>
                    <div className="col">
                      <div className="sis-tile-label">{t('Rows matching')}</div>
                      <span className="font-monospace small">{batch.total}</span>
                    </div>
                  </div>

                  {batch.status === 'expired' ? (
                    <Alert tone="warn" title={t('This preview has expired')}>
                      {t('Nothing was written. The report below is still readable, but the batch can no longer be committed — upload the file again to get a fresh preview.')}
                    </Alert>
                  ) : null}

                  <Totals
                    totals={batch.counts}
                    total={batch.total}
                    active={filter}
                    committed={batch.status === 'committed'}
                    onChange={(next) => {
                      setFilter(next);
                      /* Back to the first page: page 4 of "rejected" is almost never where the
                         four rejected rows are. */
                      setOffset(0);
                    }}
                  />
                </div>

                <RowsTable
                  rows={batch.rows || []}
                  kind={batch.kind}
                  committed={batch.status === 'committed'}
                />

                <Pagination
                  total={batch.total}
                  limit={batch.limit || PAGE}
                  offset={batch.offset || 0}
                  onChange={setOffset}
                />
              </>
            ) : (
              <Skeleton rows={6} />
            )}
          </Card>
        )}
      </div>
    </>
  );
}
