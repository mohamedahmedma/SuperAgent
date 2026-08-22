/*
 * The import flow, written once and used by three screens.
 *
 * Roster, guardians and marks are three different uploads with three different form fields and
 * one identical shape after that: parse into a batch, show a per-row outcome, commit or walk
 * away. The console this replaced implemented that shape three times and the three drifted, so
 * this module is the shape and the screens supply only what genuinely differs — the fields
 * above the file, the two calls, and the template.
 *
 * The rule the whole flow protects: **nothing is written until commit.** A preview is a dry run
 * with a per-row verdict, and a registrar must be able to read every rejected row before
 * deciding. The commit button therefore never exists on arrival, is styled as the destructive
 * control it is, and states the count it is about to write.
 *
 * On a phone the row table is the hard part, and the answer is in the column definitions: the
 * line number, the outcome and the reason are always visible, and each payload column declares
 * the breakpoint below which it is not worth its width. A rejected row on a 360px screen shows
 * "line 4 · rejected · student_number is required", which is the whole of what the registrar
 * needs to go and fix the spreadsheet.
 */
import { useState } from 'react';
import { ApiError } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import { DASH, useAction } from '../hooks.js';
import { t } from '../i18n.js';
import {
  Alert,
  Badge,
  Button,
  Card,
  Chip,
  Dropzone,
  Empty,
  ErrorNote,
  Icon,
  Table
} from './Ui.jsx';

/* -- Outcome vocabulary ---------------------------------------------------------
 *
 * One table of five codes, and the wording changes with tense: in a preview these are
 * predictions ("will be created"), after a commit they are facts. A screen that says "created"
 * over a preview has told the registrar the write already happened, which is the one thing
 * this flow must never imply.
 */
export const OUTCOMES = {
  created: { label: 'created', future: 'will be created', tone: 'ok' },
  updated: { label: 'updated', future: 'will be updated', tone: 'ok' },
  unchanged: { label: 'unchanged', future: 'already correct', tone: null },
  skipped: { label: 'skipped', future: 'will be skipped', tone: 'warn' },
  rejected: { label: 'rejected', future: 'rejected', tone: 'bad' }
};

export function outcomeOf(code) {
  return OUTCOMES[code] || { label: code, future: code, tone: null };
}

export function wordFor(code, committed) {
  const entry = outcomeOf(code);
  return committed ? entry.label : entry.future;
}

/*
 * Columns worth showing first, per upload kind, and the breakpoint each earns. `student_number`
 * and the name are always there; the rest appear as the screen widens, because a phone showing
 * eight payload columns shows none of them legibly.
 */
const PREFERRED = {
  roster: ['student_number', 'full_name_ar', 'full_name_en', 'class_code', 'starts_on'],
  guardians: ['student_number', 'phone', 'full_name_ar', 'full_name_en', 'relationship'],
  grades: ['student_number', 'subject_code', 'percentage', 'points', 'max_points', 'class_code']
};

const ALWAYS_VISIBLE = new Set(['student_number', 'phone', 'subject_code', 'percentage']);

const HEADINGS = {
  student_number: 'Student no.',
  full_name_ar: 'Name (Arabic)',
  full_name_en: 'Name (English)',
  class_code: 'Class',
  starts_on: 'From',
  subject_code: 'Subject',
  max_points: 'Out of',
  phone: 'Phone',
  relationship: 'Relationship'
};

function headingFor(key) {
  return HEADINGS[key] || key.replace(/_/g, ' ').replace(/^./, (first) => first.toUpperCase());
}

/** The payload columns actually present across these rows, preferred ones first. */
function columnsFor(kind, rows) {
  const seen = new Set();
  rows.forEach((row) => Object.keys(row.payload || {}).forEach((key) => seen.add(key)));
  const preferred = (PREFERRED[kind] || []).filter((key) => seen.has(key));
  const rest = [...seen].filter((key) => !preferred.includes(key)).sort();
  return [...preferred, ...rest];
}

/**
 * One payload cell. A blank is an em dash in the faint ink and a `0` is the string "0" — the
 * same distinction the marks screens make, and it matters here for the same reason: a
 * percentage column showing nothing where a real zero was uploaded means the registrar cannot
 * tell a child who scored nothing from a row nobody has marked, and this is the screen where
 * they would have caught it.
 */
function cellText(value) {
  if (value === null || value === undefined || value === '') {
    return <span className="sis-ungraded">{DASH}</span>;
  }
  if (typeof value === 'object') {
    return <span className="font-monospace sis-xs">{JSON.stringify(value)}</span>;
  }
  return String(value);
}

/* -- Row table ------------------------------------------------------------------ */

export function RowsTable({ rows = [], kind, committed }) {
  const keys = columnsFor(kind, rows);

  const columns = [
    {
      key: 'line',
      header: t('Line'),
      className: 'sis-num',
      /* The spreadsheet line number, so a registrar can open the file and go straight to it.
         The single most-used column on the screen, and never hidden. */
      cell: (row) => <span className="font-monospace">{row.line}</span>
    },
    {
      key: 'outcome',
      header: t('Outcome'),
      cell: (row) => (
        <Badge tone={outcomeOf(row.code).tone}>{wordFor(row.code, committed)}</Badge>
      )
    },
    ...keys.map((key) => ({
      key: `payload:${key}`,
      header: headingFor(key),
      /* Everything but the identifying columns waits for room. */
      hide: ALWAYS_VISIBLE.has(key) ? undefined : 'lg',
      className:
        key === 'full_name_ar'
          ? 'sis-name-ar'
          : key === 'full_name_en'
            ? 'sis-name-en'
            : ['percentage', 'points', 'max_points'].includes(key)
              ? 'sis-num'
              : ['student_number', 'class_code', 'subject_code', 'phone'].includes(key)
                ? 'sis-code'
                : undefined,
      cell: (row) => cellText((row.payload || {})[key])
    })),
    {
      key: 'message',
      header: t('Why'),
      cell: (row) =>
        row.message ? (
          <>
            {row.message}
            {row.field ? (
              <span className="sis-code sis-xs text-body-tertiary"> ({row.field})</span>
            ) : null}
          </>
        ) : (
          <span className="text-body-tertiary">{DASH}</span>
        )
    }
  ];

  return (
    <Table
      rows={rows}
      columns={columns}
      rowKey={(row, index) => `${row.line}:${index}`}
      rowTone={(row) => outcomeOf(row.code).tone}
      empty={<Empty title={t('No rows match this filter')}>{t('Clear the filter to see the rest.')}</Empty>}
    />
  );
}

/* -- Totals, as filters ----------------------------------------------------------
 *
 * Clickable because of the case that actually happens: a twelve-hundred-row file with fourteen
 * rejected rows in it. Reading those fourteen by scrolling is how a registrar commits a batch
 * without having read them.
 */
export function Totals({ totals = {}, total, active, committed, onChange }) {
  const codes = Object.keys(OUTCOMES).filter((code) => totals[code]);
  /* Any code the service adds later still shows up, rather than silently vanishing from a
     screen whose job is to account for every row. */
  Object.keys(totals).forEach((code) => {
    if (!codes.includes(code) && totals[code]) codes.push(code);
  });
  if (!codes.length) return null;

  return (
    <div className="d-flex flex-wrap gap-2">
      <Chip active={!active} count={total} onClick={() => onChange(null)}>
        {t('all rows')}
      </Chip>
      {codes.map((code) => (
        <Chip
          key={code}
          active={active === code}
          count={totals[code]}
          onClick={() => onChange(active === code ? null : code)}
        >
          {wordFor(code, committed)}
        </Chip>
      ))}
    </div>
  );
}

/* -- Template download ----------------------------------------------------------- */

/**
 * Hand the registrar a correctly-shaped empty file, built in the browser.
 *
 * The BOM is not decoration: without it Excel opens a UTF-8 CSV as Windows-1256 and renders
 * every Arabic name as mojibake, and the registrar's first act is to "fix" the encoding by
 * saving it wrong. CRLF for the same reason.
 */
export function downloadTemplate(name, header) {
  const blob = new Blob([`﻿${header}\r\n`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  /* Revoked on a timer rather than immediately: Safari has not started reading the blob when
     click() returns, and revoking synchronously gives a zero-byte download. */
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/* -- Recent batches -------------------------------------------------------------
 *
 * localStorage, and this is the one thing in the console that belongs there: a batch id is an
 * identifier rather than a credential, and a registrar who closes the tab mid-import needs to
 * find the batch again tomorrow. The API key is the opposite case and lives in sessionStorage.
 */
const RECENT_KEY = 'sis.recent_batches';
const RECENT_MAX = 12;

export function recent() {
  try {
    const raw = localStorage.getItem(RECENT_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list.filter((item) => item && typeof item.id === 'string') : [];
  } catch (e) {
    return [];
  }
}

export function remember(id) {
  if (!id) return;
  try {
    const list = recent().filter((item) => item.id !== id);
    list.unshift({ id, at: new Date().toISOString() });
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, RECENT_MAX)));
  } catch (e) {
    /* Storage disabled: the batch id is still in the URL, which is the copy that matters. */
  }
}

/* ==================================================================================
 * The flow
 * ================================================================================== */

export function ImportFlow({
  kind,
  ready = true,
  blocker,
  fields,
  label,
  hint,
  template,
  onPreview,
  onCommit,
  invalidate = []
}) {
  const [file, setFile] = useState(null);
  const [batch, setBatch] = useState(null);
  const [committed, setCommitted] = useState(false);
  const [filter, setFilter] = useState(null);

  const preview = useAction((chosen) => onPreview(chosen));
  const commit = useAction((batchId) => onCommit(batchId));

  function choose(chosen) {
    setFile(chosen);
    /* A new file discards the old verdict. Leaving the previous preview on screen beside a new
       filename is how a registrar commits the batch they were looking at rather than the one
       they just chose. */
    setBatch(null);
    setCommitted(false);
    setFilter(null);
    preview.reset();
    commit.reset();
  }

  const rows = (batch && batch.rows) || [];
  const shown = filter ? rows.filter((row) => row.code === filter) : rows;

  return (
    <div className="vstack gap-4">
      <Card
        title={t('1. The file')}
        actions={
          template ? (
            <Button
              size="sm"
              icon="download"
              onClick={() => downloadTemplate(template.name, template.header)}
            >
              {t('Template')}
            </Button>
          ) : null
        }
      >
        <div className="vstack gap-3">
          {fields || null}
          {ready ? (
            <>
              <Dropzone file={file} label={label} hint={hint} onFile={choose} />
              <div className="d-grid gap-2 d-sm-flex align-items-sm-center">
                <Button
                  variant="primary"
                  disabled={!file}
                  pending={preview.pending}
                  pendingLabel={t('Reading the file…')}
                  onClick={() =>
                    preview
                      .run(file)
                      .then((result) => {
                        setBatch(result);
                        setCommitted(false);
                        setFilter(null);
                        /* The batch id goes in the URL so a reload does not lose it and the
                           address can be pasted to a colleague. It is an identifier, not a
                           credential. */
                        if (result && result.batch_id) {
                          Router.setParams({ batch: result.batch_id });
                          remember(result.batch_id);
                        }
                      })
                      .catch(() => {})
                  }
                >
                  {t('Preview')}
                </Button>
                <span className="small text-body-tertiary">
                  {t('Nothing is written by a preview. Every row is checked and reported first.')}
                </span>
              </div>
            </>
          ) : (
            blocker || null
          )}
          <ErrorNote error={preview.error} />
        </div>
      </Card>

      {batch ? (
        <Card
          className="sis-rise"
          title={committed ? t('2. Committed') : t('2. Preview')}
          subtitle={<span className="sis-code">{batch.batch_id}</span>}
          actions={
            <a
              className="btn btn-sm btn-outline-secondary"
              href={Router.href('batches', { batch: batch.batch_id })}
            >
              {t('Open in Batches')}
            </a>
          }
          footer={
            committed ? (
              <>
                <Icon name="check" />
                <span className="small">
                  {t('Written. A second commit of this batch is refused, which is what makes a double-clicked button safe.')}
                </span>
              </>
            ) : (
              <div className="d-grid gap-2 d-sm-flex align-items-sm-center w-100">
                <Button
                  variant="danger"
                  pending={commit.pending}
                  pendingLabel={t('Writing…')}
                  disabled={!batch.ok_count}
                  onClick={() =>
                    commit
                      .run(batch.batch_id)
                      .then((result) => {
                        setBatch(result);
                        setCommitted(true);
                        setFilter(null);
                        invalidate.forEach((prefix) => Store.invalidate(prefix));
                        Store.toast(
                          'ok',
                          'Batch committed',
                          `${result.ok_count} row(s) written, ${result.rejected_count} rejected`
                        );
                      })
                      .catch(() => {})
                  }
                >
                  Commit {batch.ok_count} row(s)
                </Button>
                <span className="small text-body-tertiary">
                  {batch.rejected_count
                    ? `${batch.rejected_count} row(s) will not be written. Read them before committing.`
                    : 'Every row passed.'}
                </span>
              </div>
            )
          }
          tight
        >
          <div className="card-body vstack gap-3">
            <div className="d-flex flex-wrap gap-3">
              <span className="small">
                <strong className="font-monospace">{batch.total_rows}</strong> {t('row(s) read')}
              </span>
              <span className="small">
                <strong className="font-monospace">{batch.ok_count}</strong>{' '}
                {committed ? t('written') : t('ready')}
              </span>
              <span className={batch.rejected_count ? 'small' : 'small text-body-tertiary'}>
                <strong className="font-monospace">{batch.rejected_count}</strong> {t('rejected')}
              </span>
            </div>

            <Totals
              totals={batch.totals}
              total={batch.total_rows}
              active={filter}
              committed={committed}
              onChange={setFilter}
            />

            {batch.rejected_count && !committed ? (
              <Alert tone="warn">
                One malformed row is one rejected row — it never discards the rows around it.
                Committing writes the {batch.ok_count} that passed and leaves the{' '}
                {batch.rejected_count} that did not.
              </Alert>
            ) : null}

            <ErrorNote error={commit.error} />
          </div>

          <RowsTable rows={shown} kind={kind} committed={committed} />

          {rows.length < batch.total_rows ? (
            <p className="small text-body-tertiary p-3 mb-0">
              Showing the first {rows.length} of {batch.total_rows} row(s).{' '}
              <a href={Router.href('batches', { batch: batch.batch_id })}>
                {t('Open the full batch in Batches')}
              </a>{' '}
              to page through the rest.
            </p>
          ) : null}
        </Card>
      ) : null}
    </div>
  );
}
