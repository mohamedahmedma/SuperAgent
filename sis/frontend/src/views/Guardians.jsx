/*
 * Guardians — who may be told about a child, and who may read her marks.
 *
 * The most consequential screen in the console, and the two actions on it are deliberately not
 * the same action:
 *
 *   Revoke access   the relationship is real and stays on file; this adult may no longer read
 *                   the records. A court order is this. It asks for a reason, because the
 *                   service records the reason whichever way the flag went — so a later reader
 *                   can tell a deliberate restriction from a default nobody revisited.
 *
 *   Remove          the link should never have existed: the adult was entered against the wrong
 *                   child. A typo is this. It destroys the row.
 *
 * The console makes a registrar say which one they mean, both are confirmed, and neither is one
 * tap away from the other — which matters more on a phone than on a desktop, where a mis-tap is
 * cheaper.
 */
import { useState } from 'react';
import { api } from '../api.js';
import { Router } from '../router.js';
import { DASH, pickName, useAction, useQuery, useStore } from '../hooks.js';
import {
  Badge,
  Button,
  Card,
  Empty,
  ErrorNote,
  Field,
  Input,
  PageHead,
  SearchField,
  Table,
  useConfirm
} from '../components/Ui.jsx';
import { ImportFlow } from '../components/ImportFlow.jsx';
import { Store } from '../store.js';
import { t } from '../i18n.js';

const TEMPLATE = {
  name: 'guardians-template.csv',
  header: t('student_number,phone,full_name_ar,full_name_en,relationship,is_primary_contact')
};

/* -- Access, per link ------------------------------------------------------------ */

/**
 * Revoking opens a reason box first; granting does not, because restoring a parent's sight of
 * his own child's marks is the default state and does not need justifying — but the note is
 * still sent, so the record says who restored it and why.
 */
function AccessCell({ studentNumber, guardian, onChanged }) {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState('');
  const change = useAction((allowed) =>
    api.setRecordsAccess(studentNumber, guardian.phone, {
      can_view_records: allowed,
      restriction_note: note.trim()
    })
  );

  const allowed = guardian.can_view_records;

  if (open) {
    return (
      <div className="vstack gap-2" style={{ minWidth: '14rem' }}>
        <Field
          label={allowed ? t('Why is access being revoked?') : t('Why is access being restored?')}
          hint={t('Recorded on the link either way.')}
        >
          <Input
            value={note}
            placeholder={allowed ? t('Court order dated…') : t('Order lifted…')}
            onInput={setNote}
          />
        </Field>
        <ErrorNote error={change.error} />
        <div className="d-grid gap-2 d-sm-flex">
          <Button
            size="sm"
            variant={allowed ? 'danger' : 'primary'}
            pending={change.pending}
            pendingLabel={t('Saving…')}
            onClick={() =>
              change
                .run(!allowed)
                .then(() => {
                  setOpen(false);
                  setNote('');
                  Store.toast(
                    !allowed ? 'ok' : 'bad',
                    !allowed ? 'Access granted' : 'Access revoked',
                    guardian.phone
                  );
                  onChanged();
                })
                .catch(() => {})
            }
          >
            {allowed ? t('Revoke access') : t('Grant access')}
          </Button>
          <Button size="sm" variant="quiet" onClick={() => setOpen(false)}>
            {t('Cancel')}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="d-flex flex-wrap align-items-center gap-2">
      {allowed ? (
        <Badge tone="ok">{t('may read records')}</Badge>
      ) : (
        <Badge tone="bad">{t('restricted')}</Badge>
      )}
      <Button size="sm" variant="quiet" onClick={() => setOpen(true)}>
        {allowed ? t('Revoke') : t('Grant')}
      </Button>
    </div>
  );
}

/** Remove the link entirely. Confirmed, and the dialog says what it destroys. */
function RemoveCell({ studentNumber, guardian, onChanged }) {
  const [dialog, ask] = useConfirm();

  return (
    <>
      {dialog}
      <Button
        size="sm"
        variant="quiet"
        title={t('Remove this adult from this child')}
        onClick={() =>
          ask({
            title: `Remove ${guardian.phone}?`,
            tone: 'bad',
            confirmLabel: 'Yes, remove',
            body: (
              <p className="mb-0">
                {t('Remove')} <span className="sis-code">{guardian.phone}</span> from this child? Use{' '}
                <strong>{t('Revoke')}</strong> {t('instead if the relationship is real and only the access is wrong.')}
              </p>
            ),
            run: () =>
              api.unlinkGuardian(studentNumber, guardian.phone).then(() => {
                  Store.toast('ok', t('Guardian removed'), guardian.phone);
                onChanged();
              })
          })
        }
      >
        {t('Remove')}
      </Button>
    </>
  );
}

/* -- Look up one child ----------------------------------------------------------- */

function StudentLookup({ initial }) {
  const state = useStore();
  const [typed, setTyped] = useState(initial || '');
  const [asked, setAsked] = useState(initial || '');

  const result = useQuery(() => api.studentGuardians(asked), [asked], !!asked);
  const guardians = (result.value && result.value.guardians) || [];

  return (
    <Card
      title={t('Who may ask about one child')}
      subtitle={result.value ? t('{0} guardian(s) on file', [result.value.count]) : null}
      actions={
        asked ? (
          <Button size="sm" icon="refresh" onClick={result.reload}>
            {t('Reload')}
          </Button>
        ) : null
      }
      tight
    >
      <div className="card-body">
        <form
          className="row g-2 align-items-end"
          onSubmit={(event) => {
            event.preventDefault();
            const value = typed.trim();
            setAsked(value);
            Router.setParams({ student: value || null });
          }}
        >
          <Field className="col-12 col-sm-8" label={t('Student number')}>
            <SearchField className="sis-code" value={typed} placeholder="10432" onInput={setTyped} />
          </Field>
          <div className="col-12 col-sm-4 d-grid">
            <Button type="submit" variant="primary" icon="search" disabled={!typed.trim()}>
              {t('Look up')}
            </Button>
          </div>
        </form>
      </div>

      <ErrorNote error={result.error} onRetry={result.reload} />

      {!asked ? (
        <Empty title={t('No child looked up')}>
          {t('Type a student number to see the adults on file for her, and which of them may read her records.')}
        </Empty>
      ) : (
        <Table
          loading={result.loading}
          rows={guardians}
          rowKey={(row) => row.phone}
          rowTone={(row) => (row.can_view_records ? null : 'bad')}
          empty={
            <Empty title={t('No guardians on file')}>
              {t('Nobody is recorded for this child. Upload a guardians sheet above to link one.')}
            </Empty>
          }
          columns={[
            {
              key: 'phone',
              header: t('Phone'),
              className: 'sis-code',
              cell: (row) => (
                <>
                  {row.phone}
                  {row.is_primary_contact ? (
                    <>
                      <br />
                      <Badge tone="info">{t('primary')}</Badge>
                    </>
                  ) : null}
                  {(row.phones || []).length > 1 ? (
                    <div className="sis-xs text-body-tertiary font-monospace">
                      +{row.phones.length - 1} more number(s)
                    </div>
                  ) : null}
                </>
              )
            },
            {
              key: 'name',
              header: t('Name'),
              className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
              hide: 'sm',
              cell: (row) =>
                pickName(row, state.lang) || <span className="sis-ungraded">{DASH}</span>
            },
            {
              key: 'relationship',
              header: t('Relationship'),
              hide: 'lg',
              cell: (row) => row.relationship_label || row.relationship_type
            },
            {
              key: 'access',
              header: t('Records access'),
              cell: (row) => (
                <div className="vstack gap-1">
                  <AccessCell
                    studentNumber={asked}
                    guardian={row}
                    onChanged={result.reload}
                  />
                  {row.restriction_note ? (
                    <span className="sis-xs text-body-tertiary">{row.restriction_note}</span>
                  ) : null}
                </div>
              )
            },
            {
              key: 'remove',
              header: '',
              cell: (row) => (
                <RemoveCell studentNumber={asked} guardian={row} onChanged={result.reload} />
              )
            }
          ]}
        />
      )}
    </Card>
  );
}

/* -- Look up one number ---------------------------------------------------------- */

/**
 * The reverse question, and the one a parent asks: this number is on the phone, which children
 * may I discuss with them.
 *
 * `include_restricted` is off by default and labelled as what it is. The default answer is the
 * one that is safe to read aloud; seeing the restricted rows is a deliberate act by somebody who
 * has decided they are entitled to.
 */
function PhoneLookup() {
  const state = useStore();
  const [typed, setTyped] = useState('');
  const [asked, setAsked] = useState('');
  const [includeRestricted, setIncludeRestricted] = useState(false);

  const result = useQuery(
    () => api.guardianChildren(asked, includeRestricted),
    [asked, includeRestricted],
    !!asked
  );
  const children = (result.value && result.value.students) || [];

  return (
    <Card
      title={t('Which children one number may ask about')}
      subtitle={result.value ? t('{0} child(ren)', [result.value.count]) : null}
      tight
    >
      <div className="card-body vstack gap-3">
        <form
          className="row g-2 align-items-end"
          onSubmit={(event) => {
            event.preventDefault();
            setAsked(typed.trim());
          }}
        >
          <Field
            className="col-12 col-sm-8"
            label={t('Phone number')}
            hint={t("As stored. A national number is matched against the school's dialling code.")}
          >
            <Input
              className="sis-code"
              value={typed}
              placeholder="+201001234567"
              onInput={setTyped}
            />
          </Field>
          <div className="col-12 col-sm-4 d-grid">
            <Button type="submit" variant="primary" icon="search" disabled={!typed.trim()}>
              {t('Look up')}
            </Button>
          </div>
        </form>

        <div className="form-check">
          <input
            className="form-check-input"
            type="checkbox"
            id="include-restricted"
            checked={includeRestricted}
            onChange={(event) => setIncludeRestricted(event.target.checked)}
          />
          <label className="form-check-label small" htmlFor="include-restricted">
            {t('Include children this number may')} <strong>{t('not')}</strong> {t('read the records of.')}
          </label>
        </div>
      </div>

      <ErrorNote error={result.error} onRetry={result.reload} />

      {!asked ? (
        <Empty title={t('No number looked up')}>{t('Type the number that is calling.')}</Empty>
      ) : (
        <>
          {result.value && pickName(result.value, state.lang) ? (
            <p className="small px-3 mb-2">
              On file as{' '}
              <strong className={state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en'}>
                {pickName(result.value, state.lang)}
              </strong>
              .
            </p>
          ) : null}
          <Table
            loading={result.loading}
            rows={children}
            rowKey={(row) => row.student_number}
            rowTone={(row) => (row.can_view_records ? null : 'bad')}
            empty={
              <Empty title={t('No children for this number')}>
                {t('Either the number is not on file, or every child it is linked to is restricted from it.')}
              </Empty>
            }
            columns={[
              {
                key: 'number',
                header: t('Student no.'),
                className: 'sis-code',
                cell: (row) => row.student_number
              },
              {
                key: 'name',
                header: t('Name'),
                className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
                cell: (row) =>
                  pickName(row, state.lang) || <span className="sis-ungraded">{DASH}</span>
              },
              {
                key: 'relationship',
                header: t('Relationship'),
                hide: 'md',
                cell: (row) => row.relationship_label || row.relationship_type
              },
              {
                key: 'access',
                header: t('Records'),
                cell: (row) =>
                  row.can_view_records ? (
                    <Badge tone="ok">{t('may read')}</Badge>
                  ) : (
                    <Badge tone="bad">{t('restricted')}</Badge>
                  )
              },
              {
                key: 'open',
                header: '',
                cell: (row) => (
                  <a
                    className="btn btn-sm btn-quiet"
                    href={Router.href('marks', { student: row.student_number })}
                  >
                    {t('Marks')}
                  </a>
                )
              }
            ]}
          />
        </>
      )}
    </Card>
  );
}

/* -- Screen ---------------------------------------------------------------------- */

export function Guardians({ params = {} }) {
  return (
    <>
      <PageHead
        title={t('Guardians')}
        lede={t('The adults on file for each child, and which of them may be told what she scored.')}
      />

      <div className="vstack gap-4">
        <ImportFlow
          kind="guardians"
          template={TEMPLATE}
          label={t('Choose the guardians sheet')}
          hint={`Columns: ${TEMPLATE.header}.`}
          onPreview={(file) => {
            const form = new FormData();
            form.append('file', file);
            /* No year and no class: a guardian belongs to a child, not to a placement, and the
               sheet carries the student numbers it links to. */
            return api.previewGuardians(form);
          }}
          onCommit={(batchId) => api.commitGuardians(batchId)}
        />

        <StudentLookup initial={params.student} />
        <PhoneLookup />
      </div>
    </>
  );
}
