/*
 * Roster — enrol children into classes, and read the register back.
 *
 * `default_starts_on` is left empty on purpose and says why. Absent means "the first day of the
 * academic year", decided by the service; prefilling it with today would record a November
 * import as every child in the school having joined in November, and a placement is a dated
 * membership that stays true forever.
 */
import { useState } from 'react';
import { api } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import { DASH, dateText, labelOf, pickName, useQuery, useResource, useStore } from '../hooks.js';
import { Badge, Button, Card, Empty, ErrorNote, Field, Input, NoYearNotice, PageHead, Select, Table } from '../components/Ui.jsx';
import { ImportFlow } from '../components/ImportFlow.jsx';
import { t } from '../i18n.js';

const TEMPLATE = {
  name: 'roster-template.csv',
  header: t('student_number,full_name_ar,full_name_en')
};

/* -- The register ---------------------------------------------------------------- */

function Register({ year, classCode }) {
  const state = useStore();
  const [picked, setPicked] = useState(classCode || '');

  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);
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
      <div className="card-body">
        <Field label={t('Class')} hint={t('Codes are unique within an academic year.')}>
          <Select
            className="sis-code"
            value={picked}
            placeholder={t('— choose a class —')}
            options={(classes.value || []).map((section) => ({
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
          rowLabel={(row) => `Open ${pickName(row, state.lang) || row.student_number}`}
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
                <div className="d-flex gap-1">
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
  const [startsOn, setStartsOn] = useState('');

  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);

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
        label={t('Class')}
        hint={t('Optional. Leave empty if the sheet has its own class_code column.')}
      >
        <Select
          className="sis-code"
          value={classCode}
          placeholder={t('— from the file —')}
          options={(classes.value || []).map((section) => ({
            value: section.code,
            label: labelOf(section, state.lang)
          }))}
          onChange={setClassCode}
        />
      </Field>
      <Field
        className="col-12 col-sm-6 col-lg-4"
        label={t('Placements start')}
        hint={t('Optional. Empty means the first day of the academic year.')}
      >
        <Input type="date" value={startsOn} onInput={setStartsOn} />
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
          label={t('Choose the roster sheet')}
          hint={`Columns: ${TEMPLATE.header}, optionally class_code and starts_on.`}
          invalidate={['classes:']}
          fields={fields}
          onPreview={(file) => {
            const form = new FormData();
            form.append('file', file);
            form.append('academic_year_code', year);
            /* Appended only when set. An empty string is a value the parser has to reject, and
               the registrar would read the 422 as the file being wrong. */
            if (classCode) form.append('class_code', classCode);
            if (startsOn) form.append('default_starts_on', startsOn);
            return api.previewRoster(form);
          }}
          onCommit={(batchId) => api.commitRoster(batchId)}
        />

        <Register year={year} classCode={classCode || params.class} />
      </div>
    </>
  );
}
