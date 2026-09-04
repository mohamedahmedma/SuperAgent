/*
 * Level — one rung of one school's ladder, and the classes standing on it this year.
 *
 * The screen with the view toggle: the classes render either as a table or as a row of tabs, and
 * the choice is remembered. They are not the same information twice. The table is for reading —
 * codes, names, capacity — and is what a registrar wants when deciding where to put a child. The
 * tabs are for working: one tap switches the register below without leaving the screen, which is
 * what a form teacher wants when marking three classes in a row. Neither is a better default for
 * both jobs, which is why the toggle exists rather than a decision baked in.
 *
 * A rung and a class belong to different scopes, and that is the thing to keep straight: a rung
 * belongs to the *school* and outlives every year, while a class belongs to an academic *year*.
 * So this screen needs both — the rung from the URL, the year from the header — and says so when
 * either is missing rather than rendering an empty list.
 */
import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Router } from '../router.js';
import { Store } from '../store.js';
import { DASH, dateText, labelOf, pickName, useAction, useForm, useQuery, useResource, useStore } from '../hooks.js';
import { t } from '../i18n.js';
import {
  Alert,
  Button,
  Card,
  Empty,
  ErrorNote,
  Field,
  Input,
  NoYearNotice,
  PageHead,
  Breadcrumbs,
  Table,
  cx
} from '../components/Ui.jsx';

const VIEW_KEY = 'sis.class_view';

/** The remembered view. A layout preference, not a credential and not a secret. */
function storedView() {
  try {
    return localStorage.getItem(VIEW_KEY) === 'tabs' ? 'tabs' : 'table';
  } catch (e) {
    return 'table';
  }
}

/* -- Add a class to this rung ----------------------------------------------------- */

function ClassForm({ year, level, onSaved }) {
  const form = useForm({ code: '', name_en: '', name_ar: '', capacity: '' });
  const save = useAction(() =>
    api.createClassSection({
      code: form.values.code.trim(),
      academic_year_code: year,
      year_level_code: level,
      name_en: form.values.name_en.trim(),
      name_ar: form.values.name_ar.trim(),
      /* Empty is "nobody has stated a capacity", which is not zero. A zero typed deliberately
         is a class that admits nobody, and the service keeps the difference. */
      capacity: form.values.capacity === '' ? null : Number(form.values.capacity)
    })
  );

  return (
    <form
      className="vstack gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        save
          .run()
          .then((section) => {
            Store.invalidate('classes:');
            Store.toast('ok', `Class ${section.code} saved`, year);
            form.reset();
            if (onSaved) onSaved(section);
          })
          .catch(() => {});
      }}
    >
      <div className="row g-3">
        <Field
          className="col-12 col-sm-6 col-lg-3"
          label={t('Class code')}
          required
          hint={`Unique within ${year}.`}
          error={form.errorFor(save.error, 'code')}
        >
          <Input
            className="sis-code"
            value={form.values.code}
            placeholder={`${level}A`}
            onInput={form.set('code')}
          />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-3"
          label={t('Name (English)')}
          error={form.errorFor(save.error, 'name_en')}
        >
          <Input value={form.values.name_en} onInput={form.set('name_en')} />
        </Field>
        <Field className="col-12 col-sm-6 col-lg-3" label={t('Name (Arabic)')}>
          <Input className="sis-name-ar" value={form.values.name_ar} onInput={form.set('name_ar')} />
        </Field>
        <Field
          className="col-12 col-sm-6 col-lg-3"
          label={t('Capacity')}
          hint={t('Optional. Empty is not the same as 0.')}
        >
          <Input
            type="number"
            inputMode="numeric"
            value={form.values.capacity}
            onInput={form.set('capacity')}
          />
        </Field>
      </div>
      <ErrorNote error={save.error} />
      <div className="d-grid d-sm-block">
        <Button type="submit" variant="primary" pending={save.pending} pendingLabel={t('Saving…')}>
          Add class to {level}
        </Button>
      </div>
    </form>
  );
}

/* -- The register shown under the tabs -------------------------------------------
 *
 * Read-only on purpose. Editing a child belongs on her class screen, where the actions that go
 * with it — attendance, marks, moving her — are all in one place; a second set of edit controls
 * here would mean two screens that can disagree about what a class contains.
 */

function TabRegister({ classCode, year }) {
  const state = useStore();
  const register = useQuery(
    () => api.classRoster(classCode, year),
    [classCode, year],
    !!(classCode && year)
  );
  const students = (register.value && register.value.students) || [];

  return (
    <Card
      title={classCode}
      subtitle={
        register.value
          ? `${register.value.count} on the register as of ${dateText(register.value.as_of)}`
          : null
      }
      actions={
        <a
          className="btn btn-sm btn-primary"
          href={Router.href('class', { code: classCode, year })}
        >
          {t('Open the class')}
        </a>
      }
      tight
    >
      <ErrorNote error={register.error} onRetry={register.reload} />
      <Table
        loading={register.loading}
        rows={students}
        rowKey={(row) => row.student_number}
        rowHref={(row) => Router.href('student', { number: row.student_number })}
        rowLabel={(row) => t('Open {0}', [pickName(row, state.lang) || row.student_number]).join('')}
        empty={
          <Empty title={t('Nobody is in this class yet')}>
            {t('Open the class to add a child, or upload a roster.')}
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
              pickName(row, state.lang) || (
                <span className="sis-ungraded">{DASH} name not on file</span>
              )
          },
          {
            key: 'from',
            header: t('Since'),
            className: 'sis-num',
            hide: 'md',
            cell: (row) => <span className="font-monospace small">{dateText(row.starts_on)}</span>
          }
        ]}
      />
    </Card>
  );
}

/* -- Screen ---------------------------------------------------------------------- */

export function Level({ params = {} }) {
  const state = useStore();
  const school = params.school || state.school;
  const levelCode = params.code || '';
  const year = params.year || state.year;

  const [view, setView] = useState(storedView);
  const [picked, setPicked] = useState(params.klass || '');
  const [adding, setAdding] = useState(false);

  const levels = useResource(Store.keys.levels(school), () => api.schoolLevels(school), !!school);
  const classes = useResource(Store.keys.classes(year), () => api.classes(year), !!year);

  const level = (levels.value || []).find((item) => item.code === levelCode);

  /* This rung's classes in this year. Filtered from the year's classes, which are already
     cached for every other screen — one request instead of a second route answering the same
     question. */
  const onThisRung = (classes.value || []).filter(
    (section) => section.year_level_code === levelCode
  );

  /* Settle the tab selection on the first class once they arrive, so the tabs view is never a
     row of tabs with nothing under it. */
  useEffect(() => {
    if (view !== 'tabs') return;
    if (!onThisRung.some((section) => section.code === picked) && onThisRung.length) {
      setPicked(onThisRung[0].code);
    }
  }, [view, onThisRung.length, picked]);

  function toggle(next) {
    setView(next);
    try {
      localStorage.setItem(VIEW_KEY, next);
    } catch (e) {
      /* Storage disabled: the toggle still works for this session. */
    }
  }

  if (!levelCode || !school) {
    return (
      <>
        <PageHead title={t('Rung')} />
        <Alert tone="warn" title={t('No rung chosen')}>
          {t('Open one from a school.')} <a href={Router.href('school')}>{t('Go to the school')}</a>.
        </Alert>
      </>
    );
  }

  const trail = [
    { label: 'Schools', to: 'school' },
    { label: school, to: 'school', params: { code: school } },
    { label: levelCode }
  ];

  if (!year) {
    return (
      <>
        <Breadcrumbs trail={trail} />
        <PageHead title={levelCode} />
        <NoYearNotice />
      </>
    );
  }

  return (
    <>
      <Breadcrumbs trail={trail} />
      <PageHead
        title={level ? `${levelCode} — ${pickName(level, state.lang)}` : levelCode}
        lede={t('Classes on this rung in {0}. A rung belongs to the school and outlives every year; a class belongs to the year.', [year])}
        actions={
          <>
            <div className="sis-segmented" role="group" aria-label={t('How to show the classes')}>
              <button
                className={cx('btn', view === 'table' ? 'btn-primary' : 'btn-outline-secondary')}
                aria-pressed={view === 'table'}
                onClick={() => toggle('table')}
              >
                {t('Table')}
              </button>
              <button
                className={cx('btn', view === 'tabs' ? 'btn-primary' : 'btn-outline-secondary')}
                aria-pressed={view === 'tabs'}
                onClick={() => toggle('tabs')}
              >
                {t('Tabs')}
              </button>
            </div>
            <Button variant="primary" onClick={() => setAdding(!adding)}>
              {adding ? t('Close') : t('Add class')}
            </Button>
          </>
        }
      />

      {level && level.stage && level.stage !== 'unspecified' ? (
        <p className="small text-body-tertiary">
          {t('In the')} <strong>{level.stage}</strong> {t('division.')}
        </p>
      ) : null}

      <div className="vstack gap-4">
        {adding ? (
          <Card className="sis-rise" title={t('New class on {0}', [levelCode])}>
            <ClassForm year={year} level={levelCode} onSaved={() => setAdding(false)} />
          </Card>
        ) : null}

        {classes.ready && !onThisRung.length ? (
          <Card>
            <Empty title={t('No classes on {0} in {1}', [levelCode, year])}>
              {t("Add one above, or open the academic year and generate the whole ladder's classes at once.")}
            </Empty>
          </Card>
        ) : view === 'table' ? (
          <Card
            className="sis-fade"
            title={t('Classes')}
            subtitle={t('{0} on this rung', [onThisRung.length])}
            tight
          >
            <Table
              loading={classes.loading}
              rows={onThisRung}
              rowKey={(row) => row.code}
              rowHref={(row) => Router.href('class', { code: row.code, year })}
              rowLabel={(row) => t('Open class {0}', [row.code]).join('')}
              columns={[
                {
                  key: 'code',
                  header: t('Class'),
                  className: 'sis-code',
                  cell: (row) => (
                    <a
                      className="sis-plain"
                      href={Router.href('class', { code: row.code, year })}
                    >
                      {row.code}
                    </a>
                  )
                },
                {
                  key: 'name',
                  header: t('Name'),
                  className: state.lang === 'ar' ? 'sis-name-ar' : 'sis-name-en',
                  hide: 'sm',
                  cell: (row) =>
                    pickName(row, state.lang) || <span className="text-body-tertiary">{DASH}</span>
                },
                {
                  key: 'capacity',
                  header: t('Capacity'),
                  className: 'sis-num',
                  hide: 'md',
                  cell: (row) =>
                    /* Not `capacity || DASH`: a capacity of 0 is a class that admits nobody,
                       which a registrar can legitimately mean. */
                    row.capacity === null || row.capacity === undefined ? (
                      <span className="sis-ungraded">{DASH}</span>
                    ) : (
                      row.capacity
                    )
                },
                {
                  key: 'open',
                  header: '',
                  cell: (row) => (
                    <div className="d-flex gap-1">
                      <a
                        className="btn btn-sm btn-outline-secondary"
                        href={Router.href('class', { code: row.code, year })}
                      >
                        {t('Register')}
                      </a>
                      <a
                        className="btn btn-sm btn-quiet d-none d-md-inline-flex"
                        href={Router.href('class', { code: row.code, year, tab: 'attendance' })}
                      >
                        {t('Attendance')}
                      </a>
                    </div>
                  )
                }
              ]}
            />
          </Card>
        ) : (
          <div className="sis-fade">
            <div className="d-flex flex-wrap gap-2 mb-3">
              {onThisRung.map((section) => (
                <button
                  key={section.code}
                  className={cx(
                    'btn',
                    section.code === picked ? 'btn-primary' : 'btn-outline-secondary'
                  )}
                  aria-current={section.code === picked ? 'true' : undefined}
                  onClick={() => {
                    setPicked(section.code);
                    Router.setParams({ klass: section.code });
                  }}
                >
                  <span className="fw-semibold">{section.code}</span>
                </button>
              ))}
            </div>
            {picked ? (
              <TabRegister classCode={picked} year={year} />
            ) : (
              <Card>
                <Empty title={t('No class selected')}>{t('Pick one above.')}</Empty>
              </Card>
            )}
          </div>
        )}
      </div>
    </>
  );
}
