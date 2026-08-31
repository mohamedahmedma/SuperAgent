/*
 * Taking the register, starting from nothing: a date, a grade, a class, a list.
 *
 * The other way into a register is through the structure — School, Year, Rung, Class, the
 * attendance tab — and for an attendance supervisor that route does not exist. Their grants
 * are four classrooms; they hold no authority over the grade or the year those rooms sit in,
 * so every listing above the room refuses them and the drill-down has no first step. This
 * screen is that first step, and it asks the service which rooms the person holds rather
 * than asking them to navigate down to one.
 *
 * Three things it is shaped around:
 *
 * **The date comes first and stays put.** A register is a statement about a day, and the day
 * is the one field a user gets wrong in a way nothing downstream can detect — Monday's
 * register typed into Tuesday looks perfectly correct. So it is the first control, it is
 * echoed in the class list, and changing it re-reads which registers are already done.
 *
 * **The grade is a grouping, not a second query.** Every class carries its grade, so the
 * picker is built by grouping one response. A supervisor holding rooms on three rungs sees
 * three groups; one holding four rooms on one rung sees one and never has to choose.
 *
 * **A register already taken says so before it is opened.** Each class shows how much of its
 * day is marked, so the common way to record a day twice — not being able to tell that
 * somebody already did — is answered on the screen where the choice is made.
 */
import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { AttendancePanel } from '../components/AttendancePanel.jsx';
import { pickName, today, useQuery, useStore } from '../hooks.js';
import { Badge, Card, Empty, ErrorNote, Input, PageHead, Select, Skeleton } from '../components/Ui.jsx';
import { t } from '../i18n.js';

export function Attendance() {
  const state = useStore();
  const [day, setDay] = useState(today());
  const [grade, setGrade] = useState('');
  const [klass, setKlass] = useState('');

  const options = useQuery(
    () => api.registerableClasses(state.year, day),
    [state.year, day],
    !!state.year
  );
  const classes = (options.value && options.value.classes) || [];

  /* Grades, in the order the service sent them — it orders by the rung's own display order,
     which is the order a school reads its ladder in and not alphabetical. */
  const grades = [];
  classes.forEach((row) => {
    if (!grades.some((item) => item.code === row.year_level_code)) {
      grades.push({
        code: row.year_level_code,
        name_en: row.year_level_name_en,
        name_ar: row.year_level_name_ar
      });
    }
  });
  const inGrade = classes.filter((row) => row.year_level_code === grade);

  /* One grade, or one class within it, is not a choice — it is an obstacle. The effect
     settles on the only option whenever there is exactly one, and clears a class that the
     new day or grade no longer offers so the panel below never renders a stale room. */
  useEffect(() => {
    if (!grades.length) return;
    if (!grades.some((item) => item.code === grade)) setGrade(grades[0].code);
  }, [options.value]);

  useEffect(() => {
    const here = classes.filter((row) => row.year_level_code === grade);
    if (!here.some((row) => row.class_code === klass)) {
      setKlass(here.length === 1 ? here[0].class_code : '');
    }
  }, [grade, options.value]);

  const chosen = classes.find((row) => row.class_code === klass);

  return (
    <>
      <PageHead
        title={t('Take attendance')}
        lede={t('Choose a day, a grade and a class. Mark the children who are here; the rest are recorded absent.')}
      />

      <div className="vstack gap-3">
        <Card title={t('1. Day, grade and class')}>
          <div className="row g-3">
            <div className="col-12 col-md-4">
              <label className="form-label small text-body-tertiary">{t('Date')}</label>
              <Input type="date" value={day} onInput={setDay} />
            </div>
            <div className="col-12 col-md-4">
              <label className="form-label small text-body-tertiary">{t('Grade')}</label>
              <Select
                value={grade}
                onChange={setGrade}
                disabled={!grades.length}
                options={grades.map((item) => ({
                  value: item.code,
                  label: pickName(item, state.lang) || item.code
                }))}
              />
            </div>
            <div className="col-12 col-md-4">
              <label className="form-label small text-body-tertiary">{t('Class')}</label>
              <Select
                value={klass}
                onChange={setKlass}
                disabled={!inGrade.length}
                options={[
                  { value: '', label: t('Choose…') },
                  ...inGrade.map((row) => ({
                    value: row.class_code,
                    label: pickName(row, state.lang) || row.class_code
                  }))
                ]}
              />
            </div>
          </div>

          {options.loading && !options.ready ? <Skeleton rows={3} /> : null}
          {options.error ? <ErrorNote error={options.error} onRetry={options.reload} /> : null}

          {/* Progress for the whole grade on the chosen day, so a supervisor can see at a
              glance which of their rooms are still to do — and which are already done. */}
          {inGrade.length ? (
            <div className="d-flex flex-wrap gap-2 mt-3">
              {inGrade.map((row) => (
                <button
                  key={row.class_code}
                  type="button"
                  className={
                    'btn btn-sm ' +
                    (row.class_code === klass ? 'btn-primary' : 'btn-outline-secondary')
                  }
                  onClick={() => setKlass(row.class_code)}
                >
                  {pickName(row, state.lang) || row.class_code}{' '}
                  {row.is_complete ? (
                    <Badge tone="ok">{t('done')}</Badge>
                  ) : (
                    <Badge>{`${row.marked}/${row.size}`}</Badge>
                  )}
                </button>
              ))}
            </div>
          ) : null}
        </Card>

        {options.ready && !classes.length ? (
          <Card>
            <Empty title={t('No classes are assigned to this account.')}>
              {t('A register is taken by whoever holds the class. Ask whoever manages roles at your school for the classes you take.')}
            </Empty>
          </Card>
        ) : null}

        {chosen ? (
          <>
            {chosen.is_complete ? (
              /* Said before the register is opened rather than refused: a completed day is
                 still editable by anybody who may write it, and the point of the notice is
                 that they know they are editing rather than starting. */
              <Card>
                <span className="small text-body-tertiary">
                  {t('This register was already taken for this day. Saving again corrects it rather than recording it twice.')}
                </span>
              </Card>
            ) : null}
            <AttendancePanel
              classCode={chosen.class_code}
              year={state.year}
              on={day}
              scope={{
                school: state.school,
                yearLevel: chosen.year_level_code,
                classSection: chosen.class_code
              }}
            />
          </>
        ) : null}
      </div>
    </>
  );
}
