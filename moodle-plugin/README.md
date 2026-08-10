---
noteId: "0e4f0160949611f1bfd15975cc910763"
tags: []

---

# `local_schoolapi` — read-only student records for the school assistant

A Moodle plugin that answers one question well: **what are this student's grades this
term?** One call, one student, correct figures.

It exists because Moodle's core web services cannot answer that question safely or
quickly. Every claim below was measured on a live Moodle 5.1.6 — see
`d:/Work/moodle-dev/README.md` for the spike.

## Why not core web services

| | Core | This plugin |
| --- | --- | --- |
| Calls to read a term's grades | one per course | **one, total** |
| Exclusion (excused) flag | **not exposed at all** | `excludedcount`, and it is out of the denominator |
| Attendance: whose records come back | **the whole class** | one student |
| Attendance: capability needed to READ | `takeattendances` — **write-capable** | `local/schoolapi:read` |
| Service account must be enrolled | yes, per course | no |

The attendance row is the sharpest: to read one child's register through core, the
facade's token would need permission to *alter* attendance for every student in the
school, and would receive other families' records on every call.

## Installation

```bash
cp -r moodle-plugin/schoolapi <moodle>/public/local/schoolapi   # 5.x web root
php admin/cli/upgrade.php --non-interactive
```

Then grant `local/schoolapi:read` to the service account's role, at system context,
and add `local_schoolapi_get_student_grades` to its external service. No enrolment, no
teacher role, no write capability anywhere.

## The endpoint

```
local_schoolapi_get_student_grades(studentidnumber, term)
```

`studentidnumber` is **the school's student number**, not a Moodle user id — nothing
outside Moodle should have to know one. `term` is a course idnumber prefix such as
`2026-T1-`, matching the binding convention in `records/models.py`.

An unknown student returns `found: false` with an empty list, **not an error**. The
records facade deliberately makes "no such student", "not your child" and "records
restricted" indistinguishable to its caller; a distinct error here would hand back the
signal that design removes and let a token holder enumerate the school roll.

## The one calculation, and why it is the way it is

```
percentage = (finalgrade - rawgrademin) / (rawgrademax - rawgrademin) × 100
```

`grade_grades.rawgrademax` is the **per-student** aggregation maximum. Moodle has
already removed excluded and ungraded items from it. `grade_items.grademax` is the
course-wide total and keeps both. On the live instance, a student with 90/100, an
excluded 10/100 and one ungraded item:

| Source | Result |
| --- | --- |
| Sum of assignments — `(90+10)/200` | 50 % — counts the excluded item |
| Course item — `finalgrade / grade_items.grademax` = `90/400` | 22.5 % |
| **`finalgrade / grade_grades.rawgrademax` = `90/100`** | **90 % — correct** |

This mirrors Moodle's own user report, which overrides `$grade_item->grademax` with
`$grade_grade->get_grade_max()` before formatting — which is why the figure matches the
gradebook a teacher sees.

`grade_grades.excluded` is a **timestamp**, not a boolean — a real row reads
`1786347956`. The test is `<> 0`; `= 1` matches nothing and silently counts every
excused item.

**Null is not zero.** No graded work yet returns `percentage: null`. Zero is a mark a
child can earn; reporting "no grade yet" as 0 % tells a parent something false.

## Performance

**Two queries per student, whatever the subject count.** One for the course totals
(with the per-student bounds), one grouped query for the item counts across every
course at once. No per-course loop, no grade tree loaded, no aggregation recomputed —
Moodle already did that work when the teacher saved the mark.

Caching is MUC, keyed by **student id alone** rather than by (student, term). A
student has a couple of dozen course rows in total, so caching all terms costs nothing
— and the payoff is exact invalidation: `\core\event\user_graded` fires, the observer
deletes one key, and that child's figures are correct within milliseconds. A compound
key cannot be invalidated precisely, because the observer cannot enumerate the terms,
so it would have to purge everything — and in a school where somebody is always
grading, that cache is permanently cold. The 300s TTL is only a backstop for writes
that bypass the event, such as a direct SQL fix or a restore.

## Structure

```
classes/
  external/get_student_grades.php   transport only: contract, capability, delegate
  service/grade_service.php         the only class that knows caching exists
  repository/grade_repository.php   the only class that knows SQL
  repository/student_repository.php idnumber -> Moodle user
  dto/subject_grade.php             immutable result; owns the percentage rule
  observer.php                      cache invalidation
```

Each has one reason to change. The service depends on
`grade_repository_interface`, not on the query, so caching can be replaced without
touching SQL and the service can be tested without a database.

## Tests

`tests/subject_grade_test.php` covers the percentage rule exhaustively without a
database — including that a genuine zero reports as `0.0` while "nothing graded yet"
reports as `null`, which are the two cases most likely to be conflated and the ones a
parent would notice.

## Not built yet

Attendance. It follows the same shape — `get_student_attendance`, one student,
summarised server-side — and is the reason the plugin exists at all, since core cannot
serve it safely. Grades came first because they were verifiable against a figure the
gradebook already displays.
