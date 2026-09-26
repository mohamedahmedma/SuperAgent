# SIS demo data

The `DEMO` school is fictional development data. It contains Arabic and Language
sections, 30 classes, 418 students, guardians, teachers, scoped roles, attendance,
and Term 1 grades. KG is displayed as `KG`, `KG 1`, and `KG 2` in both languages.

## Commands

Run from the repository root after applying Alembic migrations:

```powershell
.\.venv\Scripts\python.exe -m sis.demo load
.\.venv\Scripts\python.exe -m sis.demo sync
.\.venv\Scripts\python.exe -m sis.demo status
.\.venv\Scripts\python.exe -m sis.demo accounts
.\.venv\Scripts\python.exe -m sis.demo classes
```

`load` is for a database without the `DEMO` school. `sync` is non-destructive and
refreshes mutable demo labels and built-in role definitions without removing students,
attendance, grades, or assignments. `reset` intentionally removes and recreates only
the `DEMO` school and should be used only when a complete clean demo is wanted.

The commands refuse production-named environments. A non-SQLite development database
also requires the explicit `--allow-remote` acknowledgement.

## Accounts

Every account uses password `Demo#2026`.

| Username | Role and scope |
| --- | --- |
| `sysadmin` | System Administrator, global |
| `owner` | School Owner, DEMO school, read-only |
| `principal` | Principal, DEMO school |
| `supervisor.p4` | Academic Year Supervisor, Arabic Fourth Primary |
| `supervisor.g3` | Academic Year Supervisor, Language Grade 3 |
| `attendance` | Attendance Supervisor for selected P1, P4, and G3 classes |
| `t.arabic` | Arabic teacher, Arabic Primary 1–3 |
| `t.maths` | Mathematics teacher, Arabic Primary 4 and Language Grade 5 |
| `t.english` | English teacher, Language Grades 3–4 |
| `t.science` | Science teacher plus Grade 9 supervisor |
| `t.social` | Social Studies teacher plus selected-class attendance supervisor |
| `t.computer` | Computer Science teacher plus Subject Coordinator |
| `t.arabic2` | Arabic teacher, Arabic Primary 4 |
| `t.unassigned` | Teacher record assigned to Maths/P4 but deliberately without a classroom |

Use `python -m sis.demo accounts` for the exact classroom scopes and teaching
assignments generated from the live blueprint.

## Useful test paths

1. Compare the Arabic ladder (`1/1 ب`, `1/1 ع`, `1/1 ث`) with Language grades.
2. Open KG in Arabic and English and confirm only `KG`, `KG 1`, or `KG 2` is used.
3. Inspect attendance around 20 November 2025; seeded classes contain present,
   absent, late, and excused records.
4. Inspect Term 1 grades; some cells are deliberately blank to prove blank is not zero.
5. Compare `t.science` and `t.social` with ordinary teachers to exercise additive roles.
6. Use `t.unassigned` to test the supervisor classroom-assignment workflow.

The account rows and password hashes exist in the SIS database. Interactive staff-login
testing additionally requires the staff session/login API and frontend login screen to be
wired; until that phase is complete, the automated demo tests validate credentials,
roles, scopes, and relationships directly against the migrated schema.
