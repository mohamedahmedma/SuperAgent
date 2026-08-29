# SIS Rework — Implementation Checklist

Living document. Updated after every completed phase. Nothing here replaces the
existing system; every item is additive or a targeted edit.

## Phases

- [x] **1. Analyse the existing project** — frameworks, DB, auth, roles, UI, models.
- [x] **2. UI foundations** — Light/Dark only (drop `system` + the tint palette),
      professional palette, layout/table/form/button consistency.
- [x] **3. Arabic / English** — complete the `ar` table, verify RTL, stage naming.
- [ ] **4. Flexible educational structure** — educational systems (Arabic / Language),
      stages incl. KG, structured grade + class records.
- [ ] **5. Grade & class naming engine** — parse `1/1 ب`, generate display names
      per language from structured fields.
- [ ] **6. RBAC core** — users, roles, permissions, user_roles; staff authentication
      alongside the existing API-key machine auth.
- [ ] **7. System Administrator** — the operator account, system status
      (active / maintenance / paused).
- [ ] **8. School Owner + Principal** — read-only school scoping; principal grants roles.
- [ ] **9. Academic Year Supervisor** — year scoping, teacher→class assignment.
- [ ] **10. Attendance Supervisor** — scoped fast register (present-only marking).
- [ ] **11. Teacher** — scoped classes/subjects/students, grade entry;
      teacher↔subject and teacher↔year assignment.
- [ ] **12. Testing & hardening** — permission boundaries, ar/en, light/dark,
      performance, regression suite green.

## Invariants held throughout

* Alembic owns the schema. No `create_all`, no hand-edited tables.
* `sis/domain/` imports no SQLAlchemy and no config.
* `sis/application/services/` never reads the environment.
* One HTTP client in the console (`api.js`); no other file talks to the network.
* Every UI path must match a real route (`tests/sis/test_ui_contract.py`).
* Codes are immutable; labels are not. Every nameable thing carries `name_en` + `name_ar`.
* A blank grade is `null`, never `0`. An unmarked register entry is never "present".
