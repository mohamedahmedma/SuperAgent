"""Stage 9: several roles at once, each bounded to a scope, enforced over HTTP.

The claims this file exists to hold down, in the order they matter.

**A person is not one role.** The whole point of the model is that a teacher can also be
an attendance supervisor, and that granting the second does not take the first away. Every
design that gets this wrong gets it wrong the same way — a `users.role` column, or an
endpoint shaped "set this person's role" — so the tests are written against the union
rather than against any single answer.

**A grant is bounded, and the boundary bites.** A teacher of `P1A` holds `grades.read`,
and a route that lets him read `P1B` has enforced nothing. Half these tests are the
negative: the permission is held, the request is refused anyway, because it was asked
about the wrong room.

**Wider scopes cover narrower ones.** A supervisor granted a *grade* answers for every
class on it, which is the whole reason the ladder exists rather than a flat list of
classes. That has to be tested through a route, because it depends on the route resolving
a class code up to its rung — the piece a plain RBAC table has no way to supply.

**Nobody can grant themselves more than they have.** `roles.assign` would otherwise be the
only permission worth stealing.

The existing integration door is untouched by all of this, and `test_authentication.py`
still proves it. These tests sign in as people; that suite calls with no credential at all,
and both must keep passing.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from sis.application.services.access import (
    catalogue_fingerprint,
    ensure_catalogue,
    sync_builtin_rbac,
)
from sis.demo.seeder import sync_roles
from sis.domain.rbac import BUILT_IN_ROLES, Permission, RoleCode, ScopeType
from sis.infrastructure.crypto import hash_password
from sis.infrastructure.db import models as m
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork
from tests.sis.test_timetable_api import SCHOOL, TERM, YEAR, registrar, school

PASSWORD = "password9-not-a-real-one"


# ---------------------------------------------------------------------------
# Building a staff room
# ---------------------------------------------------------------------------


def _make_user(session, username: str, *, school_id: int | None) -> int:
    user = m.User(
        username=username,
        password_hash=hash_password(PASSWORD),
        school_id=school_id,
        full_name_en=username,
        full_name_ar=username,
    )
    session.add(user)
    session.flush()
    return user.id


def _grant(session, user_id: int, role: RoleCode, scope: ScopeType, scope_id: int | None) -> None:
    role_id = session.scalar(select(m.Role.id).where(m.Role.code == role.value))
    assert role_id is not None, f"role {role.value} is not in the catalogue"
    session.add(
        m.UserRole(
            user_id=user_id,
            role_id=role_id,
            scope_type=scope.value,
            scope_id=scope_id,
            granted_by="test",
        )
    )


@pytest.fixture()
def ids(client: TestClient, registrar: dict[str, str], school: None) -> dict[str, int]:
    """The surrogate ids of the school built by the timetable fixture.

    Named rather than looked up in each test, because a test that spends six lines finding
    a class id reads as being about SQL when it is about who may see that class.
    """
    with SqlAlchemyUnitOfWork() as uow:
        session = uow._session
        sync_roles(session)
        found = {
            "school": session.scalar(select(m.School.id).where(m.School.code == SCHOOL)),
            "level_p1": session.scalar(
                select(m.YearLevel.id).where(m.YearLevel.code == "AR-P1")
            ),
            "level_s1": session.scalar(
                select(m.YearLevel.id).where(m.YearLevel.code == "AR-S1")
            ),
        }
        for code in ("P1A", "P1B", "LGA"):
            found[f"class_{code}"] = session.scalar(
                select(m.ClassSection.id)
                .join(m.AcademicYear, m.ClassSection.academic_year_id == m.AcademicYear.id)
                .where(m.ClassSection.code == code, m.AcademicYear.code == YEAR)
            )
        uow.commit()
    assert all(value is not None for value in found.values()), found
    return found


def _sign_in(client: TestClient, username: str) -> dict[str, str]:
    response = client.post(
        "/v1/auth/login", json={"username": username, "password": PASSWORD}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": "Bearer " + response.json()["token"]}


@pytest.fixture()
def teacher_of_p1a(client: TestClient, ids: dict[str, int]) -> dict[str, str]:
    """A teacher, granted one classroom and nothing else."""
    with SqlAlchemyUnitOfWork() as uow:
        user_id = _make_user(uow._session, "teacher.p1a", school_id=ids["school"])
        _grant(
            uow._session,
            user_id,
            RoleCode.TEACHER,
            ScopeType.CLASS_SECTION,
            ids["class_P1A"],
        )
        uow.commit()
    return _sign_in(client, "teacher.p1a")


@pytest.fixture()
def principal(client: TestClient, ids: dict[str, int]) -> dict[str, str]:
    with SqlAlchemyUnitOfWork() as uow:
        user_id = _make_user(uow._session, "principal.9", school_id=ids["school"])
        _grant(
            uow._session, user_id, RoleCode.PRINCIPAL, ScopeType.SCHOOL, ids["school"]
        )
        uow.commit()
    return _sign_in(client, "principal.9")


# ---------------------------------------------------------------------------
# Several roles at once
# ---------------------------------------------------------------------------


class TestRolesAreAdditive:
    def test_a_second_role_adds_to_the_first_rather_than_replacing_it(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        """The claim the whole stage rests on, asserted through the granting route.

        A teacher is made an attendance supervisor. If any part of this design replaced
        roles instead of accumulating them, `grades.write` would be gone afterwards.
        """
        with SqlAlchemyUnitOfWork() as uow:
            teacher_id = _make_user(uow._session, "both.9", school_id=ids["school"])
            uow.commit()

        for role in ("teacher", "attendance_supervisor"):
            response = client.post(
                f"/v1/rbac/users/{teacher_id}/roles",
                headers=principal,
                json={
                    "role_code": role,
                    "scope_type": "class_section",
                    "scope_id": ids["class_P1A"],
                },
            )
            assert response.status_code == 200, response.text

        profile = client.post(
            "/v1/auth/login", json={"username": "both.9", "password": PASSWORD}
        ).json()["profile"]

        assert {row["role_code"] for row in profile["roles"]} == {
            "teacher",
            "attendance_supervisor",
        }
        # The union, not the last one granted.
        assert "grades.write" in profile["permissions"]
        assert "attendance.write" in profile["permissions"]

    def test_the_same_role_is_held_at_two_scopes_at_once(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        """A teacher of two classes is two grants of one role, not a conflict."""
        with SqlAlchemyUnitOfWork() as uow:
            teacher_id = _make_user(uow._session, "two.rooms", school_id=ids["school"])
            uow.commit()

        for class_key in ("class_P1A", "class_P1B"):
            assert (
                client.post(
                    f"/v1/rbac/users/{teacher_id}/roles",
                    headers=principal,
                    json={
                        "role_code": "teacher",
                        "scope_type": "class_section",
                        "scope_id": ids[class_key],
                    },
                ).status_code
                == 200
            )

        grants = client.get(
            f"/v1/rbac/users/{teacher_id}/roles", headers=principal
        ).json()
        assert sorted(row["scope_id"] for row in grants) == sorted(
            [ids["class_P1A"], ids["class_P1B"]]
        )

    def test_granting_the_same_grant_twice_changes_nothing(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        """An administrator double-clicking must not leave a duplicate to puzzle over."""
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "twice.9", school_id=ids["school"])
            uow.commit()

        body = {
            "role_code": "teacher",
            "scope_type": "class_section",
            "scope_id": ids["class_P1A"],
        }
        first = client.post(
            f"/v1/rbac/users/{user_id}/roles", headers=principal, json=body
        )
        second = client.post(
            f"/v1/rbac/users/{user_id}/roles", headers=principal, json=body
        )
        assert first.status_code == second.status_code == 200
        assert len(second.json()) == 1

    def test_revoking_one_scope_leaves_the_other(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        """The teacher of 3A and 3B removed from 3A still teaches 3B."""
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "one.left", school_id=ids["school"])
            uow.commit()

        for class_key in ("class_P1A", "class_P1B"):
            client.post(
                f"/v1/rbac/users/{user_id}/roles",
                headers=principal,
                json={
                    "role_code": "teacher",
                    "scope_type": "class_section",
                    "scope_id": ids[class_key],
                },
            )

        left = client.request(
            "DELETE",
            f"/v1/rbac/users/{user_id}/roles",
            headers=principal,
            json={
                "role_code": "teacher",
                "scope_type": "class_section",
                "scope_id": ids["class_P1A"],
            },
        )
        assert left.status_code == 200, left.text
        assert [row["scope_id"] for row in left.json()] == [ids["class_P1B"]]

    def test_grade_supervisor_is_the_same_role_as_year_supervisor(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        """Two words for one job. Granting under either spelling produces one grant.

        Worth a test rather than a comment: if the alias ever became a second role row,
        revoking "year_supervisor" would silently leave "grade_supervisor" in place and
        the person would keep an access somebody believed they had removed.
        """
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "supervisor.9", school_id=ids["school"])
            uow.commit()

        for spelling in ("grade_supervisor", "year_supervisor"):
            assert (
                client.post(
                    f"/v1/rbac/users/{user_id}/roles",
                    headers=principal,
                    json={
                        "role_code": spelling,
                        "scope_type": "year_level",
                        "scope_id": ids["level_p1"],
                    },
                ).status_code
                == 200
            )

        grants = client.get(f"/v1/rbac/users/{user_id}/roles", headers=principal).json()
        assert [row["role_code"] for row in grants] == ["year_supervisor"]


# ---------------------------------------------------------------------------
# The scope actually bounds something
# ---------------------------------------------------------------------------


class TestScopesBite:
    def test_a_class_scoped_teacher_reads_his_own_register(
        self, client: TestClient, teacher_of_p1a: dict[str, str]
    ) -> None:
        response = client.get(
            "/v1/classes/P1A/attendance",
            params={"academic_year": YEAR},
            headers=teacher_of_p1a,
        )
        assert response.status_code == 200, response.text

    def test_a_class_scoped_teacher_is_refused_another_room(
        self, client: TestClient, teacher_of_p1a: dict[str, str]
    ) -> None:
        """The negative half, and the one that proves the scope is load-bearing.

        He holds `attendance.read` — the route's own gate passes. It is the *boundary*
        that refuses him, and nothing else in the request distinguishes this call from
        the one above.
        """
        response = client.get(
            "/v1/classes/P1B/attendance",
            params={"academic_year": YEAR},
            headers=teacher_of_p1a,
        )
        assert response.status_code == 403, response.text
        assert "attendance.read" in response.json()["detail"]["message"]

    def test_an_attendance_supervisor_writes_only_the_class_she_was_given(
        self, client: TestClient, ids: dict[str, int]
    ) -> None:
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "register.9", school_id=ids["school"])
            _grant(
                uow._session,
                user_id,
                RoleCode.ATTENDANCE_SUPERVISOR,
                ScopeType.CLASS_SECTION,
                ids["class_P1A"],
            )
            uow.commit()
        auth = _sign_in(client, "register.9")

        # No children are enrolled in this fixture's school, so neither call can succeed.
        # What separates them is *where* they stop: her own room gets past the permission
        # gate and is refused by the register itself, and the other room never gets that
        # far. Asserting on the difference rather than on a 200 keeps this a test about
        # authorisation instead of about roster data.
        body = {"entries": [{"student_number": "NOBODY-1", "state": "present"}]}

        mine = client.put(
            "/v1/classes/P1A/attendance",
            params={"academic_year": YEAR},
            headers=auth,
            json=body,
        )
        assert mine.status_code != 403, mine.text

        not_mine = client.put(
            "/v1/classes/P1B/attendance",
            params={"academic_year": YEAR},
            headers=auth,
            json=body,
        )
        assert not_mine.status_code == 403, not_mine.text
        assert "attendance.write" in not_mine.json()["detail"]["message"]

    def test_a_grade_scoped_supervisor_reaches_every_class_on_that_grade(
        self, client: TestClient, ids: dict[str, int]
    ) -> None:
        """The ladder, exercised through a route.

        The grant names a rung; the request names a class. Nothing in the request says
        which rung `P1B` is on, so this only passes because the route resolves the class
        up to its grade before asking. That resolution is the piece a flat permission
        table cannot supply, and this is the test that would fail if it were removed.
        """
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "rung.9", school_id=ids["school"])
            _grant(
                uow._session,
                user_id,
                RoleCode.GRADE_SUPERVISOR,
                ScopeType.YEAR_LEVEL,
                ids["level_p1"],
            )
            uow.commit()
        auth = _sign_in(client, "rung.9")

        for class_code in ("P1A", "P1B"):
            response = client.get(
                "/v1/timetable/week",
                params={"academic_year": YEAR, "class_code": class_code, "term": TERM},
                headers=auth,
            )
            assert response.status_code == 200, f"{class_code}: {response.text}"

    def test_a_grade_scoped_supervisor_stops_at_the_next_grade(
        self, client: TestClient, ids: dict[str, int]
    ) -> None:
        """`LGA` is a class on a different rung, in a different track. Same school."""
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "rung.only", school_id=ids["school"])
            _grant(
                uow._session,
                user_id,
                RoleCode.GRADE_SUPERVISOR,
                ScopeType.YEAR_LEVEL,
                ids["level_p1"],
            )
            uow.commit()
        auth = _sign_in(client, "rung.only")

        response = client.get(
            "/v1/timetable/week",
            params={"academic_year": YEAR, "class_code": "LGA", "term": TERM},
            headers=auth,
        )
        assert response.status_code == 403, response.text

    def test_a_school_scoped_principal_does_not_inherit_student_attendance_access(
        self, client: TestClient, principal: dict[str, str]
    ) -> None:
        """Stage 11 separates teacher-attendance visibility from pupil registers."""
        for class_code in ("P1A", "P1B", "LGA"):
            response = client.get(
                "/v1/classes/{}/attendance".format(class_code),
                params={"academic_year": YEAR},
                headers=principal,
            )
            assert response.status_code == 403, f"{class_code}: {response.text}"

    def test_a_write_across_two_grades_is_refused_whole(
        self, client: TestClient, ids: dict[str, int]
    ) -> None:
        """A supervisor of one rung posting a timetable that reaches off it.

        Refused entirely rather than partially applied: the request is one transaction,
        so "most of it worked" is not a state this service can report or a caller can act
        on.
        """
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "spillover.9", school_id=ids["school"])
            _grant(
                uow._session,
                user_id,
                RoleCode.YEAR_SUPERVISOR,
                ScopeType.YEAR_LEVEL,
                ids["level_p1"],
            )
            uow.commit()
        auth = _sign_in(client, "spillover.9")

        client.put(
            f"/v1/schools/{SCHOOL}/timetable-periods",
            json={
                "periods": [
                    {"period_number": n, "name_en": f"P{n}", "name_ar": f"P{n}"}
                    for n in (1, 2)
                ]
            },
            headers={"X-API-Key": "registrar-fixture-key-0000000000"},
        )

        def lesson(class_code: str, period: int) -> dict:
            return {
                "class_code": class_code,
                "term_code": TERM,
                "day_of_week": "sunday",
                "period_number": period,
                "subject_code": "MATH",
            }

        response = client.put(
            "/v1/timetable",
            json={
                "academic_year_code": YEAR,
                "entries": [lesson("P1A", 1), lesson("LGA", 1)],
            },
            headers=auth,
        )
        assert response.status_code == 403, response.text

        # And nothing from the refused request landed — not even the half of it the
        # supervisor was entitled to write.
        stored = client.get("/v1/timetable", params={"academic_year": YEAR})
        assert stored.status_code == 200, stored.text
        assert stored.json() == []


# ---------------------------------------------------------------------------
# Nobody grants themselves more than they hold
# ---------------------------------------------------------------------------


class TestGrantingIsBounded:
    def test_a_user_without_roles_assign_cannot_grant_anything(
        self, client: TestClient, ids: dict[str, int]
    ) -> None:
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "plain.9", school_id=ids["school"])
            uow.commit()
        auth = _sign_in(client, "plain.9")

        response = client.post(
            f"/v1/rbac/users/{user_id}/roles",
            headers=auth,
            json={
                "role_code": "teacher",
                "scope_type": "school",
                "scope_id": ids["school"],
            },
        )
        assert response.status_code == 403

    def test_a_principal_cannot_make_a_system_administrator(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        """The escalation this design would otherwise hand out for free.

        `system_admin` carries `system.manage`, which a principal does not hold. Without
        the check, one call turns whoever runs a school into whoever runs the estate —
        and it would look like an ordinary role assignment in the audit.
        """
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "wants.it.all", school_id=ids["school"])
            uow.commit()

        response = client.post(
            f"/v1/rbac/users/{user_id}/roles",
            headers=principal,
            json={
                "role_code": "system_admin",
                "scope_type": "school",
                "scope_id": ids["school"],
            },
        )
        assert response.status_code == 403
        assert "only Teacher" in response.json()["detail"]["message"]

    def test_a_principal_cannot_grant_at_system_scope(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "global.please", school_id=ids["school"])
            uow.commit()

        response = client.post(
            f"/v1/rbac/users/{user_id}/roles",
            headers=principal,
            json={"role_code": "teacher", "scope_type": "global", "scope_id": None},
        )
        assert response.status_code == 403

    def test_a_principal_can_appoint_the_roles_below_them(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        """The positive case the guard must not break.

        A principal holds every permission a teacher, supervisor or owner carries, which
        is why those grants are theirs to make. If this test fails after a role's bundle
        is edited, the bundle grew a permission the principal does not have — and the fix
        is that role's definition, not this guard.
        """
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "appointed.9", school_id=ids["school"])
            uow.commit()

        for role, scope, scope_id in (
            ("teacher", "class_section", ids["class_P1A"]),
            ("attendance_supervisor", "class_section", ids["class_P1A"]),
            ("grade_supervisor", "year_level", ids["level_p1"]),
        ):
            response = client.post(
                f"/v1/rbac/users/{user_id}/roles",
                headers=principal,
                json={"role_code": role, "scope_type": scope, "scope_id": scope_id},
            )
            assert response.status_code == 200, f"{role}: {response.text}"

    def test_delegated_granting_stops_at_the_delegate_s_own_scope(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        """A role-granter bounded to one rung may not grant across the school.

        The school check is not this check: both people here are in the same school. What
        is under test is whether "you may assign roles *on Grade 1*" means anything, and
        without the scope check on the target it does not — `roles.assign` held anywhere
        would let its holder grant everywhere their school reaches.

        Built by hand rather than through a built-in role because no built-in one is
        narrow today. That is exactly why it is worth a test: the gap would open the first
        time a school delegated the thing this model exists to let them delegate.
        """
        with SqlAlchemyUnitOfWork() as uow:
            session = uow._session
            deputy_id = _make_user(session, "deputy.p1", school_id=ids["school"])
            # Principal — which carries roles.assign — but only over one rung.
            _grant(
                session,
                deputy_id,
                RoleCode.PRINCIPAL,
                ScopeType.YEAR_LEVEL,
                ids["level_p1"],
            )
            subject_id = _make_user(session, "appointee.9", school_id=ids["school"])
            uow.commit()
        auth = _sign_in(client, "deputy.p1")

        on_their_rung = client.post(
            f"/v1/rbac/users/{subject_id}/roles",
            headers=auth,
            json={
                "role_code": "teacher",
                "scope_type": "class_section",
                "scope_id": ids["class_P1A"],
            },
        )
        assert on_their_rung.status_code == 200, on_their_rung.text

        # `LGA` is another rung of the same school. Their authority does not reach it.
        elsewhere = client.post(
            f"/v1/rbac/users/{subject_id}/roles",
            headers=auth,
            json={
                "role_code": "teacher",
                "scope_type": "class_section",
                "scope_id": ids["class_LGA"],
            },
        )
        assert elsewhere.status_code == 403, elsewhere.text

        # Nor may they widen a grant to the whole school, which is the escalation the
        # narrow scope is there to prevent.
        whole_school = client.post(
            f"/v1/rbac/users/{subject_id}/roles",
            headers=auth,
            json={
                "role_code": "teacher",
                "scope_type": "school",
                "scope_id": ids["school"],
            },
        )
        assert whole_school.status_code == 403, whole_school.text

    def test_a_scope_that_names_nothing_is_refused(
        self, client: TestClient, ids: dict[str, int], principal: dict[str, str]
    ) -> None:
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "nowhere.9", school_id=ids["school"])
            uow.commit()

        missing = client.post(
            f"/v1/rbac/users/{user_id}/roles",
            headers=principal,
            json={"role_code": "teacher", "scope_type": "class_section", "scope_id": 90210},
        )
        assert missing.status_code == 404

        unbounded = client.post(
            f"/v1/rbac/users/{user_id}/roles",
            headers=principal,
            json={"role_code": "teacher", "scope_type": "class_section"},
        )
        assert unbounded.status_code == 422


# ---------------------------------------------------------------------------
# The catalogue a console builds itself from
# ---------------------------------------------------------------------------


class TestCatalogue:
    def test_every_role_the_stage_asks_for_is_offered(
        self, client: TestClient, principal: dict[str, str]
    ) -> None:
        roles = client.get("/v1/rbac/roles", headers=principal)
        assert roles.status_code == 200, roles.text
        codes = {row["code"] for row in roles.json()}
        assert {
            "system_admin",
            "school_owner",
            "principal",
            "year_supervisor",
            "attendance_supervisor",
            "teacher",
        } <= codes

    def test_a_role_arrives_with_the_permissions_it_carries(
        self, client: TestClient, principal: dict[str, str]
    ) -> None:
        """So a console can explain a role without shipping the table itself."""
        rows = {row["code"]: row for row in client.get("/v1/rbac/roles", headers=principal).json()}
        assert "grades.write" in rows["teacher"]["permissions"]
        assert "attendance.write" in rows["attendance_supervisor"]["permissions"]
        # The owner looks and does not touch — asserted here rather than trusted.
        assert not [p for p in rows["school_owner"]["permissions"] if p.endswith(".write")]

    def test_the_grade_supervisor_spelling_is_advertised(
        self, client: TestClient, principal: dict[str, str]
    ) -> None:
        rows = {row["code"]: row for row in client.get("/v1/rbac/roles", headers=principal).json()}
        assert "grade_supervisor" in rows["year_supervisor"]["aliases"]

    def test_the_scope_ladder_is_served_widest_first(
        self, client: TestClient, principal: dict[str, str]
    ) -> None:
        """The six scopes this stage asks to be prepared, in the order they nest."""
        scopes = client.get("/v1/rbac/scopes", headers=principal)
        assert scopes.status_code == 200, scopes.text
        rows = scopes.json()
        assert [row["type"] for row in rows] == [
            "global",
            "school",
            "track",
            "year_level",
            "class_section",
            "subject",
        ]
        assert [row["depth"] for row in rows] == sorted(row["depth"] for row in rows)
        assert rows[0]["names_a"] == ""  # a system scope identifies nothing

    def test_the_permission_catalogue_is_this_build(
        self, client: TestClient, principal: dict[str, str]
    ) -> None:
        listed = {row["code"] for row in client.get("/v1/rbac/permissions", headers=principal).json()}
        assert listed == {permission.value for permission in Permission}

    def test_the_catalogue_needs_a_session(self, client: TestClient) -> None:
        """Anonymous callers get the open door everywhere else; not to the role table.

        There is nothing dangerous in it, but it is the one listing that describes the
        access model itself, and handing it out unauthenticated is free reconnaissance.
        """
        assert client.get("/v1/rbac/roles").status_code == 401


# ---------------------------------------------------------------------------
# What a session tells the console about itself
# ---------------------------------------------------------------------------


class TestTheProfile:
    def test_the_profile_carries_scopes_and_not_just_permission_names(
        self, client: TestClient, teacher_of_p1a: dict[str, str], ids: dict[str, int]
    ) -> None:
        """What the UI needs to hide a button on one class and show it on another.

        `permissions` alone cannot do it: it says `grades.write` and not where, so a
        console built on it either shows every class as editable or none.
        """
        me = client.get("/v1/auth/me", headers=teacher_of_p1a)
        assert me.status_code == 200, me.text
        profile = me.json()["profile"]

        assert "grades.write" in profile["permissions"]
        writable = [
            grant
            for grant in profile["grants"]
            if grant["permission"] == "grades.write"
        ]
        assert writable == [
            {
                "permission": "grades.write",
                "scope_type": "class_section",
                "scope_id": ids["class_P1A"],
                # The code, not only the id: a browser navigated to `P1A` and has never
                # seen a surrogate id, so without this the console cannot tell whether
                # the grant covers the class it is drawing.
                "scope_code": "P1A",
            }
        ]

    def test_me_answers_in_the_same_shape_as_login(
        self, client: TestClient, teacher_of_p1a: dict[str, str]
    ) -> None:
        """A console that reloads must not have to parse two different payloads."""
        me = client.get("/v1/auth/me", headers=teacher_of_p1a).json()
        assert set(me) == {
            "full_name_en",
            "full_name_ar",
            "preferred_language",
            "school_code",
            "profile",
        }
        assert me["school_code"] == SCHOOL

    def test_a_signed_out_token_stops_working(
        self, client: TestClient, teacher_of_p1a: dict[str, str]
    ) -> None:
        assert client.post("/v1/auth/logout", headers=teacher_of_p1a).status_code == 204
        assert client.get("/v1/auth/me", headers=teacher_of_p1a).status_code == 401


# ---------------------------------------------------------------------------
# The catalogue in the database
# ---------------------------------------------------------------------------


class TestTheCatalogueReconcile:
    def test_it_runs_once_and_then_says_there_was_nothing_to_do(self) -> None:
        """The guard that took a dozen writes off every single sign-in.

        Beyond cost: the reconcile writes, so running it per login means two people
        signing in at the same moment race to insert the same catalogue row.
        """
        with SqlAlchemyUnitOfWork() as uow:
            assert ensure_catalogue(uow._session) is True
            uow.commit()
        with SqlAlchemyUnitOfWork() as uow:
            assert ensure_catalogue(uow._session) is False

    def test_a_changed_catalogue_is_noticed(self) -> None:
        """A release that adds a permission must reconcile without anybody remembering to."""
        with SqlAlchemyUnitOfWork() as uow:
            ensure_catalogue(uow._session)
            uow.commit()
        with SqlAlchemyUnitOfWork() as uow:
            row = uow._session.scalar(
                select(m.SystemSetting).where(m.SystemSetting.key == "rbac.catalogue")
            )
            assert row.value == catalogue_fingerprint()
            row.value = "stale"
            uow.commit()
        with SqlAlchemyUnitOfWork() as uow:
            assert ensure_catalogue(uow._session) is True

    def test_it_never_touches_a_grant(self, client: TestClient, ids: dict[str, int]) -> None:
        """An upgrade reconciles the catalogue. It must not revoke anybody's access."""
        with SqlAlchemyUnitOfWork() as uow:
            user_id = _make_user(uow._session, "kept.9", school_id=ids["school"])
            _grant(
                uow._session,
                user_id,
                RoleCode.TEACHER,
                ScopeType.CLASS_SECTION,
                ids["class_P1A"],
            )
            uow.commit()

        with SqlAlchemyUnitOfWork() as uow:
            sync_builtin_rbac(uow._session)
            uow.commit()

        with SqlAlchemyUnitOfWork() as uow:
            held = uow._session.scalars(
                select(m.UserRole).where(m.UserRole.user_id == user_id)
            ).all()
            assert len(held) == 1
            assert held[0].scope_id == ids["class_P1A"]

    def test_the_seeder_and_the_service_agree(self) -> None:
        """One reconcile, called from two places — see `sis/demo/seeder.py::sync_roles`."""
        with SqlAlchemyUnitOfWork() as uow:
            roles, permissions = sync_roles(uow._session)
            uow.commit()
        assert roles == len(BUILT_IN_ROLES)
        assert permissions == len(list(Permission))

        with SqlAlchemyUnitOfWork() as uow:
            stored = {
                row.code
                for row in uow._session.scalars(select(m.PermissionRow)).all()
            }
            # Every permission has a label, because both callers derive it the same way.
            blank = uow._session.scalars(
                select(m.PermissionRow).where(m.PermissionRow.name_en == "")
            ).all()
        assert stored >= {permission.value for permission in Permission}
        assert blank == []


# ---------------------------------------------------------------------------
# The other door is still open
# ---------------------------------------------------------------------------


def test_an_integration_is_unaffected_by_any_of_this(
    client: TestClient, registrar: dict[str, str], school: None
) -> None:
    """Stage 9 added roles over the existing arrangement; it did not replace it.

    `test_authentication.py` states the same thing from the other side. This one is here
    because it is the specific regression a permissions change invites: a route gains an
    RBAC dependency, and the nightly import — which carries no session — starts failing at
    three in the morning.
    """
    response = client.get("/v1/classes/P1A/attendance", params={"academic_year": YEAR})
    assert response.status_code == 200, response.text
