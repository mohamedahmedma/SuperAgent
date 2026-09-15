"""Staff chat is durable, school-isolated, and derives every group membership live."""
from datetime import date

from sqlalchemy import delete, select

from sis.application.services.access import ensure_catalogue
from sis.infrastructure.crypto import hash_password
from sis.infrastructure.db import models as m
from sis.infrastructure.db.unit_of_work import SqlAlchemyUnitOfWork


PASSWORD = "StrongPass123!"
SCHOOL = "CHAT"


def _seed() -> dict[str, int]:
    with SqlAlchemyUnitOfWork() as uow:
        session = uow._session
        ensure_catalogue(session)
        school = m.School(code=SCHOOL, name_en="Chat School", name_ar="مدرسة المحادثات")
        other_school = m.School(code="OTHER", name_en="Other School", name_ar="مدرسة أخرى")
        session.add_all([school, other_school])
        session.flush()
        year = m.AcademicYear(
            code="CHAT-2026", school_id=school.id, name_en="2026", name_ar="٢٠٢٦",
            starts_on=date(2026, 9, 1), ends_on=date(2027, 6, 30), is_current=True,
        )
        session.add(year)
        session.flush()
        grade3 = m.YearLevel(code="G3", school_id=school.id, name_en="Grade 3", name_ar="الصف الثالث")
        grade4 = m.YearLevel(code="G4", school_id=school.id, name_en="Grade 4", name_ar="الصف الرابع")
        session.add_all([grade3, grade4])
        session.flush()
        math = m.Subject(code="MATH", academic_year_id=year.id, name_en="Mathematics", name_ar="الرياضيات")
        science = m.Subject(code="SCI", academic_year_id=year.id, name_en="Science", name_ar="العلوم")
        session.add_all([math, science])
        session.flush()
        session.add_all([
            m.SubjectYearLevel(subject_id=math.id, year_level_id=grade3.id),
            m.SubjectYearLevel(subject_id=science.id, year_level_id=grade4.id),
        ])
        class3 = m.ClassSection(
            academic_year_id=year.id, year_level_id=grade3.id, code="3A",
            name_en="3A", name_ar="٣ أ",
        )
        class4 = m.ClassSection(
            academic_year_id=year.id, year_level_id=grade4.id, code="4A",
            name_en="4A", name_ar="٤ أ",
        )
        session.add_all([class3, class4])
        session.flush()

        def user(username: str, name: str, school_id: int | None) -> m.User:
            row = m.User(
                username=username, password_hash=hash_password(PASSWORD),
                full_name_en=name, full_name_ar=name, school_id=school_id,
            )
            session.add(row)
            session.flush()
            return row

        alice = user("alice.chat", "Alice", school.id)
        bob = user("bob.chat", "Bob", school.id)
        bob.full_name_ar = "أحمد عبد الرحمن"
        carol = user("carol.chat", "Carol", school.id)
        supervisor = user("supervisor.chat", "Supervisor", school.id)
        manager = user("manager.chat", "Manager", school.id)
        attendance_supervisor = user("attendance.chat", "Attendance Supervisor", school.id)
        admin = user("admin.chat", "System Administrator", school.id)
        outsider = user("outside.chat", "Outside", other_school.id)

        teacher_role = session.scalar(select(m.Role).where(m.Role.code == "teacher"))
        supervisor_role = session.scalar(select(m.Role).where(m.Role.code == "floor_supervisor"))
        manager_role = session.scalar(select(m.Role).where(m.Role.code == "school_manager"))
        attendance_role = session.scalar(select(m.Role).where(m.Role.code == "attendance_supervisor"))
        admin_role = session.scalar(select(m.Role).where(m.Role.code == "admin"))
        assert all((teacher_role, supervisor_role, manager_role, attendance_role, admin_role))

        teachers = []
        for index, account in enumerate((alice, bob, carol), start=1):
            teacher = m.Teacher(
                staff_number=f"T-{index}", school_id=school.id, user_id=account.id,
                full_name_en=account.full_name_en, full_name_ar=account.full_name_ar,
            )
            session.add(teacher)
            session.flush()
            teachers.append(teacher)
        alice_teacher, bob_teacher, carol_teacher = teachers

        for account, teacher, level, subject, classroom in (
            (alice, alice_teacher, grade3, math, class3),
            (bob, bob_teacher, grade3, math, class3),
            (carol, carol_teacher, grade4, science, class4),
        ):
            session.add(m.UserRole(
                user_id=account.id, role_id=teacher_role.id,
                scope_type="class_section", scope_id=classroom.id, granted_by="test",
            ))
            session.add(m.TeacherSubject(
                teacher_id=teacher.id, subject_id=subject.id, academic_year_id=year.id,
            ))
            session.add(m.TeacherYearLevel(
                teacher_id=teacher.id, year_level_id=level.id, subject_id=subject.id,
            ))
            session.add(m.TeacherClassSection(
                teacher_id=teacher.id, class_section_id=classroom.id,
                subject_id=subject.id, assigned_by="test",
            ))
        session.add(m.UserRole(
            user_id=supervisor.id, role_id=supervisor_role.id,
            scope_type="year_level", scope_id=grade3.id, granted_by="test",
        ))
        session.add(m.UserRole(
            user_id=manager.id, role_id=manager_role.id,
            scope_type="school", scope_id=school.id, granted_by="test",
        ))
        session.add(m.UserRole(
            user_id=attendance_supervisor.id, role_id=attendance_role.id,
            scope_type="class_section", scope_id=class3.id, granted_by="test",
        ))
        session.add(m.UserRole(
            user_id=admin.id, role_id=admin_role.id,
            scope_type="global", scope_id=None, granted_by="test",
        ))
        uow.commit()
        return {
            "alice": alice.id, "bob": bob.id, "carol": carol.id,
            "supervisor": supervisor.id, "manager": manager.id,
            "attendance_supervisor": attendance_supervisor.id, "admin": admin.id,
            "outsider": outsider.id,
            "alice_teacher": alice_teacher.id,
        }


def _headers(client, username: str) -> dict[str, str]:
    response = client.post("/v1/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def _conversations(client, headers):
    response = client.get(f"/v1/schools/{SCHOOL}/chat/conversations", headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_groups_follow_current_teaching_and_supervisor_assignments(client):
    _seed()
    alice = _headers(client, "alice.chat")
    supervisor = _headers(client, "supervisor.chat")
    manager = _headers(client, "manager.chat")
    attendance_supervisor = _headers(client, "attendance.chat")

    assert {item["category"] for item in _conversations(client, alice)} == {
        "school", "floor", "subject", "class"
    }
    # A floor supervisor inherits every subject and class conversation under their
    # managed floor, plus the floor-supervisor leadership room.
    assert {item["category"] for item in _conversations(client, supervisor)} == {
        "school", "leadership", "floor", "subject", "class"
    }
    scoped = [item for item in _conversations(client, supervisor) if item["scope_code"]]
    assert {item["scope_code"] for item in scoped} == {"G3"}

    # A school manager gets the same hierarchy for every configured grade, while the
    # filter metadata keeps that larger list usable in the console.
    manager_conversations = _conversations(client, manager)
    assert {item["category"] for item in manager_conversations} == {
        "school", "leadership", "floor", "subject", "class"
    }
    assert {item["scope_code"] for item in manager_conversations if item["scope_code"]} == {
        "G3", "G4"
    }

    # Attendance supervisors are explicitly outside the manager + floor-supervisors
    # room. Their own scoped floor/class access is unchanged.
    attendance_conversations = _conversations(client, attendance_supervisor)
    assert "leadership" not in {item["category"] for item in attendance_conversations}


def test_group_message_is_shared_only_with_current_members(client):
    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    carol = _headers(client, "carol.chat")
    subject = next(item for item in _conversations(client, alice) if item["category"] == "subject")

    sent = client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{subject['id']}/messages",
        headers=alice, json={"body": "Planning for next week"},
    )
    assert sent.status_code == 201, sent.text
    read = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{subject['id']}/messages", headers=bob
    )
    assert read.status_code == 200
    assert read.json()[0]["body"] == "Planning for next week"
    assert client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{subject['id']}/messages", headers=carol
    ).status_code == 403

    # Removing the live assignments removes Alice from those automatic groups without
    # deleting history or running a membership synchroniser.
    with SqlAlchemyUnitOfWork() as uow:
        teacher_id = ids["alice_teacher"]
        uow._session.execute(delete(m.TeacherClassSection).where(m.TeacherClassSection.teacher_id == teacher_id))
        uow._session.execute(delete(m.TeacherYearLevel).where(m.TeacherYearLevel.teacher_id == teacher_id))
        uow._session.execute(delete(m.TeacherSubject).where(m.TeacherSubject.teacher_id == teacher_id))
        uow.commit()
    assert {item["category"] for item in _conversations(client, alice)} == {"school"}
    assert client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{subject['id']}/messages", headers=alice
    ).status_code == 403


def test_private_chat_search_and_messages_are_school_isolated(client):
    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    carol = _headers(client, "carol.chat")

    found = client.get(
        f"/v1/schools/{SCHOOL}/chat/people", params={"q": "Bob"}, headers=alice
    )
    assert found.status_code == 200
    assert [person["user_id"] for person in found.json()] == [ids["bob"]]
    forgiving = client.get(
        f"/v1/schools/{SCHOOL}/chat/people", params={"q": "احمدعبدالرحمن"}, headers=alice
    )
    assert [person["user_id"] for person in forgiving.json()] == [ids["bob"]]
    outside = client.get(
        f"/v1/schools/{SCHOOL}/chat/people", params={"q": "Outside"}, headers=alice
    )
    assert outside.json() == []

    opened = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct", headers=alice, json={"user_id": ids["bob"]}
    )
    assert opened.status_code == 201, opened.text
    conversation_id = opened.json()["id"]
    assert client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages",
        headers=bob, json={"body": "Private reply"},
    ).status_code == 201
    assert client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages", headers=alice
    ).json()[0]["body"] == "Private reply"
    assert client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages", headers=carol
    ).status_code == 403


def test_admin_is_invisible_to_staff_but_can_see_everyone(client):
    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    admin = _headers(client, "admin.chat")

    hidden_search = client.get(
        f"/v1/schools/{SCHOOL}/chat/people",
        params={"q": "System Administrator"},
        headers=alice,
    )
    assert hidden_search.status_code == 200
    assert hidden_search.json() == []
    hidden_direct = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct",
        headers=alice,
        json={"user_id": ids["admin"]},
    )
    assert hidden_direct.status_code == 404

    visible_to_admin = client.get(
        f"/v1/schools/{SCHOOL}/chat/people", params={"q": "Alice"}, headers=admin
    )
    assert visible_to_admin.status_code == 200
    assert [person["user_id"] for person in visible_to_admin.json()] == [ids["alice"]]

    private_chat = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct",
        headers=alice,
        json={"user_id": ids["bob"]},
    ).json()
    private_message = client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{private_chat['id']}/messages",
        headers=bob,
        json={"body": "Private staff discussion"},
    ).json()
    ordinary_delivery = client.post(
        f"/v1/schools/{SCHOOL}/chat/presence", headers=alice, json={}
    )
    assert ordinary_delivery.status_code == 200
    assert ordinary_delivery.json()["delivered_messages"] == 1
    observed = next(
        row for row in _conversations(client, admin) if row["id"] == private_chat["id"]
    )
    assert observed["observer_view"] is True
    assert "Alice" in observed["title_en"] and "Bob" in observed["title_en"]
    observed_messages = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{private_chat['id']}/messages",
        headers=admin,
    )
    assert observed_messages.status_code == 200
    assert observed_messages.json()[0]["body"] == "Private staff discussion"
    assert client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{private_chat['id']}/read",
        headers=admin,
    ).status_code == 204
    sender_receipts = client.get(
        f"/v1/schools/{SCHOOL}/chat/messages/{private_message['id']}/receipts",
        headers=bob,
    ).json()
    assert [row["user_id"] for row in sender_receipts] == [ids["alice"]]
    assert sender_receipts[0]["read_at"] is None

    opened = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct",
        headers=admin,
        json={"user_id": ids["alice"]},
    )
    assert opened.status_code == 201, opened.text
    conversation_id = opened.json()["id"]
    assert opened.json()["observer_view"] is False
    assert any(row["id"] == conversation_id for row in _conversations(client, admin))
    assert all(row["id"] != conversation_id for row in _conversations(client, alice))
    assert client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages",
        headers=alice,
    ).status_code == 404

    school_group = next(row for row in _conversations(client, admin) if row["category"] == "school")
    sent = client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{school_group['id']}/messages",
        headers=admin,
        json={"body": "Administrator-only audit note"},
    )
    assert sent.status_code == 201, sent.text
    assert sent.json()["receipts"] == {"total": 0, "delivered": 0, "read": 0}
    heartbeat = client.post(
        f"/v1/schools/{SCHOOL}/chat/presence", headers=alice, json={}
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["delivered_messages"] == 0
    staff_messages = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{school_group['id']}/messages",
        headers=alice,
    )
    assert staff_messages.status_code == 200
    assert staff_messages.json() == []
    staff_group = next(row for row in _conversations(client, alice) if row["id"] == school_group["id"])
    assert staff_group["last_message"] is None
    assert staff_group["unread_count"] == 0


def test_read_marker_clears_unread_count(client):
    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    opened = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct", headers=alice, json={"user_id": ids["bob"]}
    ).json()
    client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{opened['id']}/messages",
        headers=bob, json={"body": "Unread"},
    )
    direct = next(item for item in _conversations(client, alice) if item["id"] == opened["id"])
    assert direct["unread_count"] == 1
    assert client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{opened['id']}/read", headers=alice
    ).status_code == 204
    direct = next(item for item in _conversations(client, alice) if item["id"] == opened["id"])
    assert direct["unread_count"] == 0


def test_conversation_mute_is_private_persistent_and_keeps_unread_count(client):
    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    opened = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct", headers=alice, json={"user_id": ids["bob"]}
    ).json()
    muted = client.put(
        f"/v1/schools/{SCHOOL}/chat/conversations/{opened['id']}/mute",
        headers=alice,
        json={"duration": "one_week"},
    )
    assert muted.status_code == 200, muted.text
    assert muted.json()["is_muted"] is True
    assert muted.json()["muted_until"] is not None
    assert next(row for row in _conversations(client, alice) if row["id"] == opened["id"])["is_muted"] is True
    assert next(row for row in _conversations(client, bob) if row["id"] == opened["id"])["is_muted"] is False

    client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{opened['id']}/messages",
        headers=bob,
        json={"body": "Muted but still unread"},
    )
    conversation = next(row for row in _conversations(client, alice) if row["id"] == opened["id"])
    assert conversation["is_muted"] is True
    assert conversation["unread_count"] == 1

    unmuted = client.put(
        f"/v1/schools/{SCHOOL}/chat/conversations/{opened['id']}/mute",
        headers=alice,
        json={"duration": "off"},
    )
    assert unmuted.status_code == 200, unmuted.text
    assert unmuted.json() == {"is_muted": False, "muted_until": None}


def test_delivery_and_read_receipts_follow_recipient_activity(client):
    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    conversation_id = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct", headers=alice, json={"user_id": ids["bob"]}
    ).json()["id"]
    sent = client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages",
        headers=alice, json={"body": "Receipt test"},
    ).json()
    assert sent["receipts"] == {"total": 1, "delivered": 0, "read": 0}

    assert client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages", headers=bob
    ).status_code == 200
    after_delivery = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages", headers=alice
    ).json()[-1]
    assert after_delivery["receipts"] == {"total": 1, "delivered": 1, "read": 0}

    client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/read", headers=bob
    )
    after_read = client.get(
        f"/v1/schools/{SCHOOL}/chat/messages/{sent['id']}/receipts", headers=alice
    ).json()
    assert len(after_read) == 1
    assert after_read[0]["user_id"] == ids["bob"]
    assert after_read[0]["delivered_at"] is not None
    assert after_read[0]["read_at"] is not None


def test_presence_marks_cross_page_delivery_and_exposes_typing_and_staff_details(client):
    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    conversation_id = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct", headers=alice, json={"user_id": ids["bob"]}
    ).json()["id"]

    sent = client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages",
        headers=alice,
        json={"body": "Delivered outside the chat screen"},
    ).json()
    assert sent["receipts"] == {"total": 1, "delivered": 0, "read": 0}

    heartbeat = client.post(
        f"/v1/schools/{SCHOOL}/chat/presence", headers=bob, json={}
    )
    assert heartbeat.status_code == 200, heartbeat.text
    assert heartbeat.json()["delivered_messages"] == 1
    delivered = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages", headers=alice
    ).json()[-1]
    assert delivered["receipts"] == {"total": 1, "delivered": 1, "read": 0}

    typing = client.post(
        f"/v1/schools/{SCHOOL}/chat/presence",
        headers=bob,
        json={"conversation_id": conversation_id, "typing": True},
    )
    assert typing.status_code == 200, typing.text
    members = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/members", headers=alice
    )
    assert members.status_code == 200, members.text
    bob_member = members.json()[0]
    assert bob_member["user_id"] == ids["bob"]
    assert bob_member["online"] is True
    assert bob_member["typing"] is True
    assert [row["code"] for row in bob_member["subjects"]] == ["MATH"]
    assert [row["code"] for row in bob_member["grades"]] == ["G3"]
    assert [row["code"] for row in bob_member["classes"]] == ["3A"]
    assert "Mathematics" in bob_member["role_caption_en"]

    found = client.get(
        f"/v1/schools/{SCHOOL}/chat/people", params={"q": "Bob"}, headers=alice
    ).json()[0]
    assert found["online"] is True
    assert found["role_caption_ar"].startswith("مدرس")

    stopped = client.post(
        f"/v1/schools/{SCHOOL}/chat/presence",
        headers=bob,
        json={"conversation_id": conversation_id, "typing": False},
    )
    assert stopped.status_code == 200
    assert client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/members", headers=alice
    ).json()[0]["typing"] is False


def test_image_attachment_is_persisted_and_protected_by_conversation_membership(client, monkeypatch, tmp_path):
    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    carol = _headers(client, "carol.chat")
    monkeypatch.setenv("SIS_CHAT_ATTACHMENT_STORAGE", str(tmp_path / "chat-files"))
    conversation_id = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct", headers=alice, json={"user_id": ids["bob"]}
    ).json()["id"]
    uploaded = client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/attachments",
        headers=alice,
        data={"body": "The board photo"},
        files={"files": ("board.png", b"\x89PNG\r\n\x1a\nsmall-test", "image/png")},
    )
    assert uploaded.status_code == 201, uploaded.text
    attachment = uploaded.json()["attachments"][0]
    assert attachment["kind"] == "image"
    downloaded = client.get(
        f"/v1/schools/{SCHOOL}/chat/attachments/{attachment['id']}/file", headers=bob
    )
    assert downloaded.status_code == 200
    assert downloaded.content.startswith(b"\x89PNG")
    assert client.get(
        f"/v1/schools/{SCHOOL}/chat/attachments/{attachment['id']}/file", headers=carol
    ).status_code == 403


def test_message_edit_workflow_and_audit(client, monkeypatch):
    from datetime import UTC, datetime, timedelta

    ids = _seed()
    alice = _headers(client, "alice.chat")
    bob = _headers(client, "bob.chat")
    admin = _headers(client, "admin.chat")

    conversation_id = client.post(
        f"/v1/schools/{SCHOOL}/chat/direct", headers=alice, json={"user_id": ids["bob"]}
    ).json()["id"]

    sent = client.post(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages",
        headers=alice,
        json={"body": "First draft of lesson plan"},
    )
    assert sent.status_code == 201, sent.text
    msg = sent.json()
    msg_id = msg["id"]
    assert msg["edited"] is False
    assert msg["edited_at"] is None
    assert msg["original_body"] is None

    # Bob cannot edit Alice's message
    forbidden = client.put(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages/{msg_id}",
        headers=bob,
        json={"body": "Hacked message"},
    )
    assert forbidden.status_code == 403

    # Blank message edit refused
    blank = client.put(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages/{msg_id}",
        headers=alice,
        json={"body": "   "},
    )
    assert blank.status_code == 422

    # Alice successfully edits her own message
    edited = client.put(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages/{msg_id}",
        headers=alice,
        json={"body": "Updated lesson plan for Grade 3"},
    )
    assert edited.status_code == 200, edited.text
    edited_data = edited.json()
    assert edited_data["body"] == "Updated lesson plan for Grade 3"
    assert edited_data["edited"] is True
    assert edited_data["edited_at"] is not None

    # Bob fetches messages: sees updated body, edited flag, but NO original_body
    bob_messages = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages",
        headers=bob,
    ).json()
    bob_msg = next(m for m in bob_messages if m["id"] == msg_id)
    assert bob_msg["body"] == "Updated lesson plan for Grade 3"
    assert bob_msg["edited"] is True
    assert bob_msg["original_body"] is None

    # Admin fetches messages: sees updated body, edited flag, AND original_body preserved
    admin_messages = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages",
        headers=admin,
    ).json()
    admin_msg = next(m for m in admin_messages if m["id"] == msg_id)
    assert admin_msg["body"] == "Updated lesson plan for Grade 3"
    assert admin_msg["edited"] is True
    assert admin_msg["original_body"] == "First draft of lesson plan"

    # Second edit preserves first original_body
    second_edit = client.put(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages/{msg_id}",
        headers=alice,
        json={"body": "Final lesson plan for Grade 3"},
    )
    assert second_edit.status_code == 200
    admin_messages2 = client.get(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages",
        headers=admin,
    ).json()
    admin_msg2 = next(m for m in admin_messages2 if m["id"] == msg_id)
    assert admin_msg2["body"] == "Final lesson plan for Grade 3"
    assert admin_msg2["original_body"] == "First draft of lesson plan"

    # Cannot edit after 1 hour
    future_now = datetime.now(UTC) + timedelta(hours=2)
    import sis.api.routers.chat as chat_module
    old_datetime = chat_module.datetime

    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            return future_now

        @classmethod
        def fromisoformat(cls, date_string):
            return old_datetime.fromisoformat(date_string)

    monkeypatch.setattr(chat_module, "datetime", MockDatetime)

    expired = client.put(
        f"/v1/schools/{SCHOOL}/chat/conversations/{conversation_id}/messages/{msg_id}",
        headers=alice,
        json={"body": "Too late to edit"},
    )
    assert expired.status_code == 403
    assert "one hour" in expired.text.lower()
