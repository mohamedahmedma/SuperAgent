"""Roles, permissions and the scope a grant is true inside.

Three ideas, and keeping them apart is the whole design:

**A permission is a verb on a noun** — `students.read`, `attendance.write`. Routes ask for
one of these and never for a role. `if role == "teacher"` scattered through handlers is the
thing this module exists to prevent: it cannot express "a teacher who is also a supervisor",
it cannot be widened without editing every call site, and it makes a new role a code change
rather than a row.

**A role is a named bundle of permissions.** It is data, not a branch. Adding
"Subject Coordinator" next term is an insert.

**A grant is a role *plus the scope it applies in*.** This is the part a plain RBAC table
gets wrong for a school. "Supervisor" is not true of the whole estate; it is true of one
rung. "Teacher" is not true of every child; it is true of the rooms that teacher stands in.
So a `RoleAssignment` carries a `ScopeType` and the id of the thing it is bounded to, and
the same user may hold the same role twice at two different scopes.

**Grants are additive and never subtractive.** A user's access is the union over every
assignment. Assigning Supervisor to a teacher does not replace Teacher; a person who is
both can do both, which is what actually happens in a school. Nothing in this module can
express "deny", deliberately — a deny rule interacts with the union in ways nobody predicts
correctly, and the failure mode is a person silently losing access they were granted.

The domain never reads the clock and never touches a database. An `AccessProfile` is built
once per request from rows and then answers questions in memory.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from sis.domain.errors import ValidationError


class Permission(StrEnum):
    """Every verb the service can be asked to authorise, spelled once.

    An enum rather than free text so a typo is an ImportError at startup instead of a
    route that silently admits nobody — `students.raed` matches no grant and the screen
    is simply empty, which reads as missing data rather than as a bug.

    Named `noun.verb` so a listing sorts into groups a human can scan.
    """

    # The estate itself. System administrator only.
    SYSTEM_MANAGE = "system.manage"
    SYSTEM_STATUS_WRITE = "system.status.write"

    # Schools and their configuration.
    SCHOOLS_READ = "schools.read"
    SCHOOLS_WRITE = "schools.write"

    # People who log in, and what they are allowed to be.
    USERS_READ = "users.read"
    USERS_WRITE = "users.write"
    ROLES_ASSIGN = "roles.assign"

    # The academic ladder: systems, stages, rungs, classes, terms, subjects.
    STRUCTURE_READ = "structure.read"
    STRUCTURE_WRITE = "structure.write"

    STUDENTS_READ = "students.read"
    STUDENTS_WRITE = "students.write"

    TEACHERS_READ = "teachers.read"
    TEACHERS_WRITE = "teachers.write"
    # Which subject a teacher teaches, and on which rungs. The principal's decision.
    TEACHERS_ASSIGN_SUBJECTS = "teachers.assign_subjects"
    # Which rooms that teacher stands in. The year supervisor's decision.
    TEACHERS_ASSIGN_CLASSES = "teachers.assign_classes"
    TEACHER_ATTENDANCE_READ = "teacher_attendance.read"
    TEACHER_ATTENDANCE_WRITE = "teacher_attendance.write"

    TIMETABLE_READ = "timetable.read"
    TIMETABLE_WRITE = "timetable.write"

    ATTENDANCE_READ = "attendance.read"
    ATTENDANCE_WRITE = "attendance.write"

    GRADES_READ = "grades.read"
    GRADES_WRITE = "grades.write"

    GUARDIANS_READ = "guardians.read"
    GUARDIANS_WRITE = "guardians.write"

    IMPORTS_RUN = "imports.run"
    REPORTS_READ = "reports.read"
    AUDIT_READ = "audit.read"


class ScopeType(StrEnum):
    """How far a grant reaches.

    Ordered from widest to narrowest in `covers`, and that ordering is the only rule:
    a wider scope answers for everything a narrower one would.
    """

    # The whole estate, every school. The system administrator, and nobody else.
    GLOBAL = "global"
    # One school, all of it.
    SCHOOL = "school"
    # One academic stream within a school, such as Arabic or Languages.
    TRACK = "track"
    # One rung of the ladder — "Grade 4" — across every class on it.
    YEAR_LEVEL = "year_level"
    # One classroom.
    CLASS_SECTION = "class_section"
    # One subject, within whatever else bounds the grant.
    SUBJECT = "subject"


@dataclass(frozen=True, slots=True)
class ScopeDescriptor:
    """One rung of the scope ladder, described well enough to draw a picker from.

    The names a school reads are here rather than in the console, because the console is
    not the only client — the OpenAPI page, an administration script and a future mobile
    app all need to say "Grade" for `year_level` and must not each invent their own word
    for it.

    `names_a` is the noun whose id goes in `scope_id`. It is what turns "pick a scope" into
    two questions a person can answer: which *kind* of thing, then which one.
    """

    type: ScopeType
    name_en: str
    name_ar: str
    #: Which table `scope_id` points into. Empty for `global`, which points at nothing.
    names_a: str
    #: Widest first. A scope covers every scope with a larger depth that sits under it.
    depth: int


SCOPE_CATALOGUE: Final[tuple[ScopeDescriptor, ...]] = (
    ScopeDescriptor(ScopeType.GLOBAL, "System", "النظام", "", 0),
    ScopeDescriptor(ScopeType.SCHOOL, "School", "المدرسة", "school", 1),
    ScopeDescriptor(ScopeType.TRACK, "Track", "المسار", "educational_system", 2),
    ScopeDescriptor(ScopeType.YEAR_LEVEL, "Grade", "الصف", "year_level", 3),
    ScopeDescriptor(ScopeType.CLASS_SECTION, "Class", "الفصل", "class_section", 4),
    # Deliberately last and not a rung of the ladder: a subject cuts *across* the ladder
    # rather than sitting on it, so a subject-scoped grant narrows whatever else bounds it
    # instead of nesting inside a class.
    ScopeDescriptor(ScopeType.SUBJECT, "Subject", "المادة", "subject", 5),
)

SCOPE_BY_TYPE: Final[dict[ScopeType, ScopeDescriptor]] = {
    descriptor.type: descriptor for descriptor in SCOPE_CATALOGUE
}


class RoleCode(StrEnum):
    """The roles shipped with the service.

    A `StrEnum` of the *built-in* roles, not of every role that can exist: `roles` is a
    table and a school may add to it. These are the ones the seed creates and the ones
    other code is allowed to name, so a rename here is caught by the type checker rather
    than by a school discovering their principal cannot log in.
    """

    SYSTEM_ADMIN = "system_admin"
    SCHOOL_OWNER = "school_owner"
    PRINCIPAL = "principal"
    YEAR_SUPERVISOR = "year_supervisor"
    # Public Stage-9 terminology; kept as an alias so existing grants remain valid.
    GRADE_SUPERVISOR = "year_supervisor"
    ATTENDANCE_SUPERVISOR = "attendance_supervisor"
    TEACHER = "teacher"
    SUBJECT_COORDINATOR = "subject_coordinator"

    @classmethod
    def _missing_(cls, value: object) -> "RoleCode | None":
        """Accept the spellings a person uses for a role this table stores once.

        The same rung of a school has two names in ordinary speech — a *grade* supervisor
        and a *year* supervisor are one job, and "school manager" and "principal" are one
        person. Storing both would be two role rows that drift apart the first time
        somebody edits one of them, so there is one row and this maps the other spellings
        onto it.

        The mapping is one-way on purpose: `user_roles` only ever holds the stored code, so
        a grant made as `grade_supervisor` and a grant made as `year_supervisor` are the
        same row and revoking either revokes both.
        """
        if not isinstance(value, str):
            return None
        return _ROLE_CODE_ALIASES.get(value.strip().lower().replace("-", "_"))


#: Spellings accepted on input that are not themselves stored. Read by `_missing_` above,
#: and echoed by the role catalogue so a client can show a school its own vocabulary.
_ROLE_CODE_ALIASES: Final[dict[str, RoleCode]] = {
    "grade_supervisor": RoleCode.YEAR_SUPERVISOR,
    "academic_year_supervisor": RoleCode.YEAR_SUPERVISOR,
    "school_manager": RoleCode.PRINCIPAL,
    "manager": RoleCode.PRINCIPAL,
    "owner": RoleCode.SCHOOL_OWNER,
    "admin": RoleCode.SYSTEM_ADMIN,
    "system_administrator": RoleCode.SYSTEM_ADMIN,
}

ROLE_CODE_ALIASES: Final[dict[str, str]] = {
    spelling: role.value for spelling, role in _ROLE_CODE_ALIASES.items()
}


# The read half of the catalogue, shared by every role that looks and does not touch.
#
# Read-only roles are read-only *here*, in one table, rather than by every route
# remembering to refuse a write. School Owner is the clearest case: it holds every
# `*_READ` in the school and not one `*_WRITE`, so "can view everything, can change
# nothing" is a property of this tuple and is impossible to violate by forgetting a check.
_READS: Final[tuple[Permission, ...]] = (
    Permission.SCHOOLS_READ,
    Permission.STRUCTURE_READ,
    Permission.STUDENTS_READ,
    Permission.TEACHERS_READ,
    Permission.TEACHER_ATTENDANCE_READ,
    Permission.TIMETABLE_READ,
    Permission.ATTENDANCE_READ,
    Permission.GRADES_READ,
    Permission.GUARDIANS_READ,
    Permission.REPORTS_READ,
)


@dataclass(frozen=True, slots=True)
class RoleDefinition:
    """A built-in role: its code, its names, and the permissions it carries."""

    code: RoleCode
    name_en: str
    name_ar: str
    description_en: str
    # The scope a grant of this role is normally made at. Advisory — it drives the
    # assignment UI default and the seed; it is not enforced, because a school may
    # legitimately want a school-wide attendance supervisor.
    default_scope: ScopeType
    permissions: tuple[Permission, ...]


BUILT_IN_ROLES: Final[tuple[RoleDefinition, ...]] = (
    RoleDefinition(
        code=RoleCode.SYSTEM_ADMIN,
        name_en="System Administrator",
        name_ar="مدير النظام",
        description_en="Full access to every school, user, role and system setting.",
        default_scope=ScopeType.GLOBAL,
        # Every permission, derived rather than listed: a permission added next month is
        # one the administrator holds without anybody remembering to come back here.
        permissions=tuple(Permission),
    ),
    RoleDefinition(
        code=RoleCode.SCHOOL_OWNER,
        name_en="School Owner",
        name_ar="مالك المدرسة",
        description_en="Sees everything in their own school. Changes nothing.",
        default_scope=ScopeType.SCHOOL,
        permissions=_READS,
    ),
    RoleDefinition(
        code=RoleCode.PRINCIPAL,
        name_en="School Manager / Principal",
        name_ar="مدير المدرسة",
        description_en=(
            "Reads general information and teacher attendance in one school, and adds "
            "approved teaching and supervisor roles. Changes no ordinary school data."
        ),
        default_scope=ScopeType.SCHOOL,
        # Everything in the school, and nothing of the estate. The two system permissions
        # are the whole of the difference between this role and System Administrator, and
        # they are what stops a principal promoting themselves — see `_authorised_to_grant`
        # in the access router, which refuses to grant a role carrying a permission the
        # granter does not already hold.
        #
        # The writes are here rather than only on the narrower roles because a principal
        # who may appoint a teacher but may not correct a mark cannot appoint one: they
        # would be handing out authority they do not have, which is exactly what that
        # guard exists to refuse. Delegating a subset of your own authority is the
        # operation this role performs, so the subset has to be a subset.
        permissions=(
            Permission.SCHOOLS_READ,
            Permission.STRUCTURE_READ,
            Permission.TEACHERS_READ,
            Permission.TEACHERS_ASSIGN_SUBJECTS,
            Permission.TIMETABLE_READ,
            Permission.TEACHER_ATTENDANCE_READ,
            Permission.USERS_READ,
            Permission.ROLES_ASSIGN,
        ),
    ),
    RoleDefinition(
        code=RoleCode.YEAR_SUPERVISOR,
        name_en="Academic Year Supervisor",
        name_ar="موجّه الصف الدراسي",
        description_en=(
            "Sees everything on one rung of the ladder, and puts that rung teachers into "
            "its classrooms."
        ),
        default_scope=ScopeType.YEAR_LEVEL,
        permissions=(
            Permission.STRUCTURE_READ,
            Permission.STUDENTS_READ,
            Permission.TEACHERS_READ,
            Permission.TIMETABLE_READ,
            Permission.ATTENDANCE_READ,
            Permission.GRADES_READ,
            Permission.REPORTS_READ,
            Permission.TEACHERS_ASSIGN_CLASSES,
        ),
    ),
    RoleDefinition(
        code=RoleCode.ATTENDANCE_SUPERVISOR,
        name_en="Attendance Supervisor",
        name_ar="مشرف الحضور والغياب",
        description_en="Takes and reviews the register for the classes they are given.",
        default_scope=ScopeType.CLASS_SECTION,
        permissions=(
            Permission.STRUCTURE_READ,
            Permission.STUDENTS_READ,
            Permission.ATTENDANCE_READ,
            Permission.TIMETABLE_READ,
            Permission.ATTENDANCE_WRITE,
        ),
    ),
    RoleDefinition(
        code=RoleCode.TEACHER,
        name_en="Teacher",
        name_ar="معلّم",
        description_en=(
            "Sees the classes they are assigned to and the children in them, and records "
            "marks for their own subject."
        ),
        default_scope=ScopeType.CLASS_SECTION,
        permissions=(
            Permission.STRUCTURE_READ,
            Permission.STUDENTS_READ,
            Permission.ATTENDANCE_READ,
            Permission.GRADES_READ,
            Permission.GRADES_WRITE,
        ),
    ),
    RoleDefinition(
        code=RoleCode.SUBJECT_COORDINATOR,
        name_en="Subject Coordinator",
        name_ar="منسّق المادة",
        description_en="Sees their subject across every rung it is taught on.",
        default_scope=ScopeType.SCHOOL,
        permissions=(
            Permission.STRUCTURE_READ,
            Permission.STUDENTS_READ,
            Permission.TEACHERS_READ,
            Permission.GRADES_READ,
            Permission.REPORTS_READ,
        ),
    ),
)

ROLE_BY_CODE: Final[dict[str, RoleDefinition]] = {
    definition.code.value: definition for definition in BUILT_IN_ROLES
}


@dataclass(frozen=True, slots=True)
class Scope:
    """Where a grant is true. `GLOBAL` carries no id; everything else must."""

    type: ScopeType
    id: int | None = None

    def __post_init__(self) -> None:
        if self.type is ScopeType.GLOBAL:
            if self.id is not None:
                raise ValidationError(
                    "a global scope names nothing; leave scope_id empty", field="scope_id"
                )
            return
        if self.id is None:
            raise ValidationError(
                f"a {self.type.value} scope must name which one", field="scope_id"
            )
        # A bool is an int, and `scope_id=True` would silently mean "the thing with id 1".
        if isinstance(self.id, bool) or not isinstance(self.id, int):
            raise ValidationError("scope id must be a whole number", field="scope_id")

    def covers(self, target: "Target") -> bool:
        """True when this scope answers for the thing being asked about.

        A wider scope covers a narrower one — a school-scoped grant covers every rung and
        every classroom in that school — but only when the target *says* which school it
        is in. A target that names nothing is covered by `GLOBAL` alone, which is the
        conservative reading: an unlocated question is not one a bounded grant can answer.
        """
        if self.type is ScopeType.GLOBAL:
            return True
        if self.type is ScopeType.SCHOOL:
            return target.school_id is not None and target.school_id == self.id
        if self.type is ScopeType.TRACK:
            return target.track_id is not None and target.track_id == self.id
        if self.type is ScopeType.YEAR_LEVEL:
            return target.year_level_id is not None and target.year_level_id == self.id
        if self.type is ScopeType.CLASS_SECTION:
            return target.class_section_id is not None and target.class_section_id == self.id
        if self.type is ScopeType.SUBJECT:
            return target.subject_id is not None and target.subject_id == self.id
        return False

    def __str__(self) -> str:
        return self.type.value if self.id is None else f"{self.type.value}:{self.id}"


@dataclass(frozen=True, slots=True)
class Target:
    """The thing a permission is being asked about, as much of it as the caller knows.

    Every field is optional because a route knows different amounts at different points:
    "list the schools I can see" names nothing, "take this register" names a school, a
    rung and a class. The more it names, the more grants can match — a class-scoped
    teacher is invisible to a question that only says "school 1".

    Filling this in honestly is the caller's job. A route that omits the class id on a
    class-scoped read widens nothing (the narrow grant simply fails to match); a route
    that *invents* one would. So the failure mode of forgetting is a refusal, not a leak.
    """

    school_id: int | None = None
    track_id: int | None = None
    year_level_id: int | None = None
    class_section_id: int | None = None
    subject_id: int | None = None

    @property
    def is_unlocated(self) -> bool:
        return (
            self.school_id is None
            and self.track_id is None
            and self.year_level_id is None
            and self.class_section_id is None
            and self.subject_id is None
        )


# The empty target: "may this user do this at all, anywhere". Only a GLOBAL grant matches.
ANYWHERE: Final[Target] = Target()


@dataclass(frozen=True, slots=True)
class Grant:
    """One permission, true inside one scope. The atom a profile is a set of."""

    permission: Permission
    scope: Scope


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    """A role given to a user, bounded to a scope. One row of `user_roles`."""

    role_code: str
    scope: Scope
    # Who granted it, for the audit trail. A username, not an id, so a log line reads.
    granted_by: str = ""

    @property
    def key(self) -> tuple[str, str, int | None]:
        """Identity for de-duplication: the same role at the same scope is one grant."""
        return (self.role_code, self.scope.type.value, self.scope.id)


@dataclass(frozen=True, slots=True)
class AccessProfile:
    """Everything one user may do, flattened and ready to answer questions.

    Built once per request from the user's assignments and the role→permission table,
    then asked repeatedly. Flattening is what makes additive roles work without any
    special case: two roles granting the same permission at two scopes produce two
    grants, and `allows` stops at the first that covers the target.
    """

    user_id: int
    username: str
    # Every assignment the user holds, kept for display — "you are a Teacher and a
    # Supervisor" is something the console shows, and it is not derivable from grants.
    assignments: tuple[RoleAssignment, ...] = ()
    grants: tuple[Grant, ...] = ()
    # The school this session reads, when the user is bound to one. `None` for the system
    # administrator, who is bound to none.
    school_id: int | None = None

    def allows(self, permission: Permission, target: "Target" = ANYWHERE) -> bool:
        """The whole authorisation decision: any grant of this permission that covers it."""
        return any(
            grant.permission is permission and grant.scope.covers(target)
            for grant in self.grants
        )

    def allows_any(
        self, permissions: Iterable[Permission], target: "Target" = ANYWHERE
    ) -> bool:
        return any(self.allows(permission, target) for permission in permissions)

    def holds(self, permission: Permission) -> bool:
        """Whether the user has this permission *at all*, at any scope.

        Distinct from `allows(permission)` and the difference is the whole reason this
        method exists. `allows` with no target asks "may you do this everywhere", which
        only a global grant satisfies — so a teacher who holds `grades.write` on one
        classroom fails it, and a route that gated on it would refuse every teacher in the
        school while looking like it was checking the right thing.

        This is the question a route gate asks: is it worth letting this request reach the
        handler at all. The handler then narrows with the ids it knows, through `allows`
        and a fully-named `Target`. Two checks rather than one, because the wide one can be
        answered before any database work and the narrow one cannot.
        """
        return any(grant.permission is permission for grant in self.grants)

    def widest_scope_for(self, permission: Permission) -> ScopeType | None:
        """The least-bounded scope this permission is held at, or `None`.

        Lets a caller skip resolving a target it is about to not need: a school-wide grant
        answers a question about any class in the school without anybody having to look up
        which rung that class is on.
        """
        depths = [
            SCOPE_BY_TYPE[grant.scope.type].depth
            for grant in self.grants
            if grant.permission is permission
        ]
        if not depths:
            return None
        return SCOPE_CATALOGUE[min(depths)].type

    def has_role(self, role_code: str) -> bool:
        return any(assignment.role_code == role_code for assignment in self.assignments)

    @property
    def is_system_admin(self) -> bool:
        return self.has_role(RoleCode.SYSTEM_ADMIN.value)

    @property
    def role_codes(self) -> tuple[str, ...]:
        """Distinct role codes, in a stable order, for display and for logging."""
        seen: list[str] = []
        for assignment in self.assignments:
            if assignment.role_code not in seen:
                seen.append(assignment.role_code)
        return tuple(seen)

    def permissions(self) -> tuple[str, ...]:
        """Distinct permission codes the user holds *somewhere*, sorted.

        What the console uses to decide which buttons exist. Note the "somewhere": a
        button shown from this list still has its scope checked when it is pressed,
        because hiding a control is a courtesy and the server-side check is the boundary.
        """
        return tuple(sorted({grant.permission.value for grant in self.grants}))

    def scopes_for(self, permission: Permission) -> tuple[Scope, ...]:
        """Where this permission is held — how a screen knows which classes to list."""
        return tuple(grant.scope for grant in self.grants if grant.permission is permission)

    def scope_ids(self, permission: Permission, of: ScopeType) -> tuple[int, ...]:
        """The ids of one kind of scope this permission is held at, de-duplicated.

        The narrowing a listing route needs: "which classes may this person take a
        register for" is `scope_ids(ATTENDANCE_WRITE, ScopeType.CLASS_SECTION)`.
        """
        found: list[int] = []
        for scope in self.scopes_for(permission):
            if scope.type is of and scope.id is not None and scope.id not in found:
                found.append(scope.id)
        return tuple(found)


def build_profile(
    *,
    user_id: int,
    username: str,
    assignments: Iterable[RoleAssignment],
    permissions_by_role: dict[str, tuple[Permission, ...]],
    school_id: int | None = None,
) -> AccessProfile:
    """Flatten assignments into grants. The union, with duplicates collapsed.

    An assignment naming a role the table does not have contributes nothing rather than
    raising: a role deleted while a user still holds it is a data problem for an
    administrator to fix, and it must not stop that user logging in to find out.
    """
    held: list[RoleAssignment] = []
    seen_assignments: set[tuple[str, str, int | None]] = set()
    for assignment in assignments:
        if assignment.key in seen_assignments:
            continue
        seen_assignments.add(assignment.key)
        held.append(assignment)

    grants: list[Grant] = []
    seen_grants: set[tuple[str, str, int | None]] = set()
    for assignment in held:
        for permission in permissions_by_role.get(assignment.role_code, ()):
            key = (permission.value, assignment.scope.type.value, assignment.scope.id)
            if key in seen_grants:
                continue
            seen_grants.add(key)
            grants.append(Grant(permission=permission, scope=assignment.scope))

    return AccessProfile(
        user_id=user_id,
        username=username,
        assignments=tuple(held),
        grants=tuple(grants),
        school_id=school_id,
    )


class SystemStatus(StrEnum):
    """Whether the service is answering, and how it says no when it is not.

    `MAINTENANCE` and `PAUSED` differ in intent rather than in mechanism, and both are
    reversible from the same screen that set them. Reads stay open under `MAINTENANCE` so
    a school can still look up a child during an upgrade window; `PAUSED` closes
    everything but the administrator own way back in.
    """

    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    PAUSED = "paused"

    @property
    def allows_reads(self) -> bool:
        return self is not SystemStatus.PAUSED

    @property
    def allows_writes(self) -> bool:
        return self is SystemStatus.ACTIVE


__all__ = [
    "ANYWHERE",
    "AccessProfile",
    "BUILT_IN_ROLES",
    "Grant",
    "Permission",
    "ROLE_BY_CODE",
    "ROLE_CODE_ALIASES",
    "RoleAssignment",
    "RoleCode",
    "RoleDefinition",
    "SCOPE_BY_TYPE",
    "SCOPE_CATALOGUE",
    "Scope",
    "ScopeDescriptor",
    "ScopeType",
    "SystemStatus",
    "Target",
    "build_profile",
]
