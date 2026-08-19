"""Grade import: a student, a subject, a term, and the figure the school stated.

One rule dominates this module and every implementation that touches it: **a blank cell is
`None`, and `None` is not zero** (decision 1). Zero is a mark a child can earn. A parser or
DTO that defaults an empty cell to `0.0` is not filling a gap, it is inventing a failing
grade — and the invention is undetectable downstream, because a stored `0.0` that came from
an empty cell is byte-identical to one a teacher typed. The parent-facing consequence is a
system that reports a child as having scored 0% in a subject nobody has marked yet.

`is_graded` exists so that distinction is expressed once, as a name, instead of being
re-derived as `percentage is not None` at each call site — where sooner or later somebody
writes `percentage or 0.0` and the rule dies quietly.
"""
from dataclasses import dataclass

from sis.domain.value_objects import ClassCode, Percentage, StudentNumber, SubjectCode, TermCode


@dataclass(frozen=True, slots=True)
class ParsedGradeRow:
    """One marks line the parser understood.

    A school reports either a percentage or raw points out of a maximum; both are accepted
    and neither is converted into the other here. Rescaling points into a percentage would
    be deriving a figure, and this service stores only stated ones (decision 5) so that a
    grade it returns is one a teacher can recognise in the sheet they uploaded.

    A row whose figure was present but unusable — text in the marks column, a percentage of
    140 — never reaches this type. It becomes a diagnostic carrying `INVALID_GRADE` or
    `GRADE_OUT_OF_RANGE`, so that "unmarked" and "wrong" stay distinguishable.
    """

    line_number: int
    student_number: StudentNumber
    subject_code: SubjectCode
    #: None when the upload names one term for the whole file, which is the common case.
    term_code: TermCode | None = None
    percentage: Percentage | None = None
    points: float | None = None
    max_points: float | None = None

    @property
    def is_graded(self) -> bool:
        """False when the school has recorded nothing for this subject yet.

        Read this rather than testing `percentage` directly. It is the single place the
        null-is-not-zero rule is expressed, and every caller that respects it is a caller
        that cannot accidentally report an unmarked subject as a zero.
        """
        return self.percentage is not None or self.points is not None


@dataclass(frozen=True, slots=True)
class GradePreviewCommand:
    """Parse and validate a marks upload, writing an ImportBatch and nothing else."""

    term_code: TermCode
    filename: str
    content: bytes
    actor: str
    #: Target subject when the file covers one subject. None means each row names its own.
    subject_code: SubjectCode | None = None
    #: Narrows validation to one class, so a file uploaded against the wrong class is
    #: rejected loudly instead of silently matching students across the whole school.
    class_code: ClassCode | None = None


@dataclass(frozen=True, slots=True)
class GradeCommitCommand:
    """Apply a previously previewed marks batch. See `RosterCommitCommand`."""

    # `str`, matching `ImportBatch.batch_id`, which is a uuid4 hex and never an integer.
    batch_id: str
    actor: str
