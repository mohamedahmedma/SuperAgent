<?php
namespace local_schoolapi\dto;

defined('MOODLE_INTERNAL') || die();

/**
 * One subject's attendance for one student.
 *
 * Immutable, like `subject_grade`, and for the same reason: nothing downstream has any
 * business adjusting an attendance figure after it was computed.
 *
 * The percentage here is POINTS-weighted, not a count of days. That is not a design
 * choice — it is what mod_attendance itself reports, and a figure that disagreed with
 * the teacher's screen would be worse than no figure at all. See `from_row()`.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class subject_attendance {

    private function __construct(
        public readonly string $courseid,
        public readonly string $idnumber,
        public readonly string $shortname,
        public readonly string $fullname,
        /** Sessions the teacher has actually taken a register for. */
        public readonly int $takensessions,
        /** Points earned, summed from attendance_statuses.grade. */
        public readonly float $points,
        /** Points available across those taken sessions. */
        public readonly float $maxpoints,
        /** Points / maxpoints as a percentage. Null when no register has been taken. */
        public readonly ?float $percentage,
        /** Per-status counts keyed by acronym, e.g. ['P' => 12, 'A' => 1]. */
        public readonly array $bystatus,
    ) {
    }

    /**
     * Build from the aggregated points row plus the per-status breakdown.
     *
     * The percentage is `points / maxpoints`, matching
     * `attendance\summary::get_taken_sessions_summary_for()`, which computes
     * `attendance_calc_fraction($points, $maxpoints)` over TAKEN sessions only.
     *
     * Two things follow from copying that formula rather than inventing one:
     *
     * It is weighted. On a default status set P=2, L=1, E=1, A=0, so a student who was
     * late twice out of ten sessions reports 90%, not 80% — being late is not the same
     * as being absent, and the school's own configuration says by how much.
     *
     * It counts only sessions with a register taken. The alternative Moodle also
     * exposes, `allsessionspercentage`, divides by every scheduled session including
     * ones no teacher has marked yet AND every session still in the future — it has no
     * upper date bound at all, so it reads near zero in week one of a term. Showing a
     * parent a falling attendance rate caused by unmarked registers would be actively
     * misleading, so this reports the taken-sessions figure and states the count
     * alongside it.
     *
     * ZERO TAKEN SESSIONS — a documented divergence, not an oversight. Moodle itself
     * gives two answers here: the teacher's report shows "0.0%" (because
     * `attendance_calc_fraction` returns 0 for a zero denominator), while the gradebook
     * push writes a null rawgrade for the same student. There is no single figure to
     * match. This returns null, agreeing with the gradebook, because a child cannot be
     * absent from a class nobody recorded and "0%" in front of a parent says they were.
     *
     * EXCUSED IS HALF CREDIT, NOT EXEMPT. On the default status set E carries grade 1
     * against a maximum of 2, and no code path anywhere removes an excused session from
     * the denominator. A school wanting excused absences to be neutral changes E's grade
     * to the set maximum in Moodle — it is a configuration decision, not something to
     * special-case here. Special-casing it would put a different number in front of the
     * parent than the teacher sees, which is the one failure this whole design avoids.
     */
    public static function from_row(\stdClass $row, array $bystatus): self {
        $points = (float)($row->points ?? 0);
        $maxpoints = (float)($row->maxpoints ?? 0);

        // No register taken yet is not 0% attendance. A child cannot be absent from a
        // class nobody recorded, and reporting zero would accuse them of exactly that.
        $percentage = $maxpoints > 0 ? round(($points / $maxpoints) * 100, 2) : null;

        return new self(
            courseid: (string)$row->courseid,
            idnumber: (string)($row->courseidnumber ?? ''),
            shortname: (string)($row->shortname ?? ''),
            fullname: (string)($row->fullname ?? ''),
            takensessions: (int)($row->takensessions ?? 0),
            points: $points,
            maxpoints: $maxpoints,
            percentage: $percentage,
            bystatus: $bystatus,
        );
    }

    public function to_array(): array {
        // A list rather than a map: the web service layer needs a declared shape, and
        // acronyms are school-configurable so they cannot be enumerated in advance.
        //
        // The acronym is read from the entry rather than from its array key, because
        // the repository keys on acronym AND description to keep two status sets that
        // reuse a letter for different meanings apart.
        $statuses = [];
        foreach ($this->bystatus as $key => $detail) {
            $statuses[] = [
                'acronym' => (string)($detail['acronym'] ?? $key),
                'description' => (string)($detail['description'] ?? ''),
                'count' => (int)($detail['count'] ?? 0),
            ];
        }

        return [
            'courseid' => $this->courseid,
            'idnumber' => $this->idnumber,
            'shortname' => $this->shortname,
            'fullname' => $this->fullname,
            'takensessions' => $this->takensessions,
            'points' => $this->points,
            'maxpoints' => $this->maxpoints,
            'percentage' => $this->percentage,
            'bystatus' => $statuses,
        ];
    }
}
