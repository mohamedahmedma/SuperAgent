<?php
namespace local_schoolapi\dto;

defined('MOODLE_INTERNAL') || die();

/**
 * One subject's result for one student, as the school assistant reports it.
 *
 * Immutable. A grade that can be mutated after the service computed it is a grade
 * that can be wrong by the time it reaches a parent, and nothing downstream has any
 * business adjusting one.
 *
 * The percentage is computed ONCE, here, from the per-student aggregation bounds —
 * see `from_row()` for why that is the only correct source.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class subject_grade {

    /**
     * @param string      $courseid    Moodle course id, as a string for transport.
     * @param string      $idnumber    The school's own course key, e.g. 2026-T1-G7A-MATH.
     * @param string      $shortname   Course short name.
     * @param string      $fullname    Course full name.
     * @param float|null  $finalgrade  Raw points, null when nothing is graded yet.
     * @param float|null  $maxgrade    Per-student aggregation max, NOT the item max.
     * @param float|null  $percentage  Null when there is nothing to compute from.
     * @param int         $gradedcount Items with a grade that counted.
     * @param int         $excludedcount Items excluded from aggregation.
     * @param int         $pendingcount  Items with no grade yet.
     */
    private function __construct(
        public readonly string $courseid,
        public readonly string $idnumber,
        public readonly string $shortname,
        public readonly string $fullname,
        public readonly ?float $finalgrade,
        public readonly ?float $maxgrade,
        public readonly ?float $percentage,
        public readonly int $gradedcount,
        public readonly int $excludedcount,
        public readonly int $pendingcount,
    ) {
    }

    /**
     * Build from one joined row of the course-total query.
     *
     * The percentage is `finalgrade / rawgrademax`, and the choice of denominator is
     * the single most important line in this plugin.
     *
     * `grade_grades.rawgrademax` is the PER-STUDENT aggregation maximum: Moodle has
     * already removed excluded items and items with no grade from it. `grade_items.
     * grademax` is the course-wide total and keeps both. On real data the difference
     * is not subtle — a student with 90/100, an excluded 10/100 and one ungraded item
     * reads 90% from rawgrademax and 22.5% from the item max.
     *
     * This mirrors what the Moodle user report does when it overrides
     * `$grade_item->grademax` with `$grade_grade->get_grade_max()` before formatting,
     * which is why the figure here matches the gradebook a teacher sees.
     */
    public static function from_row(\stdClass $row, array $counts): self {
        $final = $row->finalgrade === null ? null : (float)$row->finalgrade;
        $max = $row->rawgrademax === null ? null : (float)$row->rawgrademax;
        $min = $row->rawgrademin === null ? 0.0 : (float)$row->rawgrademin;

        $percentage = null;
        // A zero or absent span means there is nothing meaningful to express as a
        // percentage — every item excluded, or none graded. Reporting 0% there would
        // claim a child scored nothing, which is a different and much worse statement
        // than "no grade yet".
        if ($final !== null && $max !== null && ($max - $min) > 0) {
            $percentage = round((($final - $min) / ($max - $min)) * 100, 2);
        }

        return new self(
            courseid: (string)$row->courseid,
            idnumber: (string)($row->courseidnumber ?? ''),
            shortname: (string)($row->shortname ?? ''),
            fullname: (string)($row->fullname ?? ''),
            finalgrade: $final,
            maxgrade: $max,
            percentage: $percentage,
            gradedcount: (int)($counts['graded'] ?? 0),
            excludedcount: (int)($counts['excluded'] ?? 0),
            pendingcount: (int)($counts['pending'] ?? 0),
        );
    }

    /**
     * The web-service shape.
     *
     * Kept next to the object it describes so the external class stays pure plumbing
     * and there is exactly one place to change when a field is added.
     */
    public function to_array(): array {
        return [
            'courseid' => $this->courseid,
            'idnumber' => $this->idnumber,
            'shortname' => $this->shortname,
            'fullname' => $this->fullname,
            'finalgrade' => $this->finalgrade,
            'maxgrade' => $this->maxgrade,
            'percentage' => $this->percentage,
            'gradedcount' => $this->gradedcount,
            'excludedcount' => $this->excludedcount,
            'pendingcount' => $this->pendingcount,
            // "Nothing is still awaiting a decision", which is what lets the assistant
            // say "final" rather than "so far".
            'iscomplete' => $this->pendingcount === 0 && $this->gradedcount > 0,
        ];
    }
}
