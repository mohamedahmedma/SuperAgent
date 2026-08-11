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
        /** Assessments only — see `academic_percentage()`. Null when not derivable. */
        public readonly ?float $academicpercentage = null,
        public readonly ?float $academicpoints = null,
        public readonly ?float $academicmaxpoints = null,
        public readonly int $academicitemcount = 0,
        /** Why the academic figure is null, when it is. Empty when it is present. */
        public readonly string $academicunavailable = '',
        /** Moodle's OWN category subtotals. Always exact; see to_array(). */
        public readonly array $categories = [],
    ) {
    }

    /**
     * Aggregation methods under which summing points is EXACT.
     *
     * Only GRADE_AGGREGATE_SUM (13, "Natural" in the UI, and Moodle's default). Under
     * that method a course total genuinely is points-earned over points-available, so
     * taking a subset of the items and summing them the same way is arithmetically the
     * same operation Moodle performs.
     *
     * Under any other method it is NOT. A weighted mean applies per-category weights, a
     * simple weighted mean normalises differently, drop-lowest removes items after
     * ranking them — none of which survive being re-derived from points. Producing a
     * plausible-looking number there is how a parent is shown 50% for a child on 90%,
     * which this codebase has already demonstrated once.
     */
    private const EXACT_AGGREGATIONS = [13];

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
    public static function from_row(\stdClass $row, array $counts, array $categories = []): self {
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

        [$academicpct, $reason] = self::academic_percentage(
            (int)($row->aggregation ?? -1),
            (float)($counts['academicpoints'] ?? 0),
            (float)($counts['academicmax'] ?? 0),
            (int)($counts['academiccount'] ?? 0)
        );

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
            academicpercentage: $academicpct,
            academicpoints: isset($counts['academicpoints']) ? (float)$counts['academicpoints'] : null,
            academicmaxpoints: isset($counts['academicmax']) ? (float)$counts['academicmax'] : null,
            academicitemcount: (int)($counts['academiccount'] ?? 0),
            academicunavailable: $reason,
            categories: $categories,
        );
    }

    /**
     * The subject grade with attendance and other non-assessment activities removed.
     *
     * `percentage` above is Moodle's course total, and it includes EVERYTHING in the
     * gradebook — so at a school that grades attendance, a parent asking "how is she
     * doing in maths" gets a figure partly determined by whether she turned up. Both
     * numbers are legitimate and they answer different questions, which is why both are
     * returned rather than one being chosen on the school's behalf.
     *
     * Returns `[percentage, reason]`. The percentage is null whenever it cannot be
     * derived exactly, and the reason says why — never an approximation. Summing points
     * reproduces Moodle's arithmetic only under Natural aggregation; under a weighted
     * mean or drop-lowest it produces a plausible, wrong number, and a consumer that
     * ignores an `is_exact` flag would put that in front of a parent.
     *
     * A school needing an academic subtotal under a weighted scheme should group its
     * assessments into a gradebook CATEGORY. Moodle then computes that subtotal itself,
     * applying the weights, and it is returned verbatim in `categories`.
     */
    private static function academic_percentage(
        int $aggregation, float $points, float $maxpoints, int $itemcount
    ): array {
        if ($itemcount === 0) {
            // No assessment items at all — a course that is purely attendance, or one
            // where nothing has been marked yet.
            return [null, 'no_assessment_items'];
        }

        if (!in_array($aggregation, self::EXACT_AGGREGATIONS, true)) {
            return [null, 'aggregation_not_summable'];
        }

        if ($maxpoints <= 0) {
            // Everything excused, or nothing graded. Same reasoning as the course
            // total: null, never zero.
            return [null, 'nothing_gradeable'];
        }

        return [round(($points / $maxpoints) * 100, 2), ''];
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
            // Moodle's course total. Includes EVERY gradeable item in the course —
            // attendance among them, if the school grades it.
            'percentage' => $this->percentage,
            'gradedcount' => $this->gradedcount,
            'excludedcount' => $this->excludedcount,
            'pendingcount' => $this->pendingcount,
            // "Nothing is still awaiting a decision", which is what lets the assistant
            // say "final" rather than "so far".
            'iscomplete' => $this->pendingcount === 0 && $this->gradedcount > 0,

            // The same subject with non-assessment activities removed.
            'academic' => [
                'percentage' => $this->academicpercentage,
                'points' => $this->academicpoints,
                'maxpoints' => $this->academicmaxpoints,
                'itemcount' => $this->academicitemcount,
                'unavailable' => $this->academicunavailable,
            ],

            // Moodle's own subtotals for any gradebook categories in this course.
            // Computed by Moodle with the course's real aggregation, so these are exact
            // whatever the scheme — the answer for a school whose weighting makes the
            // `academic` figure above underivable.
            'categories' => array_values($this->categories),
        ];
    }
}
