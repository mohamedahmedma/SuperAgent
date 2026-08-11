<?php
namespace local_schoolapi\repository;

use local_schoolapi\dto\subject_grade;

defined('MOODLE_INTERNAL') || die();

/**
 * Reads computed grades straight from the gradebook tables.
 *
 * TWO queries for a whole term, however many subjects the student takes. That is the
 * entire point of this plugin: the equivalent through core web services is one call
 * per course, each returning every grade item, and for attendance it is worse still.
 *
 * It does NOT recompute anything. `grade_grades.finalgrade` and `rawgrademax` are
 * values Moodle has already aggregated — applying category weights, drop-lowest and
 * exclusions — and are recomputed by Moodle whenever a grade changes. Reading them is
 * how this stays both fast and consistent with the gradebook a teacher sees.
 * Re-deriving a percentage from `grade_items.grademax` would be faster still and
 * wrong; see `subject_grade::from_row()`.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class grade_repository implements grade_repository_interface {

    /** @var \moodle_database */
    private $db;

    public function __construct(?\moodle_database $db = null) {
        global $DB;
        // Injected for tests, defaulted for production. Constructor injection rather
        // than reaching for the global inside each method, so a test can substitute a
        // database without touching a global.
        $this->db = $db ?? $DB;
    }

    public function find_subject_grades(int $userid, ?string $termprefix = null): array {
        $totals = $this->fetch_course_totals($userid, $termprefix);
        if (!$totals) {
            return [];
        }

        // Two extra queries for ALL courses at once rather than per course. The
        // difference between this and the obvious loop is the difference between a
        // constant number of queries and 3N.
        $courseids = array_keys($totals);
        $counts = $this->fetch_item_aggregates($userid, $courseids);
        $categories = $this->fetch_category_subtotals($userid, $courseids);

        $grades = [];
        foreach ($totals as $courseid => $row) {
            $grades[$courseid] = subject_grade::from_row(
                $row,
                $counts[$courseid] ?? [],
                $categories[$courseid] ?? []
            );
        }
        return $grades;
    }

    /**
     * The course-total row per course, with the per-student aggregation bounds.
     *
     * Enrolment is checked with EXISTS rather than a join: a student may have several
     * enrolment instances in one course (manual plus self, say), and a join would
     * return the course once per instance and silently double-count. EXISTS also lets
     * the planner stop at the first match.
     *
     * A `grade_grades` row can outlive an enrolment, so without this check a student
     * who left the school in a previous term would still report grades.
     */
    private function fetch_course_totals(int $userid, ?string $termprefix): array {
        $params = ['userid' => $userid, 'enroluserid' => $userid];
        $where = '';

        if ($termprefix !== null && $termprefix !== '') {
            // Bound the scan at the database. Moodle's LIKE helper handles the
            // dialect differences and escapes the pattern's special characters, which
            // matters because a course idnumber is operator-supplied text.
            $where = ' AND ' . $this->db->sql_like('c.idnumber', ':termprefix', false, false);
            $params['termprefix'] = $this->db->sql_like_escape($termprefix) . '%';
        }

        $sql = "SELECT gi.courseid,
                       c.idnumber AS courseidnumber,
                       c.shortname,
                       c.fullname,
                       gg.finalgrade,
                       gg.rawgrademax,
                       gg.rawgrademin,
                       gc.aggregation
                  FROM {grade_items} gi
                  JOIN {course} c ON c.id = gi.courseid
                  JOIN {grade_grades} gg ON gg.itemid = gi.id AND gg.userid = :userid
                  -- The course's own grade category carries the aggregation method.
                  -- Needed because an academic subtotal may only be derived by summing
                  -- points when that method is Natural; see subject_grade.
                  JOIN {grade_categories} gc ON gc.id = gi.iteminstance
                 WHERE gi.itemtype = 'course'
                       AND c.visible = 1
                       AND EXISTS (
                           SELECT 1
                             FROM {user_enrolments} ue
                             JOIN {enrol} e ON e.id = ue.enrolid
                            WHERE e.courseid = c.id
                                  AND ue.userid = :enroluserid
                                  -- BOTH statuses. `ue.status` is this student's
                                  -- enrolment; `e.status` is whether the enrolment
                                  -- METHOD is still enabled. A school that disables
                                  -- self-enrolment mid-year leaves active user rows
                                  -- behind a dead instance, and checking only the
                                  -- former keeps reporting those students.
                                  AND ue.status = 0
                                  AND e.status = 0
                       )
                       {$where}
              ORDER BY c.shortname";

        return $this->db->get_records_sql($sql, $params);
    }

    /**
     * Activity modules that are not assessments.
     *
     * An attendance activity puts a gradeable item in the course, so at a school that
     * grades attendance the course total is partly a measure of turning up. That figure
     * is legitimate and is still returned as `percentage` — but a parent asking how
     * their child is doing in Mathematics usually means the academic work, so the items
     * listed here are subtracted to produce the `academic` figure alongside it.
     *
     * Deliberately a small, explicit list rather than a guess at what "an assessment"
     * is. Everything not named here counts, so a module the school does grade
     * academically is never silently dropped. Should become an admin setting once a
     * second school disagrees with it.
     */
    private const NON_ASSESSMENT_MODULES = ['attendance'];

    /**
     * Counts and academic sums per course, in one grouped query.
     *
     * `excluded <> 0` is not a style choice. `grade_grades.excluded` stores the
     * TIMESTAMP at which the grade was excluded — a real row reads 1786347956 — so
     * `excluded = 1` matches nothing and would report every excluded grade as counted.
     *
     * `itemtype` excludes both 'course' and 'category': those rows are Moodle's own
     * subtotals, not things a teacher marks. Counting a category row as a pending item
     * would tell a parent there is work outstanding that does not exist.
     */
    private function fetch_item_aggregates(int $userid, array $courseids): array {
        if (!$courseids) {
            return [];
        }

        [$insql, $inparams] = $this->db->get_in_or_equal($courseids, SQL_PARAMS_NAMED, 'cid');
        [$modsql, $modparams] = $this->db->get_in_or_equal(
            self::NON_ASSESSMENT_MODULES, SQL_PARAMS_NAMED, 'mod', false
        );
        $params = array_merge($inparams, $modparams, ['userid' => $userid]);

        // The academic flag is computed ONCE per row in a derived table, then
        // aggregated outside it.
        //
        // Not merely tidier: Moodle's fix_sql_params() counts placeholder OCCURRENCES,
        // not distinct names, so repeating `:mod1` across three CASE expressions makes
        // it expect five parameters where three were supplied and fail with
        // `invalidqueryparam`. Referencing it once sidesteps that entirely.
        //
        // COALESCE because a module item carries an itemmodule and a manual one does
        // not — and NULL would fail the NOT IN comparison rather than pass it, quietly
        // dropping every manually created assessment from the subtotal.
        $sql = "SELECT courseid,
                       SUM(CASE WHEN isexcluded <> 0 THEN 1 ELSE 0 END) AS excluded,
                       SUM(CASE WHEN isexcluded = 0 AND finalgrade IS NOT NULL
                                THEN 1 ELSE 0 END) AS graded,
                       SUM(CASE WHEN isexcluded = 0 AND finalgrade IS NULL
                                THEN 1 ELSE 0 END) AS pending,
                       SUM(CASE WHEN isacademic = 1 THEN finalgrade ELSE 0 END)
                           AS academicpoints,
                       SUM(CASE WHEN isacademic = 1 THEN rawgrademax ELSE 0 END)
                           AS academicmax,
                       SUM(CASE WHEN isacademic = 1 THEN 1 ELSE 0 END) AS academiccount
                  FROM (
                        SELECT gi.courseid,
                               gg.excluded AS isexcluded,
                               gg.finalgrade,
                               gg.rawgrademax,
                               CASE WHEN gg.excluded = 0
                                         AND gg.finalgrade IS NOT NULL
                                         AND COALESCE(gi.itemmodule, '') {$modsql}
                                    THEN 1 ELSE 0 END AS isacademic
                          FROM {grade_items} gi
                          JOIN {grade_grades} gg
                               ON gg.itemid = gi.id AND gg.userid = :userid
                         WHERE gi.itemtype NOT IN ('course', 'category')
                               AND gi.courseid {$insql}
                       ) items
              GROUP BY courseid";

        $counts = [];
        foreach ($this->db->get_records_sql($sql, $params) as $row) {
            $counts[(int)$row->courseid] = [
                'graded' => (int)$row->graded,
                'excluded' => (int)$row->excluded,
                'pending' => (int)$row->pending,
                'academicpoints' => (float)$row->academicpoints,
                'academicmax' => (float)$row->academicmax,
                'academiccount' => (int)$row->academiccount,
            ];
        }
        return $counts;
    }

    /**
     * Moodle's own subtotal for each gradebook category in the course.
     *
     * This is the general answer to "the grade for part of a subject". Moodle stores a
     * computed grade for every category exactly as it does for the course total, with
     * that category's real aggregation already applied — weights, drop-lowest,
     * exclusions and all. So a school whose weighting makes the derived `academic`
     * figure underivable can group its assessments into a category and read an exact
     * number here instead.
     *
     * The percentage uses `rawgrademax` for the same reason the course total does: it
     * is the PER-STUDENT bound, with excused and ungraded items already removed.
     */
    private function fetch_category_subtotals(int $userid, array $courseids): array {
        if (!$courseids) {
            return [];
        }

        [$insql, $inparams] = $this->db->get_in_or_equal($courseids, SQL_PARAMS_NAMED, 'cid');
        $params = array_merge($inparams, ['userid' => $userid]);

        $sql = "SELECT gi.id,
                       gi.courseid,
                       gi.itemname,
                       gc.aggregation,
                       gg.finalgrade,
                       gg.rawgrademax,
                       gg.rawgrademin
                  FROM {grade_items} gi
                  JOIN {grade_grades} gg ON gg.itemid = gi.id AND gg.userid = :userid
                  JOIN {grade_categories} gc ON gc.id = gi.iteminstance
                 WHERE gi.itemtype = 'category'
                       AND gi.courseid {$insql}
              ORDER BY gi.sortorder";

        $categories = [];
        foreach ($this->db->get_records_sql($sql, $params) as $row) {
            $final = $row->finalgrade === null ? null : (float)$row->finalgrade;
            $max = $row->rawgrademax === null ? null : (float)$row->rawgrademax;
            $min = $row->rawgrademin === null ? 0.0 : (float)$row->rawgrademin;

            $percentage = null;
            if ($final !== null && $max !== null && ($max - $min) > 0) {
                $percentage = round((($final - $min) / ($max - $min)) * 100, 2);
            }

            $categories[(int)$row->courseid][] = [
                // A category with no explicit name renders as the course name in
                // Moodle; empty is more honest than repeating the subject.
                'name' => (string)($row->itemname ?? ''),
                'percentage' => $percentage,
                'finalgrade' => $final,
                'maxgrade' => $max,
            ];
        }
        return $categories;
    }
}
