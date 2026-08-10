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

        // One extra query for ALL courses at once rather than one per course. The
        // difference between this and the obvious loop is the difference between 2
        // queries and 2N.
        $counts = $this->fetch_item_counts($userid, array_keys($totals));

        $grades = [];
        foreach ($totals as $courseid => $row) {
            $grades[$courseid] = subject_grade::from_row($row, $counts[$courseid] ?? []);
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
                       gg.rawgrademin
                  FROM {grade_items} gi
                  JOIN {course} c ON c.id = gi.courseid
                  JOIN {grade_grades} gg ON gg.itemid = gi.id AND gg.userid = :userid
                 WHERE gi.itemtype = 'course'
                       AND c.visible = 1
                       AND EXISTS (
                           SELECT 1
                             FROM {user_enrolments} ue
                             JOIN {enrol} e ON e.id = ue.enrolid
                            WHERE e.courseid = c.id
                                  AND ue.userid = :enroluserid
                                  AND ue.status = 0
                       )
                       {$where}
              ORDER BY c.shortname";

        return $this->db->get_records_sql($sql, $params);
    }

    /**
     * Graded / excluded / pending counts per course, in one grouped query.
     *
     * `excluded <> 0` is not a style choice. `grade_grades.excluded` stores the
     * TIMESTAMP at which the grade was excluded — a real row reads 1786347956 — so
     * `excluded = 1` matches nothing and would report every excluded grade as counted.
     */
    private function fetch_item_counts(int $userid, array $courseids): array {
        if (!$courseids) {
            return [];
        }

        [$insql, $inparams] = $this->db->get_in_or_equal($courseids, SQL_PARAMS_NAMED, 'cid');
        $params = array_merge($inparams, ['userid' => $userid]);

        $sql = "SELECT gi.courseid,
                       SUM(CASE WHEN gg.excluded <> 0 THEN 1 ELSE 0 END) AS excluded,
                       SUM(CASE WHEN gg.excluded = 0 AND gg.finalgrade IS NOT NULL
                                THEN 1 ELSE 0 END) AS graded,
                       SUM(CASE WHEN gg.excluded = 0 AND gg.finalgrade IS NULL
                                THEN 1 ELSE 0 END) AS pending
                  FROM {grade_items} gi
                  JOIN {grade_grades} gg ON gg.itemid = gi.id AND gg.userid = :userid
                 WHERE gi.itemtype <> 'course'
                       AND gi.courseid {$insql}
              GROUP BY gi.courseid";

        $counts = [];
        foreach ($this->db->get_records_sql($sql, $params) as $row) {
            $counts[(int)$row->courseid] = [
                'graded' => (int)$row->graded,
                'excluded' => (int)$row->excluded,
                'pending' => (int)$row->pending,
            ];
        }
        return $counts;
    }
}
