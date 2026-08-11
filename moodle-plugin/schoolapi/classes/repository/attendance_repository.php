<?php
namespace local_schoolapi\repository;

use local_schoolapi\dto\subject_attendance;

defined('MOODLE_INTERNAL') || die();

/**
 * Reads attendance the way mod_attendance itself computes it.
 *
 * The SQL below is deliberately a port of mod_attendance's own
 * `\mod_attendance\summary::compute_users_points()` (classes/summary.php), widened from
 * one activity to every course in a term. Every condition is copied for a reason, and
 * dropping any one of them silently changes a child's attendance figure:
 *
 *   ats.lasttaken <> 0        Only sessions a teacher has actually taken a register
 *                             for. A scheduled-but-unmarked session is not an absence.
 *   ats.sessdate >= c.startdate
 *                             Sessions dated before the course began are ignored, which
 *                             is how the plugin copes with rollover and bad imports.
 *   stg.deleted = 0 AND stg.visible = 1
 *                             A status the school has retired must not score.
 *   maxgrade PER STATUS SET   The denominator is the highest grade in THAT session's
 *                             status set, not a constant. A school running a reduced
 *                             set for exam days has a different maximum there.
 *   group rule                A session belonging to a group counts only for members.
 *
 * It is also why this plugin exists: core's web services cannot answer this without
 * returning the whole class, requiring write-capable permissions, and demanding the
 * caller be enrolled in every course.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class attendance_repository implements attendance_repository_interface {

    /** @var \moodle_database */
    private $db;

    public function __construct(?\moodle_database $db = null) {
        global $DB;
        $this->db = $db ?? $DB;
    }

    public function find_subject_attendance(int $userid, ?string $termprefix = null): array {
        $totals = $this->fetch_points($userid, $termprefix);
        if (!$totals) {
            return [];
        }

        $bystatus = $this->fetch_status_counts($userid, array_keys($totals));

        $attendance = [];
        foreach ($totals as $courseid => $row) {
            $attendance[$courseid] = subject_attendance::from_row($row, $bystatus[$courseid] ?? []);
        }
        return $attendance;
    }

    /**
     * The FROM and WHERE both queries share.
     *
     * One definition, so the totals and the per-status breakdown cannot drift apart.
     * Two copies of a join this subtle would eventually disagree, and the symptom would
     * be a percentage that contradicts its own status counts.
     *
     * @param array $params Modified in place with the bound values this clause needs.
     */
    private function base_sql(array &$params, ?string $termprefix = null): string {
        $termclause = '';
        if ($termprefix !== null && $termprefix !== '') {
            $termclause = ' AND ' . $this->db->sql_like('c.idnumber', ':termprefix', false, false);
            $params['termprefix'] = $this->db->sql_like_escape($termprefix) . '%';
        }

        // `effectivegroupmode`: a course may force its group mode onto every activity,
        // otherwise the activity's own setting applies. mod_attendance reads this from
        // cm_info; expressed here so the whole thing stays a single query.
        //
        // `<> 0`, not `= 1`. Core tests truthiness rather than equality
        // (cm_info::get_effective_groupmode, and grouplib's empty($course->groupmodeforce)),
        // and the column is a smallint — any non-zero value forces.
        //
        // cm_info also downgrades a forced mode to NOGROUPS when the module does not
        // support groups. Inert here because attendance_supports() returns true for
        // FEATURE_GROUPS, so the two-input CASE is exact FOR THIS MODULE. It would not
        // be for others; do not lift this expression elsewhere.
        $effectivegroupmode =
            '(CASE WHEN c.groupmodeforce <> 0 THEN c.groupmode ELSE cm.groupmode END)';

        // Grouped by (attendanceid, setnumber), not by setnumber alone as
        // mod_attendance does. It can group by setnumber because it works within one
        // activity; across courses the attendance id must be part of the key, or one
        // activity's status set would supply another's maximum.
        // `HAVING MAX(grade) > 0` mirrors the course roll-up in mod_attendance's
        // locallib. A set whose grades are all zero contributes nothing to either side
        // of the fraction, so excluding it leaves the percentage unchanged — but it
        // keeps the session COUNT honest, which is what a parent is actually told.
        $maxgrade = "JOIN (SELECT attendanceid, setnumber, MAX(grade) AS maxgrade
                             FROM {attendance_statuses}
                            WHERE deleted = 0 AND visible = 1
                         GROUP BY attendanceid, setnumber
                           HAVING MAX(grade) > 0) stm
                       ON stm.attendanceid = ats.attendanceid
                       AND stm.setnumber = ats.statusset";

        return "  FROM {attendance_sessions} ats
                  JOIN {attendance} a ON a.id = ats.attendanceid
                  JOIN {course} c ON c.id = a.course
                  JOIN {modules} m ON m.name = 'attendance'
                  JOIN {course_modules} cm
                       ON cm.instance = a.id
                       AND cm.module = m.id
                       AND cm.course = c.id
                       AND cm.deletioninprogress = 0
                  JOIN {attendance_log} atl
                       ON atl.sessionid = ats.id
                       AND atl.studentid = :userid
                  JOIN {attendance_statuses} stg
                       ON stg.id = atl.statusid
                       AND stg.deleted = 0
                       AND stg.visible = 1
                  {$maxgrade}
             LEFT JOIN {groups_members} gm
                       ON gm.userid = atl.studentid
                       AND gm.groupid = ats.groupid
                 WHERE ats.lasttaken <> 0
                       AND ats.sessdate >= c.startdate
                       AND c.visible = 1
                       AND (
                            ats.groupid = 0
                            OR (gm.id IS NOT NULL AND {$effectivegroupmode} <> 0)
                       )
                       AND EXISTS (
                           SELECT 1
                             FROM {user_enrolments} ue
                             JOIN {enrol} e ON e.id = ue.enrolid
                            WHERE e.courseid = c.id
                                  AND ue.userid = :enroluserid
                                  AND ue.status = 0
                                  AND e.status = 0
                       )
                       {$termclause}";
    }

    /**
     * Points, max points and taken-session count per course.
     *
     * COUNT(DISTINCT ats.id) rather than COUNT(*), but NOT because of the group join —
     * `groups_members` carries a UNIQUE (userid, groupid) index, so that join can match
     * at most once. The real reason is `attendance_log`, which has no unique constraint
     * on (sessionid, studentid): a duplicate row there would otherwise inflate the
     * session count. Note the points and maxpoints SUMs remain vulnerable to such a
     * duplicate; DISTINCT cannot protect an aggregate over a joined column.
     */
    private function fetch_points(int $userid, ?string $termprefix): array {
        $params = ['userid' => $userid, 'enroluserid' => $userid];
        $base = $this->base_sql($params, $termprefix);

        $sql = "SELECT c.id AS courseid,
                       c.idnumber AS courseidnumber,
                       c.shortname,
                       c.fullname,
                       COUNT(DISTINCT ats.id) AS takensessions,
                       SUM(stg.grade) AS points,
                       SUM(stm.maxgrade) AS maxpoints
                  {$base}
              GROUP BY c.id, c.idnumber, c.shortname, c.fullname
              ORDER BY c.shortname";

        return $this->db->get_records_sql($sql, $params);
    }

    /**
     * How many sessions fell under each status, per course.
     *
     * Acronyms and descriptions are school-configurable — P/A/L/E are only Moodle's
     * defaults — so they are read from the data rather than assumed. A school using
     * Arabic acronyms gets those back verbatim.
     */
    private function fetch_status_counts(int $userid, array $courseids): array {
        if (!$courseids) {
            return [];
        }

        $params = ['userid' => $userid, 'enroluserid' => $userid];
        $base = $this->base_sql($params);

        [$insql, $inparams] = $this->db->get_in_or_equal($courseids, SQL_PARAMS_NAMED, 'cid');
        $params = array_merge($params, $inparams);

        $sql = "SELECT c.id AS courseid,
                       stg.acronym,
                       stg.description,
                       COUNT(DISTINCT ats.id) AS sessioncount
                  {$base}
                       AND c.id {$insql}
              GROUP BY c.id, stg.acronym, stg.description
              ORDER BY stg.acronym";

        $counts = [];
        // A recordset rather than get_records_sql: the natural key here is
        // (course, acronym), which get_records_sql cannot express, and it would
        // silently drop every row after the first per course.
        $rs = $this->db->get_recordset_sql($sql, $params);
        try {
            foreach ($rs as $row) {
                // Keyed by acronym AND description, not acronym alone. A school running
                // a second status set may reuse an acronym for a different meaning —
                // "P" as Present in one set and Practical in another — and keying on
                // the acronym would silently merge two unrelated counts into one number
                // a parent is then shown.
                $key = $row->acronym . '|' . $row->description;
                $courseid = (int)$row->courseid;

                if (isset($counts[$courseid][$key])) {
                    // Same acronym and meaning in more than one status set: genuinely
                    // the same thing to a parent, so add rather than overwrite.
                    $counts[$courseid][$key]['count'] += (int)$row->sessioncount;
                    continue;
                }

                $counts[$courseid][$key] = [
                    'acronym' => (string)$row->acronym,
                    'description' => (string)$row->description,
                    'count' => (int)$row->sessioncount,
                ];
            }
        } finally {
            // Recordsets hold a server-side cursor; not closing one leaks it for the
            // life of the request.
            $rs->close();
        }
        return $counts;
    }
}
