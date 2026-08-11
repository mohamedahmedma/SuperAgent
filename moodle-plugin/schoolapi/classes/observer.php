<?php
namespace local_schoolapi;

use local_schoolapi\service\attendance_service;
use local_schoolapi\service\grade_service;

defined('MOODLE_INTERNAL') || die();

/**
 * Keeps the caches honest.
 *
 * Every handler here is trivial and must stay that way. An observer runs inside the
 * transaction of whatever a teacher just did, so anything slow or failure-prone shows
 * up as the gradebook being slow — or worse, as a teacher's save being rolled back
 * because a cache delete threw.
 *
 * The plugin is a read-only API; these are the only writes it performs, and they are
 * writes to a cache, not to Moodle's data.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class observer {

    /**
     * One student's grade changed. Drop exactly that student's entry.
     *
     * `relateduserid` is the student who was graded; `userid` is the teacher who did
     * it. Using the wrong one invalidates the teacher's record and leaves the child's
     * stale — a mistake that reads correct until somebody checks.
     */
    public static function grade_changed(\core\event\user_graded $event): void {
        $studentid = (int)($event->relateduserid ?: 0);
        if ($studentid > 0) {
            grade_service::invalidate_student($studentid);
        }
    }

    /**
     * A course-level change that may alter every student's total.
     *
     * There is no cheap way to name the affected students from this event, and walking
     * a course's enrolment inside an observer would put a query on the critical path of
     * an ordinary course edit. Purging is blunt but correct: it is rare, and the cost
     * is a cold cache rather than a wrong grade.
     */
    public static function course_changed(\core\event\course_module_updated $event): void {
        grade_service::invalidate_all();
    }

    /**
     * A register was taken for a class.
     *
     * The event carries a session id and a group type — not a student — because taking
     * a register is a class-wide act. Resolving it to individuals would mean querying
     * the session's log inside the observer, on the teacher's save path, to save a
     * cache miss on a read that has not happened yet. Purging is the cheaper trade.
     */
    public static function attendance_taken(\mod_attendance\event\attendance_taken $event): void {
        attendance_service::invalidate_all();
    }

    /**
     * A student marked their own attendance.
     *
     * The one attendance event that names an individual: here `userid` IS the student,
     * because the student is the actor. Precise invalidation is possible, so it is used
     * — self-marking happens once per student per session and purging the whole cache
     * on each would keep it permanently cold in a school that uses the feature.
     */
    public static function attendance_self_marked(
        \mod_attendance\event\attendance_taken_by_student $event
    ): void {
        $studentid = (int)($event->userid ?: 0);
        if ($studentid > 0) {
            attendance_service::invalidate_student($studentid);
        }
    }

    /**
     * A session or status set changed.
     *
     * Both move the DENOMINATOR, not just one student's marks: adding a session raises
     * every classmate's maximum, and editing a status's point value changes both the
     * numerator and the per-set maximum it feeds. Nothing cached for that class is
     * still correct.
     *
     * Typed as the base event because five different event classes share this handler
     * and they have no common subtype beyond \core\event\base.
     */
    public static function attendance_structure_changed(\core\event\base $event): void {
        attendance_service::invalidate_all();
    }
}
