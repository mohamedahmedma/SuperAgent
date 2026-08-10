<?php
namespace local_schoolapi;

defined('MOODLE_INTERNAL') || die();

/**
 * Keeps the grade cache honest.
 *
 * Both handlers are deliberately trivial and must stay that way. An observer runs
 * inside the transaction of whatever a teacher just did, so anything slow or
 * failure-prone here shows up as the gradebook being slow, or worse, as a teacher's
 * save being rolled back because a cache delete threw.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class observer {

    /**
     * One student's grade changed. Drop exactly that student's entry.
     *
     * `relateduserid` is the student who was graded; `userid` is the teacher who did
     * it. Using the wrong one caches-invalidates the teacher's own record and leaves
     * the child's stale — a mistake that reads correct until someone checks.
     */
    public static function grade_changed(\core\event\user_graded $event): void {
        $studentid = (int)($event->relateduserid ?: 0);
        if ($studentid > 0) {
            service\grade_service::invalidate_student($studentid);
        }
    }

    /**
     * Something changed at course level that may alter every student's total.
     *
     * There is no cheap way to name the affected students from this event, and
     * enumerating a course's enrolment inside an observer would put a query on the
     * critical path of an ordinary course edit. Purging the whole definition is the
     * blunt but correct choice: it is rare, and the cost is a cold cache rather than
     * a wrong grade.
     */
    public static function course_changed(\core\event\course_module_updated $event): void {
        service\grade_service::invalidate_all();
    }
}
