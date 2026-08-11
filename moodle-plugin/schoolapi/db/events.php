<?php
/**
 * Event observers.
 *
 * Cache invalidation is driven by Moodle's own events rather than by a short TTL alone.
 * A parent who phones the school after a teacher fixes a mark should not be told the
 * old figure for another five minutes, and the school should not have to explain why
 * the assistant disagrees with the gradebook.
 *
 * Note the asymmetry between grades and attendance, which is forced by the events
 * themselves rather than chosen: `user_graded` names the student it affected, so grades
 * invalidate one key. `attendance_taken` does not — taking a register is a class-wide
 * act and the event carries only a session id — so attendance purges. The one
 * exception is a student marking their own attendance, where the event's `userid` IS
 * the student.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$observers = [
    // ---- Grades -----------------------------------------------------------------
    [
        // One student's grade created or updated. Covers the ordinary
        // teacher-saves-a-mark path and the exclusion toggle.
        'eventname' => '\core\event\user_graded',
        'callback' => '\local_schoolapi\observer::grade_changed',
        // Invalidation must not be deferred: a cron-delayed observer would leave open
        // exactly the window this exists to close.
        'internal' => false,
    ],
    [
        // A course-level change can alter every enrolled student's total at once — the
        // aggregation method, a category weight, drop-lowest being switched on.
        'eventname' => '\core\event\course_module_updated',
        'callback' => '\local_schoolapi\observer::course_changed',
        'internal' => false,
    ],

    // ---- Attendance --------------------------------------------------------------
    [
        // A register was taken for a whole class.
        'eventname' => '\mod_attendance\event\attendance_taken',
        'callback' => '\local_schoolapi\observer::attendance_taken',
        'internal' => false,
    ],
    [
        // A student marked themselves present. The only attendance event that names an
        // individual, so the only one that can invalidate precisely.
        'eventname' => '\mod_attendance\event\attendance_taken_by_student',
        'callback' => '\local_schoolapi\observer::attendance_self_marked',
        'internal' => false,
    ],
    [
        // Adding or removing a session changes the denominator for everyone in the
        // class, so every cached percentage in it is now wrong.
        'eventname' => '\mod_attendance\event\session_added',
        'callback' => '\local_schoolapi\observer::attendance_structure_changed',
        'internal' => false,
    ],
    [
        'eventname' => '\mod_attendance\event\session_deleted',
        'callback' => '\local_schoolapi\observer::attendance_structure_changed',
        'internal' => false,
    ],
    [
        // Status point values feed the numerator AND the per-set maximum, so editing
        // one moves every percentage computed from that set.
        'eventname' => '\mod_attendance\event\status_updated',
        'callback' => '\local_schoolapi\observer::attendance_structure_changed',
        'internal' => false,
    ],
    [
        'eventname' => '\mod_attendance\event\status_added',
        'callback' => '\local_schoolapi\observer::attendance_structure_changed',
        'internal' => false,
    ],
    [
        'eventname' => '\mod_attendance\event\status_removed',
        'callback' => '\local_schoolapi\observer::attendance_structure_changed',
        'internal' => false,
    ],
];
