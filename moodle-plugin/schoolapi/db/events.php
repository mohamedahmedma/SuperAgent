<?php
/**
 * Event observers.
 *
 * Cache invalidation is driven by Moodle's own grade events rather than by a short
 * TTL alone. A parent who phones the school after a teacher fixes a mark should not
 * be told the old figure for another five minutes, and the school should not have to
 * explain why the assistant disagrees with the gradebook.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$observers = [
    [
        // Fired whenever a single user's grade is created or updated, which covers the
        // ordinary teacher-saves-a-mark path and the exclusion toggle.
        'eventname' => '\core\event\user_graded',
        'callback' => '\local_schoolapi\observer::grade_changed',
        // Invalidation must not be deferred: cron-delayed observers would leave the
        // window this exists to close.
        'internal' => false,
    ],
    [
        // A course-wide regrade changes every enrolled student's total at once — the
        // aggregation method changing, a category weight being edited, drop-lowest
        // being switched on.
        'eventname' => '\core\event\course_module_updated',
        'callback' => '\local_schoolapi\observer::course_changed',
        'internal' => false,
    ],
];
