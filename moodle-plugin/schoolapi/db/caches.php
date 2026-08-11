<?php
/**
 * Cache definitions.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$definitions = [
    'studentgrades' => [
        'mode' => cache_store::MODE_APPLICATION,
        // Keyed by user id ALONE, deliberately — not by (user, term).
        //
        // A student has a couple of dozen course rows across every term they have ever
        // taken, so caching all of them costs nothing and term filtering happens in
        // PHP. The payoff is exact invalidation: when a teacher saves a grade, the
        // observer deletes ONE key and knows it has cleared everything stale for that
        // child. With a compound key the observer cannot enumerate the terms and would
        // have to purge the whole cache — which, in a school where someone is always
        // grading something, means the cache is permanently cold.
        'simplekeys' => true,
        'simpledata' => false,
        // A backstop, not the primary invalidation. Grade changes clear this within
        // milliseconds via the observer; the TTL only bounds staleness from writes
        // that bypass the event (a direct SQL fix, a restore).
        'ttl' => 300,
        'staticacceleration' => true,
        // One student is read several times in a turn — list children, then grades,
        // then one subject's detail — so the in-request cache pays for itself.
        'staticaccelerationsize' => 8,
    ],

    'studentattendance' => [
        'mode' => cache_store::MODE_APPLICATION,
        // Keyed by user id alone, for the same reason as grades above.
        'simplekeys' => true,
        'simpledata' => false,
        // Shorter than grades. Attendance invalidation is mostly a whole-cache purge —
        // `attendance_taken` names a session, not a student — so a briefer TTL bounds
        // staleness for the cases the events cannot pinpoint. It is also cheaper to
        // rebuild: two queries over a term's sessions.
        'ttl' => 180,
        'staticacceleration' => true,
        'staticaccelerationsize' => 8,
    ],
];
