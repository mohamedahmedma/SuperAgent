<?php
/**
 * Web service function and service definitions.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$functions = [
    'local_schoolapi_get_student_grades' => [
        'classname'   => 'local_schoolapi\external\get_student_grades',
        'description' => 'Every subject result for one student in one term, with the '
                       . 'percentage Moodle itself computed.',
        'type'        => 'read',
        'capabilities' => 'local/schoolapi:read',
        'services'    => ['local_schoolapi_records'],
        // No 'ajax' => true: this is for server-to-server use by the records facade,
        // and exposing it to browser sessions would put a school-wide read behind
        // whatever a logged-in user's session can be tricked into doing.
    ],
];

$services = [
    'School records (read-only)' => [
        'shortname' => 'local_schoolapi_records',
        'functions' => ['local_schoolapi_get_student_grades'],
        // Named users only: a token belongs to one service account, not to anyone who
        // happens to hold a login.
        'restrictedusers' => 1,
        'enabled' => 1,
        'downloadfiles' => 0,
        'uploadfiles' => 0,
    ],
];
