<?php
/**
 * English strings.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'School records API';
$string['schoolapi:read'] = 'Read student academic records through the school API';
$string['privacy:metadata'] = 'The School records API plugin stores no personal data. '
    . 'It reads existing grade and enrolment data and returns it to an authorised '
    . 'service; every read is subject to the local/schoolapi:read capability.';
$string['cachedef_studentgrades'] = 'Computed subject grades per student';
$string['cachedef_studentattendance'] = 'Computed subject attendance per student';
