<?php
namespace local_schoolapi\privacy;

defined('MOODLE_INTERNAL') || die();

/**
 * Privacy provider.
 *
 * A null provider, and accurately so: this plugin stores nothing. It reads grade and
 * enrolment data that core already holds and returns it to an authorised service, so
 * there is no local store for a subject access request to export or for a deletion
 * request to erase — core's own providers cover the underlying data.
 *
 * The MUC cache is not personal-data storage in the privacy API's sense: it is a
 * derived, expiring copy of data core owns, invalidated on change and purgeable at
 * will.
 *
 * Note this is separate from the audit question. Who READ a child's records is
 * recorded by the records facade, which is where the guardian relationship lives;
 * Moodle only ever sees an authorised service account.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class provider implements \core_privacy\local\metadata\null_provider {

    public static function get_reason(): string {
        return 'privacy:metadata';
    }
}
