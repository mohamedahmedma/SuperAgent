<?php
namespace local_schoolapi\repository;

defined('MOODLE_INTERNAL') || die();

/**
 * Resolves the school's own student number onto a Moodle user.
 *
 * This exists so nothing outside Moodle ever has to know a Moodle user id. The
 * records facade keys everything on the school's student number, which is what
 * appears on a letter home and what a registrar can look up; a Moodle id is an
 * internal detail that would leak Moodle into a contract meant to outlive it.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class student_repository {

    /** @var \moodle_database */
    private $db;

    public function __construct(?\moodle_database $db = null) {
        global $DB;
        $this->db = $db ?? $DB;
    }

    /**
     * Find an active student by their school student number.
     *
     * Returns null rather than throwing. "No such student" and "not this guardian's
     * child" must be indistinguishable to the caller, and that decision belongs to the
     * records facade — which can only make it if this layer reports absence plainly
     * instead of raising something with a different shape.
     *
     * Deleted and suspended accounts are excluded: a suspended student's records are a
     * registrar's business, not a chat assistant's.
     */
    public function find_by_idnumber(string $idnumber): ?\stdClass {
        if (trim($idnumber) === '') {
            return null;
        }

        $record = $this->db->get_record(
            'user',
            ['idnumber' => $idnumber, 'deleted' => 0, 'suspended' => 0],
            'id, idnumber, firstname, lastname, alternatename',
            IGNORE_MULTIPLE
        );

        return $record ?: null;
    }
}
