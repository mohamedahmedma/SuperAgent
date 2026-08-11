<?php
namespace local_schoolapi\repository;

defined('MOODLE_INTERNAL') || die();

/**
 * How the service reads attendance.
 *
 * Separate from `grade_repository_interface` rather than merged into one "school data"
 * interface: a caller that needs attendance should not be coupled to grades, and an
 * implementer of one should not be forced to satisfy the other (ISP).
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
interface attendance_repository_interface {

    /**
     * Attendance per subject for one student.
     *
     * @param int         $userid     Moodle user id of the student.
     * @param string|null $termprefix Course idnumber prefix, e.g. "2026-T1-". Null for all.
     * @return \local_schoolapi\dto\subject_attendance[] Keyed by course id.
     */
    public function find_subject_attendance(int $userid, ?string $termprefix = null): array;
}
