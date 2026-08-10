<?php
namespace local_schoolapi\repository;

defined('MOODLE_INTERNAL') || die();

/**
 * How the service reads grades. Deliberately narrow.
 *
 * The interface exists so `grade_service` depends on an abstraction rather than on
 * SQL (DIP): the service can be unit-tested against an in-memory double, and the day
 * grades move behind a different store — a reporting replica, a cache warmed by an
 * event observer — only the implementation changes.
 *
 * One method, because the service needs one thing. A broader "school data" interface
 * would force every implementer to satisfy methods it does not use (ISP).
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
interface grade_repository_interface {

    /**
     * Every subject's computed result for one student.
     *
     * @param int         $userid     Moodle user id of the student.
     * @param string|null $termprefix Course idnumber prefix, e.g. "2026-T1-". Null for
     *                                all terms. Filtering here rather than in the
     *                                caller keeps the query bounded at the database.
     * @return \local_schoolapi\dto\subject_grade[] Keyed by course id.
     */
    public function find_subject_grades(int $userid, ?string $termprefix = null): array;
}
