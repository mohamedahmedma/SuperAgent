<?php
namespace local_schoolapi\service;

use local_schoolapi\repository\grade_repository;
use local_schoolapi\repository\grade_repository_interface;

defined('MOODLE_INTERNAL') || die();

/**
 * Orchestration: cache, then repository, then filter to the requested term.
 *
 * The only class that knows caching exists. The repository knows SQL and nothing
 * else; the external class knows the web-service contract and nothing else. Each has
 * one reason to change, which is what makes the caching strategy replaceable without
 * touching a query or an endpoint.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class grade_service {

    private const CACHE_AREA = 'studentgrades';

    /** @var grade_repository_interface */
    private $repository;

    /**
     * Constructor injection with a sane default.
     *
     * Production callers construct it bare; tests pass a double. Depending on the
     * interface rather than the concrete repository is what makes the service
     * testable without a database.
     */
    public function __construct(?grade_repository_interface $repository = null) {
        $this->repository = $repository ?? new grade_repository();
    }

    /**
     * Every subject's result for one student, optionally narrowed to a term.
     *
     * The cache holds ALL of a student's subjects, and the term filter is applied
     * afterwards in PHP. That looks wasteful and is not: a student has a couple of
     * dozen course rows in total, while the alternative — caching per (student, term)
     * — makes invalidation impossible to do precisely. See db/caches.php.
     *
     * @param int         $userid
     * @param string|null $termprefix Course idnumber prefix, e.g. "2026-T1-".
     * @return \local_schoolapi\dto\subject_grade[]
     */
    public function get_subject_grades(int $userid, ?string $termprefix = null): array {
        $cache = \cache::make('local_schoolapi', self::CACHE_AREA);

        $grades = $cache->get($userid);
        if ($grades === false) {
            $grades = $this->repository->find_subject_grades($userid, null);
            $cache->set($userid, $grades);
        }

        if ($termprefix === null || $termprefix === '') {
            return $grades;
        }

        return array_filter(
            $grades,
            static fn($grade) => str_starts_with($grade->idnumber, $termprefix)
        );
    }

    /**
     * Forget one student. Called from the grade observer.
     *
     * Static because an observer has no service instance to hand and constructing one
     * would build a repository it will never use.
     */
    public static function invalidate_student(int $userid): void {
        \cache::make('local_schoolapi', self::CACHE_AREA)->delete($userid);
    }

    /** Forget everyone. Reserved for course-level changes; see observer. */
    public static function invalidate_all(): void {
        \cache::make('local_schoolapi', self::CACHE_AREA)->purge();
    }
}
