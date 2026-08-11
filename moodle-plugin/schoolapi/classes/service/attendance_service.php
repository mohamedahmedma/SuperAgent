<?php
namespace local_schoolapi\service;

use local_schoolapi\repository\attendance_repository;
use local_schoolapi\repository\attendance_repository_interface;

defined('MOODLE_INTERNAL') || die();

/**
 * Orchestration for attendance: cache, then repository, then filter to the term.
 *
 * Mirrors `grade_service` deliberately — same caching strategy, same invalidation
 * shape, same injection point. Two services that solve the same problem differently
 * is how one of them ends up with a bug the other already fixed.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class attendance_service {

    private const CACHE_AREA = 'studentattendance';

    /** @var attendance_repository_interface */
    private $repository;

    public function __construct(?attendance_repository_interface $repository = null) {
        $this->repository = $repository ?? new attendance_repository();
    }

    /**
     * Attendance per subject, optionally narrowed to a term.
     *
     * Cached across all terms and filtered afterwards, for the same reason as grades:
     * it keeps invalidation exact. See db/caches.php.
     *
     * @return \local_schoolapi\dto\subject_attendance[]
     */
    public function get_subject_attendance(int $userid, ?string $termprefix = null): array {
        $cache = \cache::make('local_schoolapi', self::CACHE_AREA);

        $attendance = $cache->get($userid);
        if ($attendance === false) {
            $attendance = $this->repository->find_subject_attendance($userid, null);
            $cache->set($userid, $attendance);
        }

        if ($termprefix === null || $termprefix === '') {
            return $attendance;
        }

        return array_filter(
            $attendance,
            static fn($record) => str_starts_with($record->idnumber, $termprefix)
        );
    }

    /** Forget one student. Called when a register is taken. */
    public static function invalidate_student(int $userid): void {
        \cache::make('local_schoolapi', self::CACHE_AREA)->delete($userid);
    }

    /** Forget everyone. For session or status changes that affect a whole class. */
    public static function invalidate_all(): void {
        \cache::make('local_schoolapi', self::CACHE_AREA)->purge();
    }
}
