<?php
namespace local_schoolapi\external;

use core_external\external_api;
use core_external\external_function_parameters;
use core_external\external_multiple_structure;
use core_external\external_single_structure;
use core_external\external_value;
use local_schoolapi\repository\student_repository;
use local_schoolapi\service\grade_service;

defined('MOODLE_INTERNAL') || die();

/**
 * `local_schoolapi_get_student_grades` — one student, one term, one call.
 *
 * Pure plumbing on purpose: declare the contract, check the capability, delegate,
 * serialise. No SQL, no caching, no arithmetic. Everything this class knows is
 * *transport*, so the day the contract gains a field or the storage changes, only one
 * of those two things has to move.
 *
 * The capability is checked at SYSTEM context and the caller is never required to be
 * enrolled anywhere. That is the difference from core's attendance API, which forces
 * a service account to hold write-capable teacher permissions in every course before
 * it can read a register.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class get_student_grades extends external_api {

    public static function execute_parameters(): external_function_parameters {
        return new external_function_parameters([
            'studentidnumber' => new external_value(
                PARAM_RAW_TRIMMED,
                "The school's student number, e.g. S1001. Not a Moodle user id."
            ),
            'term' => new external_value(
                PARAM_RAW_TRIMMED,
                'Course idnumber prefix identifying the term, e.g. "2026-T1-". '
                    . 'Empty for every term.',
                VALUE_DEFAULT,
                ''
            ),
        ]);
    }

    /**
     * @param string $studentidnumber
     * @param string $term
     * @return array
     */
    public static function execute(string $studentidnumber, string $term = ''): array {
        [
            'studentidnumber' => $studentidnumber,
            'term' => $term,
        ] = self::validate_parameters(self::execute_parameters(), [
            'studentidnumber' => $studentidnumber,
            'term' => $term,
        ]);

        // System context: a service account belongs to no course, and requiring it to
        // is the constraint that makes core's attendance API unusable here.
        $context = \context_system::instance();
        self::validate_context($context);
        require_capability('local/schoolapi:read', $context);

        $student = (new student_repository())->find_by_idnumber($studentidnumber);

        // An unknown student returns an empty result, NOT an error. The records facade
        // deliberately makes "no such student", "not your child" and "records
        // restricted" indistinguishable to its caller; a distinct error code here
        // would hand back exactly the signal that design removes, and let anyone
        // holding the token enumerate the school roll.
        if ($student === null) {
            return [
                'found' => false,
                'studentidnumber' => $studentidnumber,
                'term' => $term,
                'subjects' => [],
            ];
        }

        $grades = (new grade_service())->get_subject_grades(
            (int)$student->id,
            $term === '' ? null : $term
        );

        return [
            'found' => true,
            'studentidnumber' => (string)$student->idnumber,
            'term' => $term,
            'subjects' => array_values(array_map(
                static fn($grade) => $grade->to_array(),
                $grades
            )),
        ];
    }

    public static function execute_returns(): external_single_structure {
        return new external_single_structure([
            'found' => new external_value(
                PARAM_BOOL,
                'False when no such active student. Not an error — see the class docs.'
            ),
            'studentidnumber' => new external_value(PARAM_RAW, 'Echoed student number.'),
            'term' => new external_value(PARAM_RAW, 'Echoed term prefix.'),
            'subjects' => new external_multiple_structure(
                new external_single_structure([
                    'courseid' => new external_value(PARAM_RAW, 'Moodle course id.'),
                    'idnumber' => new external_value(PARAM_RAW, 'Course idnumber.'),
                    'shortname' => new external_value(PARAM_RAW, 'Course short name.'),
                    'fullname' => new external_value(PARAM_RAW, 'Course full name.'),
                    'finalgrade' => new external_value(
                        PARAM_FLOAT,
                        'Raw points Moodle aggregated. Null when nothing is graded.',
                        VALUE_OPTIONAL
                    ),
                    'maxgrade' => new external_value(
                        PARAM_FLOAT,
                        'PER-STUDENT aggregation maximum, excluding excused and '
                            . 'ungraded items. Not the course-wide item maximum.',
                        VALUE_OPTIONAL
                    ),
                    'percentage' => new external_value(
                        PARAM_FLOAT,
                        'The figure to show a parent. Null when nothing is gradeable '
                            . 'yet — which is NOT the same as zero.',
                        VALUE_OPTIONAL
                    ),
                    'gradedcount' => new external_value(PARAM_INT, 'Items that counted.'),
                    'excludedcount' => new external_value(
                        PARAM_INT,
                        'Items excused. Surfaced so a figure can be explained rather '
                            . 'than merely stated.'
                    ),
                    'pendingcount' => new external_value(PARAM_INT, 'Items not yet graded.'),
                    'iscomplete' => new external_value(
                        PARAM_BOOL,
                        'True when nothing is outstanding — lets the assistant say '
                            . '"final" rather than "so far".'
                    ),
                ])
            ),
        ]);
    }
}
