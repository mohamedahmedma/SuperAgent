<?php
namespace local_schoolapi\external;

use core_external\external_api;
use core_external\external_function_parameters;
use core_external\external_multiple_structure;
use core_external\external_single_structure;
use core_external\external_value;
use local_schoolapi\repository\student_repository;
use local_schoolapi\service\attendance_service;

defined('MOODLE_INTERNAL') || die();

/**
 * `local_schoolapi_get_student_attendance` — one student's attendance, one call.
 *
 * This is the endpoint the plugin was written for. Core Moodle cannot serve it safely:
 * `mod_attendance_get_session` returns EVERY student in the class, requires
 * `takeattendances` or `changeattendances` (both write-capable, so a token that reads
 * one child's register can rewrite the whole school's), and requires the caller to be
 * enrolled in the course. All three were verified on a live instance.
 *
 * Here: one student, a read-only capability, no enrolment, summarised server-side.
 *
 * Pure plumbing, like its grades counterpart — contract, capability, delegate,
 * serialise. The arithmetic lives in the DTO and the query in the repository.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
final class get_student_attendance extends external_api {

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

    public static function execute(string $studentidnumber, string $term = ''): array {
        [
            'studentidnumber' => $studentidnumber,
            'term' => $term,
        ] = self::validate_parameters(self::execute_parameters(), [
            'studentidnumber' => $studentidnumber,
            'term' => $term,
        ]);

        $context = \context_system::instance();
        self::validate_context($context);
        require_capability('local/schoolapi:read', $context);

        $student = (new student_repository())->find_by_idnumber($studentidnumber);

        // Unknown student returns an empty result, not an error — the records facade
        // makes "no such student", "not your child" and "records restricted"
        // indistinguishable to its caller, and a distinct error here would hand back
        // the signal that design removes.
        if ($student === null) {
            return [
                'found' => false,
                'studentidnumber' => $studentidnumber,
                'term' => $term,
                'subjects' => [],
            ];
        }

        $attendance = (new attendance_service())->get_subject_attendance(
            (int)$student->id,
            $term === '' ? null : $term
        );

        return [
            'found' => true,
            'studentidnumber' => (string)$student->idnumber,
            'term' => $term,
            'subjects' => array_values(array_map(
                static fn($record) => $record->to_array(),
                $attendance
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
                    'takensessions' => new external_value(
                        PARAM_INT,
                        'Sessions a teacher has actually taken a register for. Scheduled '
                            . 'but unmarked sessions are excluded — they are not absences.'
                    ),
                    'points' => new external_value(
                        PARAM_FLOAT,
                        'Points earned, weighted by the school\'s own status values.'
                    ),
                    'maxpoints' => new external_value(
                        PARAM_FLOAT,
                        'Points available across those taken sessions.'
                    ),
                    'percentage' => new external_value(
                        PARAM_FLOAT,
                        'points/maxpoints. Matches what mod_attendance shows a teacher. '
                            . 'Null when no register has been taken — which is NOT 0%.',
                        VALUE_OPTIONAL
                    ),
                    'bystatus' => new external_multiple_structure(
                        new external_single_structure([
                            'acronym' => new external_value(
                                PARAM_RAW,
                                'School-configured code. P/A/L/E are only the defaults.'
                            ),
                            'description' => new external_value(PARAM_RAW, 'Human label.'),
                            'count' => new external_value(PARAM_INT, 'Sessions with this status.'),
                        ]),
                        'Per-status breakdown, so a figure can be explained rather than '
                            . 'merely stated.'
                    ),
                ])
            ),
        ]);
    }
}
