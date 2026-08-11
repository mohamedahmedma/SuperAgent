<?php
namespace local_schoolapi;

use local_schoolapi\dto\subject_attendance;

defined('MOODLE_INTERNAL') || die();

/**
 * The attendance percentage rule.
 *
 * Pure arithmetic over a row, so these run without a database.
 *
 * The rule under test is NOT "days present over days total". mod_attendance weights
 * each status by a school-configured point value — on Moodle's defaults P=2, L=1, E=1,
 * A=0 — and divides by the maximum available across sessions actually taken. Copying
 * that formula rather than inventing a simpler one is the whole point: a figure that
 * disagreed with the register a teacher is looking at would be worse than none.
 *
 * @package   local_schoolapi
 * @covers    \local_schoolapi\dto\subject_attendance
 */
final class subject_attendance_test extends \basic_testcase {

    private function row(array $overrides = []): \stdClass {
        return (object)array_merge([
            'courseid' => 2,
            'courseidnumber' => '2026-T1-G7A-MATH',
            'shortname' => '2026-T1-G7A-MATH',
            'fullname' => 'Mathematics',
            'takensessions' => 1,
            'points' => 2.0,
            'maxpoints' => 2.0,
        ], $overrides);
    }

    public function test_full_marks_is_one_hundred_percent(): void {
        // Present at the one session taken: 2 of a possible 2.
        $attendance = subject_attendance::from_row($this->row(), ['P' => ['count' => 1]]);

        $this->assertSame(100.0, $attendance->percentage);
    }

    public function test_excused_earns_partial_credit_not_zero(): void {
        // The case that separates this from a naive count. On the seeded instance a
        // student marked Excused scores 1 of a possible 2 — 50%, not the 0% a
        // present/total count would report. The school's own status configuration
        // decides how much an excused absence costs; this code does not.
        $attendance = subject_attendance::from_row(
            $this->row(['points' => 1.0, 'maxpoints' => 2.0]),
            ['E' => ['count' => 1]]
        );

        $this->assertSame(50.0, $attendance->percentage);
    }

    public function test_late_is_weighted_between_present_and_absent(): void {
        // Ten sessions: eight present (16), two late (2) out of a possible 20.
        $attendance = subject_attendance::from_row(
            $this->row(['takensessions' => 10, 'points' => 18.0, 'maxpoints' => 20.0]),
            ['P' => ['count' => 8], 'L' => ['count' => 2]]
        );

        // 90%, not the 80% that counting only "present" days would give.
        $this->assertSame(90.0, $attendance->percentage);
    }

    public function test_absent_throughout_is_zero_percent(): void {
        // A genuine zero. Distinct from "no register taken", below.
        $attendance = subject_attendance::from_row(
            $this->row(['points' => 0.0, 'maxpoints' => 4.0]),
            ['A' => ['count' => 2]]
        );

        $this->assertSame(0.0, $attendance->percentage);
    }

    public function test_no_register_taken_is_null_not_zero(): void {
        // A child cannot be absent from a class nobody recorded. Reporting 0% here
        // accuses them of exactly that.
        $attendance = subject_attendance::from_row(
            $this->row(['takensessions' => 0, 'points' => 0.0, 'maxpoints' => 0.0]),
            []
        );

        $this->assertNull($attendance->percentage);
        $this->assertSame(0, $attendance->takensessions);
    }

    public function test_null_points_from_the_database_do_not_crash(): void {
        // SUM() over no rows returns NULL, not 0.
        $attendance = subject_attendance::from_row(
            $this->row(['points' => null, 'maxpoints' => null]),
            []
        );

        $this->assertNull($attendance->percentage);
        $this->assertSame(0.0, $attendance->points);
    }

    public function test_status_breakdown_is_carried_through(): void {
        // Keys are "acronym|description" as the repository builds them; the acronym
        // itself is carried in the entry.
        $attendance = subject_attendance::from_row($this->row(), [
            'P|Present' => ['acronym' => 'P', 'description' => 'Present', 'count' => 8],
            'A|Absent' => ['acronym' => 'A', 'description' => 'Absent', 'count' => 2],
        ]);

        $bystatus = $attendance->to_array()['bystatus'];
        $this->assertCount(2, $bystatus);
        $this->assertSame('P', $bystatus[0]['acronym']);
        $this->assertSame('Present', $bystatus[0]['description']);
        $this->assertSame(8, $bystatus[0]['count']);
    }

    public function test_one_acronym_with_two_meanings_stays_two_entries(): void {
        // A school running a second status set may reuse a letter for something else.
        // Merging them on the acronym alone would show a parent one number covering two
        // unrelated things.
        $attendance = subject_attendance::from_row($this->row(), [
            'P|Present' => ['acronym' => 'P', 'description' => 'Present', 'count' => 8],
            'P|Practical' => ['acronym' => 'P', 'description' => 'Practical', 'count' => 3],
        ]);

        $bystatus = $attendance->to_array()['bystatus'];
        $this->assertCount(2, $bystatus);
        $this->assertSame(['Present', 'Practical'], array_column($bystatus, 'description'));
    }

    public function test_school_configured_acronyms_survive_verbatim(): void {
        // P/A/L/E are Moodle's defaults, not a contract. A school running Arabic
        // acronyms must get those back, not a translation into ours.
        $attendance = subject_attendance::from_row($this->row(), [
            'ح|حاضر' => ['acronym' => 'ح', 'description' => 'حاضر', 'count' => 5],
        ]);

        $bystatus = $attendance->to_array()['bystatus'];
        $this->assertSame('ح', $bystatus[0]['acronym']);
        $this->assertSame('حاضر', $bystatus[0]['description']);
    }

    public function test_percentage_is_rounded_to_two_places(): void {
        // 1 of 3 sessions present on a P=2 scale: 2/6.
        $attendance = subject_attendance::from_row(
            $this->row(['takensessions' => 3, 'points' => 2.0, 'maxpoints' => 6.0]),
            []
        );

        $this->assertSame(33.33, $attendance->percentage);
    }

    public function test_it_is_immutable(): void {
        $attendance = subject_attendance::from_row($this->row(), []);

        $this->expectException(\Error::class);
        $attendance->percentage = 100.0;
    }
}
