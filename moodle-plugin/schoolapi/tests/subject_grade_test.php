<?php
namespace local_schoolapi;

use local_schoolapi\dto\subject_grade;

defined('MOODLE_INTERNAL') || die();

/**
 * The percentage rule.
 *
 * Pure arithmetic over a row, so these run without a database and without fixtures —
 * which matters, because this is the calculation a parent is shown and it should be
 * cheap enough to test exhaustively.
 *
 * The scenario throughout is the one measured on a live Moodle: 90/100 on one item,
 * an excluded 10/100 on another, one ungraded. Moodle aggregates that to
 * finalgrade=90 with a per-student rawgrademax of 100.
 *
 * @package   local_schoolapi
 * @covers    \local_schoolapi\dto\subject_grade
 */
final class subject_grade_test extends \basic_testcase {

    private function row(array $overrides = []): \stdClass {
        return (object)array_merge([
            'courseid' => 2,
            'courseidnumber' => '2026-T1-G7A-MATH',
            'shortname' => '2026-T1-G7A-MATH',
            'fullname' => 'Mathematics',
            'finalgrade' => 90.0,
            'rawgrademax' => 100.0,
            'rawgrademin' => 0.0,
        ], $overrides);
    }

    public function test_percentage_uses_the_per_student_maximum(): void {
        $grade = subject_grade::from_row($this->row(), []);

        // 90 of a per-student max of 100. The excluded item is already out of the
        // denominator because Moodle removed it before this row was written.
        $this->assertSame(90.0, $grade->percentage);
    }

    public function test_it_does_not_use_the_course_wide_item_maximum(): void {
        // The course item's grademax on the same live data was 400. Had the DTO used
        // it, this student would be reported at 22.5%.
        $grade = subject_grade::from_row($this->row(), []);

        $this->assertNotEquals(22.5, $grade->percentage);
        $this->assertSame(100.0, $grade->maxgrade);
    }

    public function test_a_nonzero_minimum_shifts_the_scale(): void {
        // Scales do not always start at zero. 60 within 40..100 is halfway.
        $grade = subject_grade::from_row(
            $this->row(['finalgrade' => 70.0, 'rawgrademax' => 100.0, 'rawgrademin' => 40.0]),
            []
        );

        $this->assertSame(50.0, $grade->percentage);
    }

    public function test_no_grade_yet_is_null_not_zero(): void {
        // Zero is a mark a child can earn. "Not graded yet" is not, and reporting it
        // as 0% tells a parent their child scored nothing.
        $grade = subject_grade::from_row($this->row(['finalgrade' => null]), []);

        $this->assertNull($grade->percentage);
        $this->assertNull($grade->finalgrade);
    }

    public function test_everything_excluded_is_null_not_zero(): void {
        // Every item excused leaves an empty denominator. Same reasoning as above.
        $grade = subject_grade::from_row(
            $this->row(['finalgrade' => 0.0, 'rawgrademax' => 0.0]),
            []
        );

        $this->assertNull($grade->percentage);
    }

    public function test_a_zero_span_cannot_divide(): void {
        $grade = subject_grade::from_row(
            $this->row(['finalgrade' => 50.0, 'rawgrademax' => 50.0, 'rawgrademin' => 50.0]),
            []
        );

        $this->assertNull($grade->percentage);
    }

    public function test_a_genuine_zero_is_reported_as_zero(): void {
        // The counterpart to the null cases: a child who really scored nothing on
        // graded work must not be shown as "no grade yet".
        $grade = subject_grade::from_row($this->row(['finalgrade' => 0.0]), []);

        $this->assertSame(0.0, $grade->percentage);
    }

    public function test_counts_are_carried_through(): void {
        $grade = subject_grade::from_row(
            $this->row(),
            ['graded' => 1, 'excluded' => 1, 'pending' => 1]
        );

        $this->assertSame(1, $grade->gradedcount);
        $this->assertSame(1, $grade->excludedcount);
        $this->assertSame(1, $grade->pendingcount);
    }

    public function test_a_term_with_pending_work_is_not_complete(): void {
        $grade = subject_grade::from_row($this->row(), ['graded' => 2, 'pending' => 1]);

        $this->assertFalse($grade->to_array()['iscomplete']);
    }

    public function test_a_fully_graded_term_is_complete(): void {
        $grade = subject_grade::from_row($this->row(), ['graded' => 3, 'pending' => 0]);

        $this->assertTrue($grade->to_array()['iscomplete']);
    }

    public function test_an_empty_term_is_not_complete(): void {
        // Nothing graded and nothing pending is an empty term, not a finished one.
        $grade = subject_grade::from_row($this->row(), ['graded' => 0, 'pending' => 0]);

        $this->assertFalse($grade->to_array()['iscomplete']);
    }

    public function test_it_is_immutable(): void {
        $grade = subject_grade::from_row($this->row(), []);

        $this->expectException(\Error::class);
        $grade->percentage = 100.0;
    }
}
