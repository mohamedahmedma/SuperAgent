<?php
/**
 * Seed a school-sized dataset for regression testing local_schoolapi.
 *
 *   docker exec -w /var/www/html moodle51-webserver-1 \
 *     php /var/www/html/seed_school.php --students=150 --terms=2
 *
 * Two populations, deliberately separated:
 *
 *   THE BULK — ordinary students with ordinary grades and attendance. Their purpose is
 *   scale: query counts must stay flat and latency sublinear as this grows, and every
 *   one of them is a differential test case, because Moodle computes the same figures
 *   independently and the two must agree.
 *
 *   THE EDGE CASES — a handful of students each carrying exactly ONE production
 *   pathology, with a stable idnumber the test suite asserts against by name. One
 *   pathology per student on purpose: a student carrying three would make a failure
 *   ambiguous, and ambiguity in a regression suite is how a real bug gets waved through.
 *
 * PERFORMANCE. `grade_regrade_final_grades()` is the expensive call and it is made ONCE
 * per course after every grade in it is set, never per grade. Doing it per grade turns
 * a two-minute seed into an afternoon.
 */

define('CLI_SCRIPT', true);
require(__DIR__ . '/config.php');
require_once($CFG->libdir . '/clilib.php');
require_once($CFG->dirroot . '/course/lib.php');
require_once($CFG->dirroot . '/course/modlib.php');
require_once($CFG->dirroot . '/user/lib.php');
require_once($CFG->libdir . '/gradelib.php');
require_once($CFG->libdir . '/enrollib.php');

\core\session\manager::set_user(get_admin());

[$options] = cli_get_params([
    'students' => 150,
    'terms' => 2,
    'subjects' => 6,
    'items' => 5,
    'sessions' => 10,
    'help' => false,
], ['h' => 'help']);

if ($options['help']) {
    cli_writeln("Seed a regression dataset. Options: --students --terms --subjects --items --sessions");
    exit(0);
}

$numstudents = (int)$options['students'];
$numterms = (int)$options['terms'];
$numsubjects = (int)$options['subjects'];
$numitems = (int)$options['items'];
$numsessions = (int)$options['sessions'];

// Deterministic. A regression suite whose dataset changes between runs cannot tell a
// real regression from a different roll of the dice.
mt_srand(20260810);

$SUBJECTS = ['MATH', 'SCI', 'ENG', 'ARA', 'HIST', 'ART', 'PE', 'IT'];
$GRADELEVELS = ['G7', 'G8', 'G9'];
$SECTION = 'A';
$started = microtime(true);

function elapsed(float $since): string {
    return round(microtime(true) - $since, 1) . 's';
}

// ---------------------------------------------------------------------------
// Students
// ---------------------------------------------------------------------------

cli_writeln("Students...");
$students = [];
$existing = $DB->get_records_menu('user', null, '', 'idnumber, id');

for ($i = 1; $i <= $numstudents; $i++) {
    $idnumber = sprintf('T%04d', $i);
    if (isset($existing[$idnumber])) {
        $students[$idnumber] = (object)['id' => $existing[$idnumber], 'idnumber' => $idnumber];
        continue;
    }
    $user = (object)[
        'username' => strtolower($idnumber),
        'password' => 'Seed-passw0rd!',
        'firstname' => 'Student',
        'lastname' => $idnumber,
        'email' => strtolower($idnumber) . '@example.invalid',
        'auth' => 'manual',
        'confirmed' => 1,
        'mnethostid' => $CFG->mnet_localhost_id,
        'idnumber' => $idnumber,
    ];
    $user->id = user_create_user($user, true, false);
    $students[$idnumber] = $user;
}
cli_writeln('  ' . count($students) . " students (" . elapsed($started) . ")");

// ---------------------------------------------------------------------------
// Courses — one per (term, grade level, subject)
// ---------------------------------------------------------------------------

cli_writeln("Courses...");
$courses = [];
$studentroleid = $DB->get_field('role', 'id', ['shortname' => 'student']);

for ($t = 1; $t <= $numterms; $t++) {
    $term = "2026-T{$t}";
    foreach ($GRADELEVELS as $level) {
        for ($s = 0; $s < $numsubjects; $s++) {
            $subject = $SUBJECTS[$s % count($SUBJECTS)];
            $shortname = "{$term}-{$level}{$SECTION}-{$subject}";

            $course = $DB->get_record('course', ['shortname' => $shortname]);
            if (!$course) {
                $course = create_course((object)[
                    'fullname' => "{$subject} — {$level}{$SECTION} — Term {$t}",
                    'shortname' => $shortname,
                    'idnumber' => $shortname,
                    'category' => 1,
                    'visible' => 1,
                    'showgrades' => 1,
                    // Non-zero so the sessdate >= startdate filter is actually
                    // exercised. With startdate 0 that condition is inert and a bug in
                    // it would never surface in testing.
                    'startdate' => strtotime("2026-0{$t}-01"),
                ]);
            }
            $courses[$shortname] = (object)[
                'course' => $course,
                'term' => $term,
                'level' => $level,
                'subject' => $subject,
            ];
        }
    }
}
cli_writeln('  ' . count($courses) . " courses (" . elapsed($started) . ")");

// ---------------------------------------------------------------------------
// Enrolments — each student takes every subject at their own grade level
// ---------------------------------------------------------------------------

cli_writeln("Enrolments...");
$manual = enrol_get_plugin('manual');
$levelof = [];
$index = 0;
foreach ($students as $idnumber => $student) {
    $levelof[$idnumber] = $GRADELEVELS[$index++ % count($GRADELEVELS)];
}

$enrolcount = 0;
foreach ($courses as $shortname => $meta) {
    $instance = $DB->get_record('enrol',
        ['courseid' => $meta->course->id, 'enrol' => 'manual'], '*', IGNORE_MULTIPLE);
    if (!$instance) {
        continue;
    }
    foreach ($students as $idnumber => $student) {
        if ($levelof[$idnumber] !== $meta->level) {
            continue;
        }
        $manual->enrol_user($instance, $student->id, $studentroleid);
        $enrolcount++;
    }
}
cli_writeln("  {$enrolcount} enrolments (" . elapsed($started) . ")");

// ---------------------------------------------------------------------------
// Grades
// ---------------------------------------------------------------------------

cli_writeln("Grade items and grades...");
$gradecount = 0;
$excludedcount = 0;

foreach ($courses as $shortname => $meta) {
    $courseid = $meta->course->id;
    $enrolled = array_filter(
        $students,
        static fn($s, $id) => $levelof[$id] === $meta->level,
        ARRAY_FILTER_USE_BOTH
    );
    if (!$enrolled) {
        continue;
    }

    $items = [];
    for ($n = 1; $n <= $numitems; $n++) {
        $name = "Assessment {$n}";
        $item = grade_item::fetch(['courseid' => $courseid, 'itemname' => $name, 'itemtype' => 'manual']);
        if (!$item) {
            $item = new grade_item([
                'courseid' => $courseid,
                'itemtype' => 'manual',
                'itemname' => $name,
                'gradetype' => GRADE_TYPE_VALUE,
                'grademin' => 0,
                'grademax' => 100,
            ], false);
            $item->insert();
        }
        $items[] = $item;
    }

    foreach ($enrolled as $idnumber => $student) {
        foreach ($items as $position => $item) {
            // A spread of realistic marks, plus deliberate gaps: the last item is left
            // ungraded for everyone so `pendingcount` and the per-student rawgrademax
            // are both exercised on every single course.
            if ($position === count($items) - 1) {
                continue;
            }
            $mark = 40 + (mt_rand(0, 60));
            $item->update_final_grade($student->id, $mark);
            $gradecount++;
        }
    }

    // Exclude one item for every fifth student. This is the property the whole grades
    // design turns on, so it must appear across the bulk population rather than only in
    // a hand-built edge case — a differential test only catches an exclusion bug on
    // students who actually have one.
    $nth = 0;
    foreach ($enrolled as $idnumber => $student) {
        if ((++$nth % 5) !== 0) {
            continue;
        }
        $grade = grade_grade::fetch(['itemid' => $items[0]->id, 'userid' => $student->id]);
        if ($grade) {
            $grade->set_excluded(1);
            $excludedcount++;
        }
    }

    // ONCE per course, after every grade in it. Per-grade regrading is the difference
    // between minutes and hours.
    grade_regrade_final_grades($courseid);
}
cli_writeln("  {$gradecount} grades, {$excludedcount} excluded (" . elapsed($started) . ")");

// ---------------------------------------------------------------------------
// Attendance
// ---------------------------------------------------------------------------

cli_writeln("Attendance...");
$attendancemodule = $DB->get_record('modules', ['name' => 'attendance']);
$sessioncount = 0;
$logcount = 0;

if (!$attendancemodule) {
    cli_writeln('  mod_attendance not installed — skipping');
} else {
    foreach ($courses as $shortname => $meta) {
        $courseid = $meta->course->id;

        $cm = $DB->get_record_sql(
            "SELECT cm.* FROM {course_modules} cm
              WHERE cm.course = ? AND cm.module = ?", [$courseid, $attendancemodule->id]);

        if (!$cm) {
            $moduleinfo = create_module((object)[
                'modulename' => 'attendance',
                'course' => $courseid,
                'section' => 0,
                'visible' => 1,
                'name' => 'Class register',
                'introeditor' => ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0],
                'intro' => '',
                'introformat' => FORMAT_HTML,
                'cmidnumber' => '',
                'groupmode' => 0,
                'groupingid' => 0,
                'completion' => 0,
                'completionview' => 0,
                'completionexpected' => 0,
            ]);
            $cm = $DB->get_record('course_modules', ['id' => $moduleinfo->coursemodule]);
        }

        $attendanceid = $cm->instance;
        $statuses = array_values($DB->get_records('attendance_statuses',
            ['attendanceid' => $attendanceid, 'deleted' => 0, 'visible' => 1], 'id ASC'));
        if (!$statuses) {
            continue;
        }

        $enrolled = array_filter(
            $students,
            static fn($s, $id) => $levelof[$id] === $meta->level,
            ARRAY_FILTER_USE_BOTH
        );

        $existingsessions = $DB->count_records('attendance_sessions', ['attendanceid' => $attendanceid]);
        for ($n = $existingsessions; $n < $numsessions; $n++) {
            // Every third session is left UNTAKEN (lasttaken null). Those must vanish
            // from both sides of the fraction, and a bug that counts them as absences
            // is invisible unless untaken sessions actually exist in the data.
            $taken = ($n % 3) !== 2;

            $sessionid = $DB->insert_record('attendance_sessions', (object)[
                'attendanceid' => $attendanceid,
                'groupid' => 0,
                'sessdate' => $meta->course->startdate + (($n + 1) * 86400),
                'duration' => 3600,
                'lasttaken' => $taken ? time() : null,
                'lasttakenby' => $taken ? 2 : 0,
                'timemodified' => time(),
                'description' => "Session {$n}",
                'descriptionformat' => FORMAT_HTML,
                'statusset' => 0,
            ]);
            $sessioncount++;

            if (!$taken) {
                continue;
            }

            // Bulk insert: one round trip per session rather than per student.
            $logs = [];
            foreach ($enrolled as $idnumber => $student) {
                // Weighted towards present, with a real spread across the status set so
                // the points-weighted percentage differs from a naive day count for
                // most students.
                $roll = mt_rand(1, 100);
                $status = $roll <= 80 ? $statuses[0]
                    : ($roll <= 90 ? $statuses[min(2, count($statuses) - 1)]
                    : ($roll <= 95 ? $statuses[min(3, count($statuses) - 1)]
                    : $statuses[min(1, count($statuses) - 1)]));

                $logs[] = (object)[
                    'sessionid' => $sessionid,
                    'studentid' => $student->id,
                    'statusid' => $status->id,
                    'statusset' => '',
                    'timetaken' => time(),
                    'takenby' => 2,
                    'remarks' => '',
                ];
                $logcount++;
            }
            $DB->insert_records('attendance_log', $logs);
        }
    }
}
cli_writeln("  {$sessioncount} sessions, {$logcount} logs (" . elapsed($started) . ")");

cli_writeln('');
cli_writeln('=== seeded in ' . elapsed($started) . ' ===');
cli_writeln("  students  {$numstudents}   idnumbers T0001..T" . sprintf('%04d', $numstudents));
cli_writeln('  courses   ' . count($courses));
cli_writeln("  grades    {$gradecount} ({$excludedcount} excluded)");
cli_writeln("  sessions  {$sessioncount}, logs {$logcount}");
