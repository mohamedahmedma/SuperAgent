<?php
/**
 * Seed one student per production scenario.
 *
 *   docker exec -w /var/www/html moodle51-webserver-1 php /var/www/html/seed_scenarios.php
 *
 * ONE PATHOLOGY PER STUDENT. Every student's idnumber names the scenario they carry, so
 * a failing assertion names its own cause rather than leaving someone to bisect a
 * dataset. A student carrying three pathologies makes a failure ambiguous, and
 * ambiguity in a regression suite is how a real bug gets waved through.
 *
 * Scale is deliberately small. Fifty students that each mean something catch far more
 * than five thousand that are all the same student — and on this bind mount, a large
 * seed costs hours and proves less.
 *
 * The suite that consumes this is `regression_suite.py`, which checks two things:
 *   DIFFERENTIAL — our figure must equal what Moodle computes independently. Moodle is
 *                  the oracle; divergence is a bug in us by definition.
 *   ASSERTED     — cases where Moodle has no answer or is ambiguous, hand-computed here.
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
mt_srand(20260810);           // Deterministic: a shifting dataset cannot prove a regression.

[$options] = cli_get_params(['reset' => false, 'help' => false], ['h' => 'help']);

if (!empty($options['help'])) {
    cli_writeln("Seed one student per scenario.");
    cli_writeln("  --reset   delete data from a previous run of this script first");
    exit(0);
}

$TERM = '2026-T1';

// A namespace of its own. The suite MUST own its data: an earlier ad-hoc fixture had
// already created "2026-T1-G7A-MATH" with its own grade items, and reusing that
// shortname put two populations in one course — which showed up as pendingcount being 5
// where the scenario intended 1. The percentages were unaffected, because rawgrademax
// excludes ungraded items, but a suite whose numbers depend on what someone seeded
// last month is not a regression suite.
//
// "R7A" rather than "G7A" keeps the term prefix intact, so term filtering is still
// exercised exactly as in production.
$SUITE = 'R7A';
$started = microtime(true);

function say(string $line): void {
    cli_writeln($line);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create or fetch a student carrying one scenario. */
function seed_student(string $idnumber, string $scenario, array $overrides = []): stdClass {
    global $DB, $CFG;

    $existing = $DB->get_record('user', ['idnumber' => $idnumber]);
    if ($existing) {
        return $existing;
    }

    // Several scenarios deliberately carry an idnumber containing a quote, a percent or
    // an underscore. Those are the point of the test, but Moodle validates usernames, so
    // the username is reduced to a safe slug while the IDNUMBER keeps the hostile value.
    $slug = strtolower(preg_replace('/[^a-z0-9]/i', '', $idnumber));

    $user = (object)array_merge([
        'username' => ($slug !== '' ? $slug : 'case') . '_' . mt_rand(1000, 9999),
        'password' => 'Seed-passw0rd!',
        'firstname' => 'Case',
        'lastname' => $scenario,
        'email' => 'case' . mt_rand(100000, 999999) . '@example.invalid',
        'auth' => 'manual',
        'confirmed' => 1,
        'mnethostid' => $CFG->mnet_localhost_id,
        'idnumber' => $idnumber,
    ], $overrides);

    $user->id = user_create_user($user, true, false);
    return $DB->get_record('user', ['id' => $user->id]);
}

/** Create or fetch a course. */
function seed_course(string $shortname, array $overrides = []): stdClass {
    global $DB;

    $course = $DB->get_record('course', ['shortname' => $shortname]);
    if ($course) {
        return $course;
    }

    return create_course((object)array_merge([
        'fullname' => $shortname,
        'shortname' => $shortname,
        'idnumber' => $shortname,
        'category' => 1,
        'visible' => 1,
        'showgrades' => 1,
        // Non-zero on purpose. With startdate 0 the `sessdate >= startdate` filter is
        // inert, so a bug in it could never surface in testing.
        'startdate' => strtotime('2026-01-01'),
    ], $overrides));
}

/** Enrol, returning the user_enrolment id so a scenario can suspend it afterwards. */
function enrol_student(stdClass $course, stdClass $user, string $enrolmethod = 'manual'): ?int {
    global $DB;

    $instance = $DB->get_record('enrol',
        ['courseid' => $course->id, 'enrol' => $enrolmethod], '*', IGNORE_MULTIPLE);
    if (!$instance) {
        return null;
    }

    $roleid = $DB->get_field('role', 'id', ['shortname' => 'student']);
    enrol_get_plugin($enrolmethod)->enrol_user($instance, $user->id, $roleid);

    return $DB->get_field('user_enrolments', 'id',
        ['enrolid' => $instance->id, 'userid' => $user->id]);
}

/** Create or fetch a manual grade item. */
function seed_item(int $courseid, string $name, float $max = 100, float $min = 0): grade_item {
    $item = grade_item::fetch(['courseid' => $courseid, 'itemname' => $name, 'itemtype' => 'manual']);
    if ($item) {
        return $item;
    }
    $item = new grade_item([
        'courseid' => $courseid,
        'itemtype' => 'manual',
        'itemname' => $name,
        'gradetype' => GRADE_TYPE_VALUE,
        'grademin' => $min,
        'grademax' => $max,
    ], false);
    $item->insert();
    return $item;
}

/** The attendance activity for a course, created on demand. */
function seed_attendance_activity(stdClass $course, int $groupmode = 0): stdClass {
    global $DB;

    $module = $DB->get_record('modules', ['name' => 'attendance'], '*', MUST_EXIST);
    $cm = $DB->get_record_sql(
        "SELECT * FROM {course_modules} WHERE course = ? AND module = ? ORDER BY id",
        [$course->id, $module->id]);

    if ($cm) {
        return $cm;
    }

    $info = create_module((object)[
        'modulename' => 'attendance',
        'course' => $course->id,
        'section' => 0,
        'visible' => 1,
        'name' => 'Class register',
        'introeditor' => ['text' => '', 'format' => FORMAT_HTML, 'itemid' => 0],
        'intro' => '',
        'introformat' => FORMAT_HTML,
        'cmidnumber' => '',
        'groupmode' => $groupmode,
        'groupingid' => 0,
        'completion' => 0,
        'completionview' => 0,
        'completionexpected' => 0,
    ]);
    return $DB->get_record('course_modules', ['id' => $info->coursemodule]);
}

/** A session. `$taken = false` leaves lasttaken NULL, as Moodle does for unmarked ones. */
function seed_session(int $attendanceid, int $daysafterstart, bool $taken = true,
                      int $groupid = 0, int $statusset = 0, int $coursestart = 0): int {
    global $DB;

    return $DB->insert_record('attendance_sessions', (object)[
        'attendanceid' => $attendanceid,
        'groupid' => $groupid,
        'sessdate' => $coursestart + ($daysafterstart * 86400),
        'duration' => 3600,
        // NULL, not 0: `add_session()` inserts before defaulting the in-memory object,
        // so untaken sessions really do hold NULL in the database.
        'lasttaken' => $taken ? time() : null,
        'lasttakenby' => $taken ? 2 : 0,
        'timemodified' => time(),
        'description' => 'Seeded',
        'descriptionformat' => FORMAT_HTML,
        'statusset' => $statusset,
    ]);
}

/** Mark one student in one session. */
function mark(int $sessionid, int $userid, int $statusid): void {
    global $DB;

    $DB->insert_record('attendance_log', (object)[
        'sessionid' => $sessionid,
        'studentid' => $userid,
        'statusid' => $statusid,
        'statusset' => '',
        'timetaken' => time(),
        'takenby' => 2,
        'remarks' => '',
    ]);
}

/** Status ids for an activity, keyed by acronym. */
function statuses_for(int $attendanceid): array {
    global $DB;

    $out = [];
    foreach ($DB->get_records('attendance_statuses',
        ['attendanceid' => $attendanceid, 'deleted' => 0], 'id ASC') as $status) {
        $out[$status->acronym] = $status;
    }
    return $out;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

/**
 * Delete everything a previous run of THIS script created.
 *
 * Needed because both the students and the courses are fetched-or-created: without a
 * reset, a re-run after changing the course namespace would enrol the existing
 * scenario students into the NEW courses while leaving them in the old ones, so
 * `subject_count` assertions would fail for reasons that have nothing to do with the
 * plugin. A regression suite that cannot be re-run from a known state is not one.
 *
 * Scoped by the suite's own namespace and idnumber prefixes, so it cannot touch a
 * developer's other data on the same instance.
 */
function reset_suite_data(string $term, string $suite): void {
    global $DB;

    // course/lib.php and user/lib.php are already required at the top of this script,
    // and their paths moved under public/ in Moodle 5.x — so do not re-require them
    // relative to __DIR__ here.

    // Courses first: delete_course() takes grades, attendance, enrolments and the
    // module instances with it, which is most of the cleanup.
    $like = $DB->sql_like('shortname', ':pattern');
    $courses = $DB->get_records_select('course', $like,
        ['pattern' => $DB->sql_like_escape("{$term}-{$suite}-") . '%']);
    foreach ($courses as $course) {
        delete_course($course, false);
    }
    $extra = $DB->get_records_list('course', 'shortname',
        ["2026-T2-{$suite}-MATH", 'LEGACY-COURSE-NO-TERM']);
    foreach ($extra as $course) {
        delete_course($course, false);
    }
    say('  deleted ' . (count($courses) + count($extra)) . ' suite courses');

    // Then the students. delete_user() clears the idnumber, which is what frees it for
    // the next run — a hard row delete would leave the unique-ish idnumber in place on
    // a soft-deleted row and collide.
    $deleted = 0;
    foreach (['GRD-%', 'ATT-%', 'LIF-%', 'ID-%', 'ID\_%'] as $pattern) {
        $sql = $DB->sql_like('idnumber', ':pattern', true, true, false, '\\');
        foreach ($DB->get_records_select('user', "{$sql} AND deleted = 0",
            ['pattern' => $pattern]) as $user) {
            delete_user($user);
            $deleted++;
        }
    }
    say("  deleted {$deleted} scenario students");
}

if (!empty($options['reset'])) {
    say('Resetting previous suite data...');
    reset_suite_data($TERM, $SUITE);
    purge_all_caches();
    say('');
}

// ---------------------------------------------------------------------------
// Courses
// ---------------------------------------------------------------------------

say('Courses...');

// The main course most scenarios share — students really do share classes, and sharing
// exercises the per-student aggregation bounds that a one-student course would not.
$math = seed_course("{$TERM}-{$SUITE}-MATH", ['fullname' => 'Mathematics R7A T1']);
$sci = seed_course("{$TERM}-{$SUITE}-SCI", ['fullname' => 'Science R7A T1']);

// Course-level pathologies need their own courses.
$hidden = seed_course("{$TERM}-{$SUITE}-HIDDEN", ['fullname' => 'Hidden course', 'visible' => 0]);
$unconventional = seed_course('LEGACY-COURSE-NO-TERM', ['fullname' => 'Outside the term convention']);
$nextterm = seed_course("2026-T2-{$SUITE}-MATH", ['fullname' => 'Mathematics R7A T2',
    'startdate' => strtotime('2026-04-01')]);
$grouped = seed_course("{$TERM}-{$SUITE}-GROUPS", ['fullname' => 'Grouped course',
    'groupmode' => SEPARATEGROUPS, 'groupmodeforce' => 1]);

// Disabling an enrolment INSTANCE affects every student on it, so that scenario needs a
// course of its own. Putting it on a shared course would silently break every other
// student there — the kind of cross-contamination that makes a suite untrustworthy and
// is very hard to spot once the failures start.
$disabledenrol = seed_course("{$TERM}-{$SUITE}-DISABLED", ['fullname' => 'Disabled enrolment method']);

say('  7 courses');

// ---------------------------------------------------------------------------
// Attendance activities and their status sets
// ---------------------------------------------------------------------------

say('Attendance activities...');
$mathcm = seed_attendance_activity($math);
$mathatt = (int)$mathcm->instance;
$mathstatus = statuses_for($mathatt);

$groupedcm = seed_attendance_activity($grouped, SEPARATEGROUPS);
$groupedatt = (int)$groupedcm->instance;
$groupedstatus = statuses_for($groupedatt);
say("  math attendance id {$mathatt}, grouped id {$groupedatt}");
say('  statuses: ' . implode(', ', array_map(
    fn($s) => "{$s->acronym}={$s->grade}", array_values($mathstatus))));

// ---------------------------------------------------------------------------
// The scenarios
// ---------------------------------------------------------------------------

say('');
say('Scenarios...');

$registry = [];   // idnumber => [scenario, expectation] for the harness manifest.

/**
 * Register a scenario student and describe what the suite should expect.
 *
 * `expect` is written here, next to the setup that produces it, so the two cannot drift
 * apart the way a separate expectations file always eventually does.
 */
function scenario(string $idnumber, string $title, array $expect, callable $setup): void {
    global $registry;

    $user = seed_student($idnumber, $title);
    $setup($user);

    // The Moodle user id is recorded ONLY for the test harness. The plugin's own
    // contract never exposes it — the facade keys on the school's student number, and a
    // Moodle id is an internal detail that would leak Moodle into a contract meant to
    // outlive it. But the differential oracle,
    // `gradereport_user_get_grade_items(courseid, userid)`, is a core function that
    // takes one, and resolving it over the wire would cost a ~48s call per student.
    $registry[$idnumber] = ['title' => $title, 'userid' => (int)$user->id] + $expect;
    say("  {$idnumber}  {$title}");
}

$mathitems = [
    seed_item($math->id, 'Assessment 1'),
    seed_item($math->id, 'Assessment 2'),
    seed_item($math->id, 'Assessment 3'),
];
$sciitems = [seed_item($sci->id, 'Sci 1'), seed_item($sci->id, 'Sci 2')];

// ---- Grades ---------------------------------------------------------------

scenario('GRD-BASELINE', 'All graded, nothing excluded',
    ['grades_pct' => 80.0, 'differential' => true],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        foreach ($mathitems as $item) { $item->update_final_grade($u->id, 80); }
    });

scenario('GRD-ONE-EXCLUDED', 'One item excluded leaves the denominator',
    ['grades_pct' => 90.0, 'differential' => true,
     'note' => '90 and 10 with the 10 excluded is 90%, not 50%'],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 90);
        $mathitems[1]->update_final_grade($u->id, 10);
        $mathitems[2]->update_final_grade($u->id, 90);
        $g = grade_grade::fetch(['itemid' => $mathitems[1]->id, 'userid' => $u->id]);
        if ($g) { $g->set_excluded(1); }
    });

scenario('GRD-ALL-EXCLUDED', 'Every item excluded yields null, never zero',
    ['grades_pct' => null, 'differential' => false,
     'note' => 'empty denominator; 0% would claim the child scored nothing'],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        foreach ($mathitems as $item) {
            $item->update_final_grade($u->id, 50);
            $g = grade_grade::fetch(['itemid' => $item->id, 'userid' => $u->id]);
            if ($g) { $g->set_excluded(1); }
        }
    });

scenario('GRD-NONE', 'Enrolled but never graded',
    ['grades_pct' => null, 'differential' => false],
    function ($u) use ($math) { enrol_student($math, $u); });

scenario('GRD-ALL-ZERO', 'A genuine zero is zero, not null',
    ['grades_pct' => 0.0, 'differential' => true,
     'note' => 'the counterpart to GRD-NONE; these must not be conflated'],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        foreach ($mathitems as $item) { $item->update_final_grade($u->id, 0); }
    });

scenario('GRD-FULL', 'Full marks',
    ['grades_pct' => 100.0, 'differential' => true],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        foreach ($mathitems as $item) { $item->update_final_grade($u->id, 100); }
    });

scenario('GRD-PARTIAL', 'One item still ungraded',
    // TWO pending, not one. Assessment 3 is ungraded, and so is the attendance
    // activity's OWN grade item — creating a "Class register" in a course adds a
    // gradeable item to it, whether or not anybody has marked a register.
    //
    // That is correct rather than a quirk to filter out, and it has a consequence
    // worth knowing: if a school enables attendance grading, attendance feeds the
    // course total, so the percentage a parent is shown for Mathematics already
    // includes it. Excluding attendance items here would make our item list disagree
    // with the gradebook, which is the one failure this design exists to avoid.
    ['grades_pct' => 75.0, 'pendingcount' => 2, 'differential' => true,
     'note' => 'Assessment 3 plus the attendance grade item; the ungraded items must '
             . 'leave the denominator, not score zero'],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 75);
        $mathitems[1]->update_final_grade($u->id, 75);
    });

scenario('GRD-DECIMAL', 'Fractional marks round to two places',
    ['grades_pct' => 66.67, 'differential' => true],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        foreach ($mathitems as $item) { $item->update_final_grade($u->id, 66.666); }
    });

scenario('GRD-ATTENDANCE-GRADED', 'Attendance graded: course total and academic diverge',
    // THE case that justifies returning two figures.
    //
    // Three assessments at 80 (240/300 = 80%) plus a graded attendance item at 20/100.
    // Moodle's course total is (240+20)/400 = 65%, because attendance is a gradeable
    // item like any other. The academic figure ignores it and reports 80%.
    //
    // Both are true and they answer different questions. A parent asking "how is she
    // doing in maths" almost always means the 80%; the school's official course total
    // is the 65%. Reporting only one would be wrong in one direction or the other, and
    // silently choosing would be worse than either.
    ['grades_pct' => 65.0, 'academic_pct' => 80.0, 'differential' => true,
     'note' => 'course total includes attendance; academic excludes it'],
    function ($u) use ($math, $mathitems, $mathatt) {
        enrol_student($math, $u);
        foreach ($mathitems as $item) { $item->update_final_grade($u->id, 80); }

        // The attendance activity's own grade item — created by the activity, graded
        // here the way a school with attendance grading enabled would have it.
        $attitem = grade_item::fetch([
            'courseid' => $math->id,
            'itemtype' => 'mod',
            'itemmodule' => 'attendance',
            'iteminstance' => $mathatt,
        ]);
        if ($attitem) {
            $attitem->update_final_grade($u->id, 20);
        }
    });

scenario('GRD-TWO-COURSES', 'Two subjects report independently',
    ['subject_count' => 2, 'differential' => true],
    function ($u) use ($math, $sci, $mathitems, $sciitems) {
        enrol_student($math, $u);
        enrol_student($sci, $u);
        foreach ($mathitems as $item) { $item->update_final_grade($u->id, 60); }
        foreach ($sciitems as $item) { $item->update_final_grade($u->id, 90); }
    });

scenario('GRD-OTHER-TERM', 'Term filter excludes another term',
    ['subject_count' => 0, 'term' => '2026-T1-', 'differential' => false,
     'note' => 'enrolled only in T2; asking for T1 must return nothing'],
    function ($u) use ($nextterm) {
        enrol_student($nextterm, $u);
        $item = seed_item($nextterm->id, 'T2 Assessment');
        $item->update_final_grade($u->id, 70);
    });

scenario('GRD-HIDDEN-COURSE', 'A hidden course is not reported',
    ['subject_count' => 0, 'differential' => false,
     'note' => 'policy choice: c.visible = 1 in the repository'],
    function ($u) use ($hidden) {
        enrol_student($hidden, $u);
        $item = seed_item($hidden->id, 'Hidden Assessment');
        $item->update_final_grade($u->id, 88);
    });

scenario('GRD-NO-TERM-CONVENTION', 'A course outside the naming convention',
    ['subject_count' => 0, 'term' => '2026-T1-', 'differential' => false],
    function ($u) use ($unconventional) {
        enrol_student($unconventional, $u);
        $item = seed_item($unconventional->id, 'Legacy Assessment');
        $item->update_final_grade($u->id, 55);
    });

// ---- Attendance -----------------------------------------------------------

$mathsessions = [];
for ($d = 1; $d <= 6; $d++) {
    // Every third session left untaken, so the lasttaken filter is exercised.
    $mathsessions[] = seed_session($mathatt, $d, ($d % 3) !== 0, 0, 0, (int)$math->startdate);
}
$takensessions = [];
foreach ($mathsessions as $i => $sid) {
    if (($i + 1) % 3 !== 0) { $takensessions[] = $sid; }
}

scenario('ATT-ALL-PRESENT', 'Present at every taken session',
    ['attendance_pct' => 100.0, 'taken' => count($takensessions), 'differential' => true],
    function ($u) use ($math, $takensessions, $mathstatus) {
        enrol_student($math, $u);
        foreach ($takensessions as $sid) { mark($sid, $u->id, $mathstatus['P']->id); }
    });

scenario('ATT-ALL-ABSENT', 'Absent throughout is a real zero',
    ['attendance_pct' => 0.0, 'differential' => true],
    function ($u) use ($math, $takensessions, $mathstatus) {
        enrol_student($math, $u);
        foreach ($takensessions as $sid) { mark($sid, $u->id, $mathstatus['A']->id); }
    });

scenario('ATT-ALL-EXCUSED', 'Excused is half credit, not exemption',
    ['attendance_pct' => 50.0, 'differential' => true,
     'note' => 'E=1 against a set max of 2. A naive present/total count says 0%'],
    function ($u) use ($math, $takensessions, $mathstatus) {
        enrol_student($math, $u);
        foreach ($takensessions as $sid) { mark($sid, $u->id, $mathstatus['E']->id); }
    });

scenario('ATT-ALL-LATE', 'Late is weighted between present and absent',
    ['attendance_pct' => 50.0, 'differential' => true],
    function ($u) use ($math, $takensessions, $mathstatus) {
        enrol_student($math, $u);
        foreach ($takensessions as $sid) { mark($sid, $u->id, $mathstatus['L']->id); }
    });

scenario('ATT-MIXED', 'A realistic mixture',
    ['differential' => true,
     'note' => 'no hand-computed expectation; Moodle is the oracle'],
    function ($u) use ($math, $takensessions, $mathstatus) {
        enrol_student($math, $u);
        $order = ['P', 'P', 'L', 'A'];
        foreach ($takensessions as $i => $sid) {
            mark($sid, $u->id, $mathstatus[$order[$i % count($order)]]->id);
        }
    });

scenario('ATT-NO-REGISTER', 'Enrolled but never marked',
    ['attendance_pct' => null, 'taken' => 0, 'differential' => false,
     'note' => 'a child cannot be absent from a class nobody recorded'],
    function ($u) use ($math) { enrol_student($math, $u); });

scenario('ATT-UNTAKEN-ONLY', 'Marked only in an untaken session',
    ['attendance_pct' => null, 'taken' => 0, 'differential' => false,
     'note' => 'lasttaken IS NULL must exclude the session entirely'],
    function ($u) use ($math, $mathsessions, $mathstatus) {
        enrol_student($math, $u);
        // Index 2 is the third session, left untaken above.
        mark($mathsessions[2], $u->id, $mathstatus['P']->id);
    });

scenario('ATT-BEFORE-COURSE-START', 'A session dated before the course began',
    ['attendance_pct' => null, 'taken' => 0, 'differential' => false,
     'note' => 'sessdate >= course.startdate; core calls these hidden sessions'],
    function ($u) use ($math, $mathatt, $mathstatus) {
        enrol_student($math, $u);
        $sid = seed_session($mathatt, -30, true, 0, 0, (int)$math->startdate);
        mark($sid, $u->id, $mathstatus['P']->id);
    });

// ---- Lifecycle ------------------------------------------------------------

scenario('LIF-SUSPENDED-USER', 'A suspended account is not found',
    ['found' => false, 'differential' => false],
    function ($u) use ($math, $mathitems, $DB) {
        enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 70);
        $DB->set_field('user', 'suspended', 1, ['id' => $u->id]);
    });

scenario('LIF-SUSPENDED-ENROLMENT', 'A suspended enrolment is excluded',
    ['subject_count' => 0, 'differential' => false],
    function ($u) use ($math, $mathitems, $DB) {
        $ueid = enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 70);
        if ($ueid) { $DB->set_field('user_enrolments', 'status', 1, ['id' => $ueid]); }
    });

scenario('LIF-DISABLED-ENROL-METHOD', 'A disabled enrolment instance is excluded',
    ['subject_count' => 0, 'differential' => false,
     'note' => 'e.status, not ue.status — the bug the design review caught'],
    function ($u) use ($disabledenrol) {
        enrol_student($disabledenrol, $u);
        $item = seed_item($disabledenrol->id, 'Assessment');
        $item->update_final_grade($u->id, 70);
    });

scenario('LIF-UNENROLLED-WITH-DATA', 'Grades left behind after unenrolment',
    ['subject_count' => 0, 'differential' => false,
     'note' => 'grade_grades rows outlive the enrolment; EXISTS must gate them'],
    function ($u) use ($math, $mathitems, $DB) {
        $ueid = enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 70);
        if ($ueid) { $DB->delete_records('user_enrolments', ['id' => $ueid]); }
    });

// ---- Identity and injection ----------------------------------------------

scenario('ID-WITH-SPACES', 'An idnumber that differs only by whitespace',
    ['found' => true, 'differential' => false],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 65);
    });

scenario("ID-QUOTE'S", 'An idnumber containing a quote',
    ['found' => true, 'differential' => false,
     'note' => 'parameterised queries must handle it; no SQL error'],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 65);
    });

scenario('ID-100%-PCT', 'An idnumber containing a LIKE wildcard',
    ['found' => true, 'differential' => false,
     'note' => 'must match exactly, never as a pattern'],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 65);
    });

scenario('ID_UNDERSCORE', 'An idnumber containing a LIKE single-char wildcard',
    ['found' => true, 'differential' => false],
    function ($u) use ($math, $mathitems) {
        enrol_student($math, $u);
        $mathitems[0]->update_final_grade($u->id, 65);
    });

// Regrade every touched course ONCE, after all grades are in. Per-grade regrading is
// the difference between a two-minute seed and an afternoon.
say('');
say('Regrading...');
foreach ([$math, $sci, $hidden, $unconventional, $nextterm, $grouped, $disabledenrol] as $course) {
    grade_regrade_final_grades($course->id);
}

// Disable the enrolment INSTANCE on its dedicated course, after grading, so the grade
// rows exist and only the instance status makes them unreportable. That is exactly the
// production shape: a school turns off self-enrolment and leaves live user_enrolments
// rows behind a dead instance.
$deadinstance = $DB->get_record('enrol',
    ['courseid' => $disabledenrol->id, 'enrol' => 'manual'], '*', IGNORE_MULTIPLE);
if ($deadinstance) {
    $DB->set_field('enrol', 'status', 1, ['id' => $deadinstance->id]);
    say('  disabled the enrolment instance on ' . $disabledenrol->shortname);
}

purge_all_caches();

// ---------------------------------------------------------------------------
// Manifest — the harness reads this rather than duplicating expectations
// ---------------------------------------------------------------------------

$manifest = [
    'term' => $TERM . '-',
    'seeded_at' => date('c'),
    'courses' => [
        'math' => $math->idnumber,
        'sci' => $sci->idnumber,
    ],
    'scenarios' => $registry,
];

$path = '/var/www/moodledata/schoolapi_scenarios.json';
file_put_contents($path, json_encode($manifest, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));

say('');
say('=== seeded ' . count($registry) . ' scenarios in ' . round(microtime(true) - $started, 1) . 's ===');
say("manifest: {$path}");
