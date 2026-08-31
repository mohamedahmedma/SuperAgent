/*
 * Entry point: the stylesheet order, the route table, and the one call that mounts the app.
 *
 * **The import order of the four stylesheets is load-bearing** and is the only place it is
 * expressed, which is an improvement on the previous build where it was a list of script tags
 * with a comment asking the reader not to reorder them:
 *
 *   bootstrap    the framework, first, so everything after it can override
 *   tokens       our variables — the palette, the type scale, the motion
 *   theme        the bridge that re-points Bootstrap's `--bs-*` at those tokens
 *   base + sis   document rules, then the handful of components Bootstrap has no equivalent
 *                for (bidirectional names, the school strip, the stage ladder)
 *
 * Bootstrap comes from npm and is bundled rather than fetched, so nothing reaches a CDN at
 * runtime — schools run this offline or behind a filter, and that constraint has not changed.
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import 'bootstrap/dist/css/bootstrap.min.css';
import './styles/tokens.css';
import './styles/theme.css';
import './styles/base.css';
import './styles/sis.css';

import { Router } from './router.js';
import { App } from './App.jsx';
import { School } from './views/School.jsx';
import { Level } from './views/Level.jsx';
import { Year } from './views/Year.jsx';
import { Klass } from './views/Klass.jsx';
import { Student } from './views/Student.jsx';
import { Roster } from './views/Roster.jsx';
import { Guardians } from './views/Guardians.jsx';
import { Marks } from './views/Marks.jsx';
import { Batches } from './views/Batches.jsx';
import { Roles } from './views/Roles.jsx';
import { GradeAssignments } from './views/GradeAssignments.jsx';
import { TeacherSetup } from './views/TeacherSetup.jsx';
import { Attendance } from './views/Attendance.jsx';

/*
 * Schools -> School -> Rung -> Class -> Child is a containment hierarchy, and the first five
 * entries are its rungs. `level`, `year`, `class` and `student` are reached by clicking rather
 * than from the nav: each needs a code in the URL to mean anything, and a nav link to "Class"
 * with no class chosen is a link to an error message.
 *
 * `school` is first, and that position is load-bearing — the router sends an unrecognised hash
 * to `routes[0]`, and the top of the hierarchy is the right place to land.
 */
const ROUTES = [
  { name: 'school', view: School, title: 'School' },
  { name: 'level', view: Level, title: 'Rung' },
  { name: 'year', view: Year, title: 'Year' },
  { name: 'class', view: Klass, title: 'Class' },
  { name: 'student', view: Student, title: 'Student' },
  { name: 'roster', view: Roster, title: 'Roster' },
  { name: 'guardians', view: Guardians, title: 'Guardians' },
  { name: 'marks', view: Marks, title: 'Marks' },
  { name: 'batches', view: Batches, title: 'Batches' },
  { name: 'roles', view: Roles, title: 'Teacher roles' },
  { name: 'teacherSetup', view: TeacherSetup, title: 'Teacher setup' },
  { name: 'gradeAssignments', view: GradeAssignments, title: 'Class assignments' },
  { name: 'attendance', view: Attendance, title: 'Take attendance' }
];

/* The document title follows the route: browser history and a taskbar full of tabs are both
   unreadable when nine screens share one title. */
Router.subscribe((current) => {
  if (current) document.title = `${current.route.title} — SIS Registrar`;
});

Router.start(ROUTES);

createRoot(document.getElementById('app')).render(
  <StrictMode>
    <App />
  </StrictMode>
);
