<?php
/**
 * Capabilities.
 *
 * One capability, and it is READ ONLY. That is the entire reason this plugin exists.
 *
 * Core's attendance API forces a caller to hold `mod/attendance:takeattendances` or
 * `changeattendances` just to READ a register — both write-capable — so a token that
 * answers "how many days has she missed" can also rewrite attendance for the whole
 * school. `local/schoolapi:read` grants reading and nothing else, and there is no
 * write path in this plugin for a capability to unlock.
 *
 * Deliberately NOT granted to any archetype. A role that can use this is created
 * explicitly by an administrator for a named service account; no teacher, manager or
 * admin picks it up by default, so the set of things that can read student records
 * stays small and auditable.
 *
 * @package   local_schoolapi
 * @license   http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$capabilities = [
    'local/schoolapi:read' => [
        // System level: a service account is never a course participant, and requiring
        // enrolment is precisely the constraint that makes core's attendance API
        // unusable for this.
        'contextlevel' => CONTEXT_SYSTEM,
        'captype'      => 'read',
        'riskbitmask'  => RISK_PERSONAL,
        'archetypes'   => [],
    ],
];
