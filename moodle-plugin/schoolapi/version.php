<?php
// This file is part of local_schoolapi for Moodle.
//
// Read-only, per-student academic records for the school assistant.

/**
 * Plugin version and requirements.
 *
 * @package    local_schoolapi
 * @copyright  2026 BHCR
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$plugin->component = 'local_schoolapi';
$plugin->version   = 2026081000;
// Moodle 5.1. Pinned rather than left open because this plugin reads
// `grade_grades.rawgrademax` directly, and the aggregation semantics behind that
// column are the one thing that would break silently across a major version.
$plugin->requires  = 2025100600;
$plugin->supported = [501, 501];
$plugin->maturity  = MATURITY_ALPHA;
$plugin->release   = '0.1.0';
