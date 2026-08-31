/*
 * The vocabulary of a school's ladder, in one place.
 *
 * These were local to the school screen until the subject board needed the same words. Two
 * copies of a five-entry list is not obviously a problem, which is exactly why it is worth
 * refusing: the day a school adds a division, one screen groups its rungs under the new
 * heading and the other files them under "Not yet grouped", and nothing about either screen
 * says the two lists were ever meant to agree.
 *
 * Labels are English keys, not display text. They pass through `t()` at render time, because
 * a module-level table would freeze whichever language happened to be current at import and
 * would not follow the language switch.
 */

/** The order a school lists its own divisions in, and the labels it uses. */
export const STAGES = [
  { key: 'garden', label: 'Garden' },
  { key: 'primary', label: 'Primary' },
  { key: 'preparatory', label: 'Preparatory' },
  { key: 'secondary', label: 'Secondary' },
  { key: 'unspecified', label: 'Not yet grouped' }
];

/*
 * The stages a school can switch on when it is created: the column each one is stored in and
 * the most grades the curriculum defines for it. Derived from STAGES rather than restated, so
 * the two lists cannot drift apart and a stage is named the same word in both.
 */
export const GRADE_LIMITS = { garden: 3, primary: 6, preparatory: 3, secondary: 3 };

export const SCHOOL_LEVELS = STAGES.filter((stage) => stage.key in GRADE_LIMITS).map((stage) => ({
  ...stage,
  column: stage.key === 'garden' ? 'kg_grade_count' : `${stage.key}_grade_count`,
  max: GRADE_LIMITS[stage.key]
}));

/**
 * Rungs grouped by division, in `STAGES` order, with empty divisions dropped.
 *
 * A rung with no stage is grouped under `unspecified` rather than hidden. Every rung
 * predating the stage column has one, and a ladder that silently omits half its rungs is
 * worse than one with an honest "Not yet grouped" heading.
 */
export function byStage(levels) {
  return STAGES.map((stage) => ({
    stage,
    levels: (levels || []).filter((level) => (level.stage || 'unspecified') === stage.key)
  })).filter((group) => group.levels.length > 0);
}
