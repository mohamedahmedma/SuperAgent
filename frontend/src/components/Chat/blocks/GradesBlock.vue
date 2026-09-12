<template>
  <section class="answer-block gr" :dir="dir" :lang="language">
    <header class="ab-head">
      <span class="ab-icon" aria-hidden="true"><i class="fa-solid fa-chart-simple"></i></span>
      <div class="ab-titles">
        <h4 class="ab-title">{{ copy.grades }}</h4>
        <p v-if="block.data.term_label" class="ab-meta">{{ block.data.term_label }}</p>
      </div>
    </header>

    <ul class="gr-list">
      <li v-for="(row, index) in rows" :key="index" class="gr-row">
        <div class="gr-line">
          <span class="gr-subject">{{ row.subject }}</span>
          <span v-if="row.hasGrade" class="gr-score">
            <span class="gr-value" dir="ltr">{{ row.percentText }}%</span>
            <span v-if="row.letter" class="gr-letter">{{ row.letter }}</span>
          </span>
          <span v-else class="gr-none">{{ copy.noGrade }}</span>
        </div>
        <!-- No bar at all for a blank grade: an empty bar reads as a zero, and it is not one. -->
        <div
          v-if="row.hasGrade"
          class="gr-meter"
          role="meter"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="row.fill"
          :aria-label="row.subject"
        >
          <span class="gr-fill" :style="{ inlineSize: `${row.fill}%` }"></span>
        </div>
        <div v-if="row.missing || row.inProgress" class="gr-tags">
          <span v-if="row.missing" class="ab-pill is-warning">
            <i class="fa-solid fa-circle-exclamation" aria-hidden="true"></i>{{ copy.missing(row.missing) }}
          </span>
          <span v-if="row.inProgress" class="ab-pill">
            <i class="fa-solid fa-hourglass-half" aria-hidden="true"></i>{{ copy.inProgress }}
          </span>
        </div>
      </li>
    </ul>
  </section>
</template>

<script setup lang="ts">
/**
 * A term's marks, one row per subject: the figure as the school reported it, a bar to
 * scan down, and the two caveats the markdown carries — work not submitted, a mark still in
 * progress — as labelled tags rather than colour alone.
 *
 * The bar is one hue for every subject. It shows where a mark sits on the scale, not
 * whether it is good: painting a child's weaker subject red is a judgement the school did
 * not make, and the answer is not the place to make it.
 */
import { computed } from 'vue';

import type { GradesAnswerBlock } from '@/types/chat';
import { blockCopy, blockLanguage, buildGradeRows } from '@/utils/answerBlocks';

const props = defineProps<{ block: GradesAnswerBlock }>();

const language = computed(() => blockLanguage(props.block));
const dir = computed(() => (language.value === 'ar' ? 'rtl' : 'ltr'));
const copy = computed(() => blockCopy(language.value));
const rows = computed(() => buildGradeRows(props.block.data));
</script>
