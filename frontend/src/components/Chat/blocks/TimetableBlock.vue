<template>
  <section class="answer-block tt" :dir="dir" :lang="language">
    <header class="ab-head">
      <span class="ab-icon" aria-hidden="true"><i class="fa-solid fa-calendar-days"></i></span>
      <div class="ab-titles">
        <h4 class="ab-title">{{ copy.timetable }}</h4>
        <p v-if="meta" class="ab-meta">{{ meta }}</p>
      </div>
    </header>

    <!-- Any column narrower than the whole week — which is every phone: one day at a time. -->
    <div class="tt-compact">
      <div
        v-if="view.days.length > 1"
        class="tt-tabs"
        role="tablist"
        :aria-label="copy.days"
        @keydown="onTabKeydown"
      >
        <button
          v-for="(day, index) in view.days"
          :id="`${uid}-tab-${index}`"
          :key="day.key || index"
          :ref="(el) => rememberTab(el, index)"
          type="button"
          role="tab"
          class="tt-tab"
          :class="{ 'is-selected': index === selected, 'is-today': day.relative === 'today' }"
          :aria-selected="index === selected ? 'true' : 'false'"
          :aria-controls="`${uid}-panel`"
          :aria-label="day.relative ? `${day.label} (${relativeWord(day.relative)})` : day.label"
          :tabindex="index === selected ? 0 : -1"
          @click="select(index, { reveal: true })"
        >
          <span class="tt-tab-label">{{ tabLabel(day, language) }}</span>
          <span v-if="day.relative === 'today'" class="tt-tab-dot" aria-hidden="true"></span>
        </button>
      </div>
      <div v-else class="tt-day-head">
        <span class="tt-day-name">{{ current.label }}</span>
        <span
          v-if="current.relative"
          class="ab-pill"
          :class="{ 'is-accent': current.relative === 'today' }"
        >{{ relativeWord(current.relative) }}</span>
      </div>

      <div
        :id="`${uid}-panel`"
        class="tt-panel"
        :role="view.days.length > 1 ? 'tabpanel' : undefined"
        :aria-labelledby="view.days.length > 1 ? `${uid}-tab-${selected}` : undefined"
        @touchstart.passive="onTouchStart"
        @touchend.passive="onTouchEnd"
      >
        <Transition :name="`tt-slide-${direction}`" mode="out-in">
          <ol :key="selected" class="tt-list">
            <li
              v-for="(item, index) in current.items"
              :key="index"
              class="tt-item"
              :class="[
                `is-${item.kind}`,
                { 'is-now': item.isNow, 'is-free': item.kind === 'lesson' && item.isFree },
              ]"
            >
              <span class="tt-num">
                <template v-if="item.kind === 'lesson'">{{ item.period }}</template>
                <i v-else class="fa-solid fa-mug-hot" aria-hidden="true"></i>
              </span>
              <!-- "Now" rides inline after the name rather than taking a column of its own,
                   so flagging a lesson never costs its row the width to stay on one line. -->
              <span class="tt-subject">{{ itemTitle(item) }}<span
                v-if="item.isNow"
                class="ab-pill is-accent tt-now"
              >{{ copy.now }}</span></span>
              <span v-if="item.startsAt" class="tt-time" dir="ltr">{{ span(item.startsAt, item.endsAt) }}</span>
            </li>
          </ol>
        </Transition>
      </div>
    </div>

    <!-- A column wide enough for the whole week: the grid. -->
    <div class="tt-wide">
      <table class="tt-grid">
        <caption class="ab-sr-only">{{ meta ? `${copy.timetable} — ${meta}` : copy.timetable }}</caption>
        <thead>
          <tr>
            <th scope="col" class="tt-corner"><span class="ab-sr-only">{{ copy.day }}</span></th>
            <th
              v-for="column in view.columns"
              :key="column.number"
              scope="col"
              :class="[column.kind === 'break' ? 'tt-col-break' : 'tt-col', { 'is-now': column.isNow }]"
              :title="column.kind === 'break' ? breakTitle(column) : undefined"
            >
              <template v-if="column.kind === 'period'">
                <span class="tt-col-num">{{ column.number }}</span>
                <span v-if="column.startsAt" class="tt-col-time" dir="ltr">
                  <span>{{ column.startsAt }}</span>
                  <span v-if="column.endsAt">{{ column.endsAt }}</span>
                </span>
              </template>
              <template v-else>
                <i class="fa-solid fa-mug-hot" aria-hidden="true"></i>
                <span class="ab-sr-only">{{ breakTitle(column) }}</span>
              </template>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(day, dayIndex) in view.days"
            :key="day.key || dayIndex"
            :class="{ 'is-today': day.relative === 'today' }"
          >
            <th scope="row" class="tt-row-head">
              {{ day.label }}
              <span v-if="day.relative === 'today'" class="tt-row-dot" aria-hidden="true"></span>
              <span v-if="day.relative === 'today'" class="ab-sr-only">({{ copy.today }})</span>
            </th>
            <td v-for="(cell, index) in day.cells" :key="index" :class="cellClass(cell)">
              <span v-if="cell.kind === 'lesson'" class="tt-chip">{{ cell.isFree ? copy.free : cell.subject }}</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>

<script lang="ts">
/** Ids pairing each block's tabs with its panel, unique per block on the page. */
let nextId = 0;
</script>

<script setup lang="ts">
/**
 * A week of lessons, drawn for the width it is given.
 *
 * Phone first, because that is where parents read it. A row of days, today marked, opening
 * on today until its last bell and on the next school day after it; under it that day's
 * lessons as a list, breaks in place and the current lesson flagged. A sideways swipe moves
 * between days, and the tabs answer to the arrow keys the way a tab list should.
 *
 * A column wide enough for the whole week gets the grid instead: days down the side,
 * periods across the top with their bell times, the break as a narrow column. Which of the
 * two shows is decided by the block's OWN width — a container query in answerBlock.css —
 * rather than the screen's, because a chat column beside an open sidebar is narrow on a
 * wide monitor.
 *
 * Everything that decides anything is in utils/answerBlocks.ts, where node can test it.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';

import type { TimetableAnswerBlock } from '@/types/chat';
import {
  blockCopy,
  blockLanguage,
  buildTimetableView,
  swipeStep,
  tabLabel,
  type TimetableCell,
  type TimetableColumn,
  type TimetableItem,
} from '@/utils/answerBlocks';

const props = defineProps<{ block: TimetableAnswerBlock }>();

nextId += 1;
const uid = `tt-${nextId}`;

// What "now" and "today" are read against. Ticks each minute, so the current lesson moves
// on while the answer stays open on screen.
const now = ref(new Date());
let timer: number | undefined;
onMounted(() => {
  timer = window.setInterval(() => {
    now.value = new Date();
  }, 60_000);
});
onBeforeUnmount(() => window.clearInterval(timer));

const language = computed(() => blockLanguage(props.block));
const dir = computed(() => (language.value === 'ar' ? 'rtl' : 'ltr'));
const copy = computed(() => blockCopy(language.value));
const view = computed(() => buildTimetableView(props.block.data, now.value));
const meta = computed(() =>
  [props.block.data.class_label, props.block.data.term_label].filter(Boolean).join(' · '),
);

const selected = ref(view.value.initialIndex);
const direction = ref<'next' | 'prev'>('next');
// A different record in this place — the answer was replaced — opens on its own day.
watch(
  () => props.block,
  () => {
    selected.value = view.value.initialIndex;
  },
);
const current = computed(() => view.value.days[selected.value] || view.value.days[0]);

const tabs: HTMLElement[] = [];
const rememberTab = (el: unknown, index: number) => {
  if (el instanceof HTMLElement) tabs[index] = el;
};

function select(index: number, options: { focus?: boolean; reveal?: boolean } = {}) {
  if (index === selected.value || index < 0 || index >= view.value.days.length) return;
  direction.value = index > selected.value ? 'next' : 'prev';
  selected.value = index;
  const tab = tabs[index];
  if (!tab) return;
  if (options.focus) tab.focus({ preventScroll: true });
  // Only for a tab the reader just used, which is therefore on screen: `nearest` then
  // scrolls the row of days sideways and never moves the conversation.
  if (options.reveal) tab.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

function onTabKeydown(event: KeyboardEvent) {
  const forward = dir.value === 'rtl' ? 'ArrowLeft' : 'ArrowRight';
  const backward = dir.value === 'rtl' ? 'ArrowRight' : 'ArrowLeft';
  const last = view.value.days.length - 1;
  let target: number | null = null;
  if (event.key === forward) target = selected.value === last ? 0 : selected.value + 1;
  else if (event.key === backward) target = selected.value === 0 ? last : selected.value - 1;
  else if (event.key === 'Home') target = 0;
  else if (event.key === 'End') target = last;
  if (target === null) return;
  event.preventDefault();
  select(target, { focus: true, reveal: true });
}

let touchStart: { x: number; y: number } | null = null;
function onTouchStart(event: TouchEvent) {
  const touch = event.changedTouches[0];
  touchStart = touch ? { x: touch.clientX, y: touch.clientY } : null;
}
function onTouchEnd(event: TouchEvent) {
  const touch = event.changedTouches[0];
  const start = touchStart;
  touchStart = null;
  if (!touch || !start) return;
  // No wrapping on a swipe: past the last day, the finger has reached the end of the week.
  const step = swipeStep(touch.clientX - start.x, touch.clientY - start.y, dir.value);
  if (step) select(selected.value + step);
}

const relativeWord = (relative: 'today' | 'tomorrow') => copy.value[relative];
const span = (from: string, to: string) => (to ? `${from}–${to}` : from);
const breakTitle = (column: TimetableColumn) => {
  const name = column.label || copy.value.break;
  return column.startsAt ? `${name} ${span(column.startsAt, column.endsAt)}` : name;
};
const itemTitle = (item: TimetableItem) =>
  item.kind === 'lesson' ? (item.isFree ? copy.value.free : item.subject) : item.label || copy.value.break;
const cellClass = (cell: TimetableCell) => [
  'tt-cell',
  `is-${cell.kind}`,
  {
    'is-now': cell.kind !== 'empty' && cell.isNow,
    'is-free': cell.kind === 'lesson' && cell.isFree,
  },
];
</script>
