<template>
  <div class="cx-backdrop" @click.self="onClose">
    <section class="cx" role="dialog" aria-modal="true" aria-label="Chunk inspector">
      <header class="cx-head">
        <div class="cx-title">
          <span class="cx-eyebrow">Index inspector</span>
          <h2>{{ documentStore.inspecting }}</h2>
          <p v-if="list">
            {{ list.total.toLocaleString() }} chunks · {{ levels }}
            <template v-if="isFiltered">
              · <strong class="cx-hits">{{ list.match_count.toLocaleString() }} match</strong>
            </template>
          </p>
          <p v-else-if="documentStore.chunksLoading">Reading the index…</p>
        </div>
        <button type="button" class="cx-close" aria-label="Close" @click="onClose">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </header>

      <div class="cx-controls">
        <div class="cx-tabs" role="tablist">
          <button
            v-for="tab in tabs"
            :key="tab.key"
            type="button"
            role="tab"
            :aria-selected="view === tab.key"
            :class="['cx-tab', { active: view === tab.key }]"
            @click="view = tab.key"
          >
            <i :class="tab.icon"></i> {{ tab.label }}
          </button>
        </div>

        <label class="cx-search">
          <i class="fa-solid fa-magnifying-glass"></i>
          <input v-model="query" type="search" placeholder="Find text inside the chunks…" />
          <button v-if="query" type="button" class="cx-clear" aria-label="Clear" @click="query = ''">
            <i class="fa-solid fa-xmark"></i>
          </button>
        </label>

        <button
          type="button"
          :class="['cx-toggle', { on: showPins }]"
          :aria-pressed="showPins"
          @click="showPins = !showPins"
        >
          <i class="fa-solid fa-tags"></i> Metadata
        </button>
      </div>

      <div class="cx-legend">
        <span class="cx-key cx-key-l1">L1 document part</span>
        <span class="cx-key cx-key-l2">L2 section</span>
        <span class="cx-key cx-key-l3">L3 leaf · searchable</span>
        <span v-if="isFiltered" class="cx-key cx-key-hit">match</span>
        <span class="cx-legend-note">
          Matching ignores case, Arabic diacritics and the alef / teh-marbuta spellings —
          the folding retrieval itself uses.
        </span>
      </div>

      <div v-if="documentStore.chunksError" class="cx-empty cx-error">
        <i class="fa-solid fa-triangle-exclamation"></i>
        <strong>{{ documentStore.chunksError }}</strong>
      </div>

      <div v-else-if="documentStore.chunksLoading && !list" class="cx-empty">
        <i class="fa-solid fa-spinner fa-spin"></i>
        <strong>Reading the index…</strong>
      </div>

      <div v-else-if="list && !list.chunks.length" class="cx-empty">
        <i class="fa-regular fa-folder-open"></i>
        <strong>This document has no chunks indexed</strong>
      </div>

      <!-- DOCUMENT: the corpus as it reads, with each chunk's boundary drawn round it -->
      <div v-else-if="view === 'document'" class="cx-body">
        <ChunkDocumentNode
          v-for="node in tree"
          :key="node.chunk_id"
          :node="node"
          :pins="showPins"
        />
        <p v-if="list!.truncated" class="cx-note">
          Showing the first {{ list!.returned.toLocaleString() }} of
          {{ list!.total.toLocaleString() }} chunks.
        </p>
      </div>

      <!-- LIST: every chunk flat, narrowed to the matches while a filter is running -->
      <div v-else class="cx-body">
        <p v-if="isFiltered" class="cx-note">
          {{ visible.length.toLocaleString() }} of {{ list!.returned.toLocaleString() }} chunks
          contain “{{ list!.query.trim() }}”.
        </p>
        <div v-if="isFiltered && !visible.length" class="cx-empty">
          <i class="fa-regular fa-face-frown"></i>
          <strong>Nothing matches “{{ list!.query.trim() }}”</strong>
        </div>
        <ChunkCard
          v-for="chunk in visible"
          :key="chunk.chunk_id"
          :chunk="chunk"
          :pins="showPins"
        />
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { useDocumentStore } from '@/stores/documents';
import type { ChunkNode } from '@/types/document';
import { buildChunkTree, levelSummary } from '@/utils/chunkTree';
import ChunkDocumentNode from './ChunkDocumentNode.vue';
import ChunkCard from './ChunkCard.vue';

const documentStore = useDocumentStore();

const tabs = [
  { key: 'document' as const, label: 'Document', icon: 'fa-solid fa-file-lines' },
  { key: 'list' as const, label: 'All chunks', icon: 'fa-solid fa-list' },
];
const view = ref<'document' | 'list'>('document');
const showPins = ref(true);

const list = computed(() => documentStore.chunkList);
const isFiltered = computed(() => Boolean(list.value?.query.trim()));
const tree = computed<ChunkNode[]>(() => buildChunkTree(list.value?.chunks || []));
const levels = computed(() => levelSummary(list.value?.chunks || []));

/**
 * The list narrows to the hits; the document never does.
 *
 * The server MARKS rather than removes, so both views read the same response: the
 * document keeps its structure to draw, and the flat list filters on the flag the server
 * already decided. Neither view folds anything itself.
 */
const visible = computed(() => {
  const chunks = list.value?.chunks || [];
  return isFiltered.value ? chunks.filter((chunk) => chunk.matched) : chunks;
});

/** Typing is a request per keystroke without this. */
const DEBOUNCE_MS = 250;
let timer: ReturnType<typeof setTimeout> | undefined;

const query = computed({
  get: () => documentStore.chunkQuery,
  set: (value: string) => {
    documentStore.chunkQuery = value;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => documentStore.loadChunks(), DEBOUNCE_MS);
  },
});

function onClose() {
  documentStore.closeInspector();
}

function onEscape(event: KeyboardEvent) {
  if (event.key === 'Escape') onClose();
}

watch(
  () => documentStore.inspecting,
  (open) => {
    if (open) window.addEventListener('keydown', onEscape);
    else window.removeEventListener('keydown', onEscape);
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  if (timer) clearTimeout(timer);
  window.removeEventListener('keydown', onEscape);
});
</script>

<style scoped>
/*
 * Self-contained colour, dark first with an explicit light override.
 *
 * This panel previously painted light backgrounds while inheriting the app's text
 * colour, because it referenced `--surface`/`--text`/`--border` — tokens this app does
 * not define. Every fallback fired and the result was white text on white boxes. Nothing
 * below reads a variable it did not declare, and every background states its own colour.
 */
.cx-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(3, 12, 22, 0.72);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 18px;
  z-index: 200;
}

.cx {
  display: flex;
  flex-direction: column;
  width: min(1100px, 100%);
  max-height: min(90vh, 940px);
  background: #08243c;
  color: #eaf4fb;
  border: 1px solid rgba(72, 160, 200, 0.3);
  border-radius: 16px;
  box-shadow: 0 28px 70px rgba(0, 0, 0, 0.55);
  overflow: hidden;
}

.cx-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 18px 20px 12px;
  border-bottom: 1px solid rgba(72, 160, 200, 0.22);
}

.cx-eyebrow {
  font-size: 0.66rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #36c6e8;
  font-weight: 700;
}

.cx-title h2 {
  margin: 3px 0 4px;
  font-size: 1.04rem;
  color: #f1f8fd;
  word-break: break-all;
}

.cx-title p {
  margin: 0;
  font-size: 0.79rem;
  color: #8fb4d0;
}

.cx-hits {
  color: #f5b544;
}

.cx-close {
  border: 1px solid rgba(72, 160, 200, 0.28);
  background: rgba(8, 30, 50, 0.7);
  color: #cfe6f5;
  font-size: 1rem;
  cursor: pointer;
  padding: 6px 11px;
  border-radius: 9px;
  flex-shrink: 0;
}

.cx-close:hover {
  background: rgba(14, 54, 83, 0.9);
  color: #ffffff;
}

.cx-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  padding: 12px 20px 0;
}

.cx-tabs {
  display: inline-flex;
  gap: 3px;
  background: rgba(6, 23, 40, 0.75);
  border: 1px solid rgba(72, 160, 200, 0.2);
  padding: 3px;
  border-radius: 10px;
}

.cx-tab {
  border: none;
  background: transparent;
  padding: 7px 14px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.82rem;
  color: #a9c8de;
}

.cx-tab:hover {
  color: #eaf4fb;
}

.cx-tab.active {
  background: #145a7d;
  color: #ffffff;
  font-weight: 650;
}

.cx-search {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1 1 240px;
  border: 1px solid rgba(72, 160, 200, 0.28);
  background: rgba(6, 23, 40, 0.75);
  border-radius: 10px;
  padding: 7px 12px;
  color: #8fb4d0;
}

.cx-search input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  font: inherit;
  color: #eaf4fb;
  min-width: 0;
}

.cx-search input::placeholder {
  color: #6f93b1;
}

.cx-clear {
  border: none;
  background: transparent;
  cursor: pointer;
  color: #8fb4d0;
}

.cx-toggle {
  border: 1px solid rgba(72, 160, 200, 0.28);
  background: rgba(6, 23, 40, 0.75);
  color: #a9c8de;
  border-radius: 10px;
  padding: 7px 12px;
  cursor: pointer;
  font-size: 0.8rem;
}

.cx-toggle.on {
  border-color: #36c6e8;
  color: #36c6e8;
}

.cx-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  padding: 10px 20px 0;
  font-size: 0.7rem;
}

.cx-key {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 9px;
  border-radius: 999px;
  border: 1px solid currentColor;
}

.cx-key::before {
  content: '';
  width: 8px;
  height: 8px;
  border-radius: 2px;
  background: currentColor;
}

.cx-key-l1 { color: #8b7cf6; }
.cx-key-l2 { color: #22c8ee; }
.cx-key-l3 { color: #43d39e; }
.cx-key-hit { color: #f5b544; }

.cx-legend-note {
  color: #7fa3c2;
  flex: 1 1 260px;
  min-width: 0;
}

.cx-body {
  flex: 1;
  overflow-y: auto;
  padding: 14px 20px 20px;
  display: flex;
  flex-direction: column;
  gap: 9px;
}

.cx-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 44px 20px;
  color: #8fb4d0;
  text-align: center;
}

.cx-empty i {
  font-size: 1.5rem;
}

.cx-error {
  color: #ff9d9d;
}

.cx-note {
  margin: 0;
  font-size: 0.76rem;
  color: #8fb4d0;
}

/* ---- light theme ------------------------------------------------------------ */
html[data-theme='light'] .cx {
  background: #ffffff;
  color: #16293c;
  border-color: rgba(70, 110, 145, 0.22);
}

html[data-theme='light'] .cx-head { border-bottom-color: rgba(70, 110, 145, 0.18); }
html[data-theme='light'] .cx-eyebrow { color: #0f7f9e; }
html[data-theme='light'] .cx-title h2 { color: #10243a; }
html[data-theme='light'] .cx-title p,
html[data-theme='light'] .cx-note,
html[data-theme='light'] .cx-empty,
html[data-theme='light'] .cx-legend-note { color: #5b7893; }
html[data-theme='light'] .cx-hits { color: #b9791a; }

html[data-theme='light'] .cx-close,
html[data-theme='light'] .cx-tabs,
html[data-theme='light'] .cx-search,
html[data-theme='light'] .cx-toggle {
  background: #f3f6fa;
  border-color: rgba(70, 110, 145, 0.22);
  color: #33536f;
}

html[data-theme='light'] .cx-search input { color: #16293c; }
html[data-theme='light'] .cx-search input::placeholder { color: #86a2bb; }
html[data-theme='light'] .cx-tab { color: #4a6b88; }
html[data-theme='light'] .cx-tab.active { background: #1496b9; color: #ffffff; }
html[data-theme='light'] .cx-toggle.on { border-color: #0f7f9e; color: #0f7f9e; }
html[data-theme='light'] .cx-clear { color: #5b7893; }
html[data-theme='light'] .cx-key-l2 { color: #1496b9; }
html[data-theme='light'] .cx-key-l3 { color: #1ea073; }
html[data-theme='light'] .cx-key-hit { color: #b9791a; }
html[data-theme='light'] .cx-error { color: #b3261e; }

@media (max-width: 640px) {
  .cx-backdrop { padding: 0; }

  .cx {
    max-height: 100vh;
    height: 100vh;
    border-radius: 0;
    border: none;
  }

  .cx-head,
  .cx-body { padding-left: 13px; padding-right: 13px; }
  .cx-controls,
  .cx-legend { padding-left: 13px; padding-right: 13px; }
}
</style>
