<template>
  <div :class="['cdoc-node', 'cdoc-l' + node.chunk_level, { 'is-match': node.matched }]">
    <header class="cdoc-tag">
      <span class="cdoc-level">L{{ node.chunk_level }}</span>
      <span class="cdoc-role">{{ roleLabel }}</span>
      <span v-if="node.modality !== 'text'" class="cdoc-modality">
        <i :class="node.modality === 'figure' ? 'fa-regular fa-image' : 'fa-solid fa-table'"></i>
        {{ node.modality }}
      </span>
      <span v-if="node.matched" class="cdoc-hit"><i class="fa-solid fa-magnifying-glass"></i> match</span>
      <span class="cdoc-tag-spacer"></span>
      <span class="cdoc-chars">{{ node.char_count.toLocaleString() }} chars</span>
      <button
        type="button"
        class="cdoc-fold"
        :aria-expanded="open"
        :title="open ? 'Collapse' : 'Expand'"
        @click="open = !open"
      >
        <i :class="open ? 'fa-solid fa-chevron-up' : 'fa-solid fa-chevron-down'"></i>
      </button>
    </header>

    <template v-if="open">
      <!--
        A parent's text is its children's text concatenated, so rendering both would print
        the document three times over. A box shows its OWN text only when it has no
        children; otherwise it is a boundary drawn around the boxes inside it, and the
        leaves are what carry the words. That is what makes this read as the document.
      -->
      <div v-if="node.children.length" class="cdoc-children">
        <ChunkDocumentNode
          v-for="child in node.children"
          :key="child.chunk_id"
          :node="child"
          :pins="pins"
        />
      </div>
      <p v-else class="cdoc-text" dir="auto">{{ node.text || '(empty chunk)' }}</p>

      <dl v-if="pins" class="cdoc-pins">
        <div v-for="pin in metadata" :key="pin.label" class="cdoc-pin" :title="pin.title || pin.label">
          <dt>{{ pin.label }}</dt>
          <dd>{{ pin.value }}</dd>
        </div>
      </dl>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ChunkNode } from '@/types/document';

const props = withDefaults(
  defineProps<{
    node: ChunkNode;
    /** Whether the developer metadata strip is shown under each box. */
    pins?: boolean;
  }>(),
  { pins: true },
);

const open = ref(true);

/**
 * What the level IS, not just its number.
 *
 * Only level 3 is vectorised. A search matches a leaf, and the levels above exist so a
 * hit can be merged up into enough surrounding text to answer from — which is the whole
 * reason the boundaries are worth drawing.
 */
const roleLabel = computed(() => {
  if (props.node.chunk_level === 3) return 'leaf · searchable';
  if (props.node.chunk_level === 2) return 'section · merge target';
  if (props.node.chunk_level === 1) return 'document part';
  return 'chunk';
});

const metadata = computed(() => {
  const chunk = props.node;
  const out: { label: string; value: string; title?: string }[] = [
    { label: 'id', value: chunk.chunk_id, title: 'chunk_id' },
    { label: 'idx', value: String(chunk.chunk_idx), title: 'chunk_idx' },
    { label: 'page', value: String(chunk.page_number), title: 'page_number' },
    { label: 'modality', value: chunk.modality },
  ];
  if (chunk.parent_chunk_id) out.push({ label: 'parent', value: chunk.parent_chunk_id, title: 'parent_chunk_id' });
  if (chunk.root_chunk_id) out.push({ label: 'root', value: chunk.root_chunk_id, title: 'root_chunk_id' });
  if (chunk.asset_ids.length) {
    out.push({ label: 'assets', value: chunk.asset_ids.join(', '), title: 'The image this chunk was transcribed from' });
  }
  return out;
});
</script>

<style scoped>
/*
 * Every colour is declared here in full, dark first with an explicit light override.
 *
 * The first cut of this panel inherited `color` from the app while painting its own
 * light backgrounds, on the assumption that `--surface` and `--text` existed. They do
 * not — this app names its tokens `--ax-*` — so every fallback fired, and near-white
 * text landed on near-white boxes. Nothing here relies on a variable it did not define,
 * and no background is ever set without the colour that has to sit on it.
 */
.cdoc-node {
  --cdoc-accent: #8b7cf6;
  --cdoc-wash: rgba(139, 124, 246, 0.07);
  --cdoc-ink: #eaf4fb;
  --cdoc-dim: #9fbcd4;

  border: 1px solid var(--cdoc-accent);
  border-inline-start-width: 3px;
  border-radius: 10px;
  background: var(--cdoc-wash);
  color: var(--cdoc-ink);
  padding: 8px 10px 10px;
  margin: 0 0 8px;
}

.cdoc-l1 {
  --cdoc-accent: #8b7cf6;
  --cdoc-wash: rgba(139, 124, 246, 0.08);
}

.cdoc-l2 {
  --cdoc-accent: #22c8ee;
  --cdoc-wash: rgba(34, 200, 238, 0.07);
}

.cdoc-l3 {
  --cdoc-accent: #43d39e;
  --cdoc-wash: rgba(67, 211, 158, 0.06);
}

.cdoc-node.is-match {
  --cdoc-accent: #f5b544;
  --cdoc-wash: rgba(245, 181, 68, 0.13);
  box-shadow: 0 0 0 1px rgba(245, 181, 68, 0.45);
}

.cdoc-tag {
  display: flex;
  align-items: center;
  gap: 7px;
  flex-wrap: wrap;
  font-size: 0.68rem;
  margin-bottom: 6px;
}

.cdoc-tag-spacer {
  flex: 1;
}

.cdoc-level {
  font-weight: 800;
  letter-spacing: 0.03em;
  padding: 1px 7px;
  border-radius: 999px;
  background: var(--cdoc-accent);
  color: #08192a;
}

.cdoc-role,
.cdoc-chars {
  color: var(--cdoc-dim);
}

.cdoc-modality {
  padding: 1px 7px;
  border-radius: 999px;
  border: 1px solid var(--cdoc-accent);
  color: var(--cdoc-accent);
}

.cdoc-hit {
  padding: 1px 7px;
  border-radius: 999px;
  background: #f5b544;
  color: #33240a;
  font-weight: 700;
}

.cdoc-fold {
  border: none;
  background: transparent;
  cursor: pointer;
  color: var(--cdoc-dim);
  padding: 0 2px;
}

.cdoc-fold:hover {
  color: var(--cdoc-ink);
}

.cdoc-children {
  display: flex;
  flex-direction: column;
}

.cdoc-children > .cdoc-node:last-child {
  margin-bottom: 0;
}

/* The document itself: the one place the corpus's own words are shown, so it gets the
   readable colour and the preserved line breaks. */
.cdoc-text {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.87rem;
  line-height: 1.6;
  color: var(--cdoc-ink);
}

.cdoc-pins {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin: 8px 0 0;
  padding-top: 7px;
  border-top: 1px dashed rgba(120, 165, 200, 0.25);
}

.cdoc-pin {
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  border: 1px solid rgba(120, 165, 200, 0.28);
  border-radius: 999px;
  padding: 1px 8px;
  background: rgba(6, 23, 40, 0.5);
  max-width: 100%;
}

.cdoc-pin dt {
  margin: 0;
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #7fa3c2;
}

.cdoc-pin dd {
  margin: 0;
  font-size: 0.7rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #d7e8f5;
  overflow-wrap: anywhere;
}

html[data-theme='light'] .cdoc-node {
  --cdoc-ink: #16293c;
  --cdoc-dim: #5b7893;
}

html[data-theme='light'] .cdoc-l1 { --cdoc-wash: rgba(139, 124, 246, 0.10); }
html[data-theme='light'] .cdoc-l2 { --cdoc-wash: rgba(20, 150, 185, 0.09); }
html[data-theme='light'] .cdoc-l3 { --cdoc-wash: rgba(30, 160, 115, 0.09); }
html[data-theme='light'] .cdoc-l2 { --cdoc-accent: #1496b9; }
html[data-theme='light'] .cdoc-l3 { --cdoc-accent: #1ea073; }

html[data-theme='light'] .cdoc-node.is-match {
  --cdoc-accent: #b9791a;
  --cdoc-wash: rgba(245, 181, 68, 0.20);
}

html[data-theme='light'] .cdoc-level { color: #ffffff; }
html[data-theme='light'] .cdoc-hit { background: #b9791a; color: #ffffff; }

html[data-theme='light'] .cdoc-pin {
  background: rgba(255, 255, 255, 0.85);
  border-color: rgba(70, 110, 145, 0.25);
}

html[data-theme='light'] .cdoc-pin dt { color: #5b7893; }
html[data-theme='light'] .cdoc-pin dd { color: #16293c; }

html[data-theme='light'] .cdoc-pins { border-top-color: rgba(70, 110, 145, 0.22); }

@media (max-width: 640px) {
  .cdoc-node {
    padding: 7px 8px 8px;
  }
}
</style>
