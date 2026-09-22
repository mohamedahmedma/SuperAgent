<template>
  <article :class="['ck', 'ck-l' + chunk.chunk_level, { 'is-match': chunk.matched }]">
    <header class="ck-head">
      <span class="ck-level">L{{ chunk.chunk_level }}</span>
      <span class="ck-role">{{ roleLabel }}</span>
      <span v-if="chunk.modality !== 'text'" class="ck-modality">
        <i :class="chunk.modality === 'figure' ? 'fa-regular fa-image' : 'fa-solid fa-table'"></i>
        {{ chunk.modality }}
      </span>
      <span class="ck-spacer"></span>
      <span class="ck-chars">{{ chunk.char_count.toLocaleString() }} chars</span>
      <button
        type="button"
        class="ck-copy"
        :title="copied ? 'Copied' : 'Copy chunk id'"
        @click="onCopyId"
      >
        <i :class="copied ? 'fa-solid fa-check' : 'fa-regular fa-copy'"></i>
      </button>
    </header>

    <!-- `dir=auto` per chunk, not per document: a bilingual corpus holds both. -->
    <p class="ck-text" dir="auto">{{ chunk.text || '(empty chunk)' }}</p>

    <dl v-if="pins" class="ck-pins">
      <div v-for="pin in metadata" :key="pin.label" class="ck-pin" :title="pin.title || pin.label">
        <dt>{{ pin.label }}</dt>
        <dd>{{ pin.value }}</dd>
      </div>
    </dl>
  </article>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ChunkInfo } from '@/types/document';

const props = withDefaults(
  defineProps<{
    chunk: ChunkInfo;
    pins?: boolean;
  }>(),
  { pins: true },
);

const copied = ref(false);

const roleLabel = computed(() => {
  if (props.chunk.chunk_level === 3) return 'leaf · searchable';
  if (props.chunk.chunk_level === 2) return 'section · merge target';
  if (props.chunk.chunk_level === 1) return 'document part';
  return 'chunk';
});

const metadata = computed(() => {
  const chunk = props.chunk;
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

async function onCopyId() {
  try {
    await navigator.clipboard.writeText(props.chunk.chunk_id);
    copied.value = true;
    setTimeout(() => (copied.value = false), 1200);
  } catch {
    // A clipboard the browser refuses is not worth an error state; the id is on screen.
  }
}
</script>

<style scoped>
/* Colours declared in full here, for the reason written in ChunkInspector.vue: this app
   names its tokens `--ax-*`, and the first cut of this card referenced `--surface` and
   `--text`, which do not exist. Every background below states the colour that sits on it. */
.ck {
  --ck-accent: #43d39e;
  --ck-ink: #eaf4fb;
  --ck-dim: #9fbcd4;

  border: 1px solid rgba(72, 160, 200, 0.26);
  border-inline-start: 3px solid var(--ck-accent);
  border-radius: 11px;
  padding: 11px 13px;
  background: rgba(9, 32, 52, 0.75);
  color: var(--ck-ink);
}

.ck-l1 { --ck-accent: #8b7cf6; }
.ck-l2 { --ck-accent: #22c8ee; }
.ck-l3 { --ck-accent: #43d39e; }

.ck.is-match {
  --ck-accent: #f5b544;
  box-shadow: 0 0 0 1px rgba(245, 181, 68, 0.45);
}

.ck-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
  font-size: 0.7rem;
}

.ck-spacer { flex: 1; }

.ck-level {
  font-weight: 800;
  padding: 1px 8px;
  border-radius: 999px;
  background: var(--ck-accent);
  color: #08192a;
}

.ck-role,
.ck-chars { color: var(--ck-dim); }

.ck-modality {
  padding: 1px 8px;
  border-radius: 999px;
  border: 1px solid var(--ck-accent);
  color: var(--ck-accent);
}

.ck-copy {
  border: none;
  background: transparent;
  cursor: pointer;
  color: var(--ck-dim);
}

.ck-copy:hover { color: var(--ck-ink); }

.ck-text {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.87rem;
  line-height: 1.6;
  color: var(--ck-ink);
}

.ck-pins {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin: 9px 0 0;
  padding-top: 8px;
  border-top: 1px dashed rgba(120, 165, 200, 0.25);
}

.ck-pin {
  display: inline-flex;
  align-items: baseline;
  gap: 5px;
  border: 1px solid rgba(120, 165, 200, 0.28);
  border-radius: 999px;
  padding: 1px 8px;
  background: rgba(6, 23, 40, 0.6);
  max-width: 100%;
}

.ck-pin dt {
  margin: 0;
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #7fa3c2;
}

.ck-pin dd {
  margin: 0;
  font-size: 0.7rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #d7e8f5;
  overflow-wrap: anywhere;
}

html[data-theme='light'] .ck {
  --ck-ink: #16293c;
  --ck-dim: #5b7893;
  background: #ffffff;
  border-color: rgba(70, 110, 145, 0.22);
}

html[data-theme='light'] .ck-l2 { --ck-accent: #1496b9; }
html[data-theme='light'] .ck-l3 { --ck-accent: #1ea073; }
html[data-theme='light'] .ck.is-match { --ck-accent: #b9791a; }
html[data-theme='light'] .ck-level { color: #ffffff; }

html[data-theme='light'] .ck-pin {
  background: #f3f6fa;
  border-color: rgba(70, 110, 145, 0.25);
}

html[data-theme='light'] .ck-pin dt { color: #5b7893; }
html[data-theme='light'] .ck-pin dd { color: #16293c; }
html[data-theme='light'] .ck-pins { border-top-color: rgba(70, 110, 145, 0.22); }
</style>
