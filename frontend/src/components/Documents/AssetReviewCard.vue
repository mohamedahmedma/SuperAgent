<template>
  <article :class="['ar', { 'is-review': asset.needs_review, 'is-failed': asset.status === 'failed' }]">
    <div class="ar-shot">
      <img v-if="source" :src="source" :alt="asset.caption || 'figure'" @click="openFull" />
      <div v-else-if="failed" class="ar-shot-missing">
        <i class="fa-regular fa-image"></i>
        <span>image unavailable</span>
      </div>
      <div v-else class="ar-shot-missing"><i class="fa-solid fa-spinner fa-spin"></i></div>
    </div>

    <div class="ar-body">
      <header class="ar-head">
        <span class="ar-page">p{{ asset.page_number }}</span>
        <span :class="['ar-status', 'ar-status-' + asset.status]">{{ asset.status }}</span>
        <span v-if="asset.needs_review" class="ar-flag" title="The extractor flagged this: low confidence, or no text recovered">
          <i class="fa-solid fa-triangle-exclamation"></i> needs review
        </span>
        <span v-if="!asset.indexable" class="ar-flag ar-flag-quiet" title="Stored, but produces no retrievable chunk">
          not indexed
        </span>
        <span class="ar-spacer"></span>
        <span v-if="asset.model_used" class="ar-model" :title="'Confidence ' + asset.confidence.toFixed(2)">
          {{ asset.model_used }} · {{ asset.confidence.toFixed(2) }}
        </span>
      </header>

      <p v-if="asset.error" class="ar-error">{{ asset.error }}</p>

      <!-- The retrieval surface, in the order `render_surrogate` writes it: caption
           identifies, description explains, transcription carries the rare tokens. -->
      <dl class="ar-fields">
        <div v-for="field in fields" :key="field.label" class="ar-field">
          <dt>{{ field.label }}</dt>
          <dd dir="auto">{{ field.value }}</dd>
        </div>
      </dl>

      <div v-if="asset.tags.length" class="ar-tags">
        <span v-for="tag in asset.tags" :key="tag" class="ar-tag">{{ tag }}</span>
      </div>

      <footer class="ar-foot">
        <button type="button" class="ar-chunks" :disabled="!asset.chunk_ids.length" @click="emit('show-chunks', asset)">
          <i class="fa-solid fa-diagram-project"></i>
          {{ asset.chunk_ids.length }} chunk{{ asset.chunk_ids.length === 1 ? '' : 's' }}
        </button>
        <span class="ar-dims">{{ asset.width }}×{{ asset.height }} · {{ kb }} KB</span>
      </footer>
    </div>
  </article>
</template>

<script setup lang="ts">
/**
 * One image, with everything extraction made of it, for manual review.
 *
 * The picture is beside the text on purpose: a transcription is only judgeable against
 * the thing it claims to transcribe, and until this view existed the two had never been
 * on screen together.
 *
 * Fetching the bytes follows `MessageAssets.vue`, including the three traps it already
 * documents: the endpoint is Bearer-authenticated so a plain `<img src>` would 401;
 * `authStore.authorizedFetch` sends a token that is still valid when the request lands,
 * which reading `authStore.token` once does not; and `apiUrl` sends the path to the API
 * rather than resolving it against the page's own origin.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';

import { useAuthStore } from '../../stores/auth';
import type { AssetInfo } from '../../types/document';
import { apiUrl } from '../../utils/api';

const props = defineProps<{ asset: AssetInfo }>();
const emit = defineEmits<{ (event: 'show-chunks', asset: AssetInfo): void }>();

const authStore = useAuthStore();
const source = ref('');
const failed = ref(false);
let objectUrl = '';

const kb = computed(() => Math.max(1, Math.round(props.asset.byte_size / 1024)));

const fields = computed(() =>
  [
    { label: 'caption', value: props.asset.caption },
    { label: 'description', value: props.asset.description },
    { label: 'transcription', value: props.asset.transcription },
  ].filter((field) => field.value),
);

const openFull = () => {
  if (source.value) window.open(source.value, '_blank', 'noopener');
};

onMounted(async () => {
  if (!props.asset.url) {
    failed.value = true;
    return;
  }
  try {
    const response = await authStore.authorizedFetch(apiUrl(props.asset.url));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    objectUrl = URL.createObjectURL(await response.blob());
    source.value = objectUrl;
  } catch {
    failed.value = true;
  }
});

// Object URLs hold the blob in memory until revoked.
onBeforeUnmount(() => {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
});
</script>

<style scoped>
/* Colours are declared in full, for the reason ChunkInspector.vue records: this app
   names its tokens `--ax-*`, and styling against `--surface`/`--text` painted white on
   white. Every background below states the colour that sits on it. */
.ar {
  --ar-accent: #43d39e;
  --ar-ink: #eaf4fb;
  --ar-dim: #9fbcd4;

  display: grid;
  grid-template-columns: 132px minmax(0, 1fr);
  gap: 13px;
  border: 1px solid rgba(72, 160, 200, 0.26);
  border-inline-start: 3px solid var(--ar-accent);
  border-radius: 11px;
  padding: 11px 13px;
  background: rgba(9, 32, 52, 0.75);
  color: var(--ar-ink);
}

.ar.is-review { --ar-accent: #f5b544; }
.ar.is-failed { --ar-accent: #f2708f; }

.ar-shot {
  display: flex;
  align-items: flex-start;
  justify-content: center;
}

.ar-shot img {
  max-width: 100%;
  max-height: 150px;
  border-radius: 7px;
  background: rgba(3, 16, 28, 0.6);
  cursor: zoom-in;
  object-fit: contain;
}

.ar-shot-missing {
  display: grid;
  gap: 5px;
  justify-items: center;
  width: 100%;
  padding: 22px 6px;
  border: 1px dashed rgba(120, 170, 205, 0.34);
  border-radius: 7px;
  color: var(--ar-dim);
  font-size: 10px;
  text-align: center;
}

.ar-body {
  display: grid;
  gap: 8px;
  min-width: 0;
}

.ar-head {
  display: flex;
  align-items: center;
  gap: 7px;
  flex-wrap: wrap;
  font-size: 10px;
}

.ar-spacer { flex: 1 1 auto; }

.ar-page {
  padding: 1px 6px;
  border-radius: 999px;
  background: rgba(67, 211, 158, 0.16);
  color: #8ce6c4;
  font-weight: 600;
}

.ar-status {
  color: var(--ar-dim);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.ar-status-failed { color: #f2708f; }

.ar-flag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 1px 7px;
  border-radius: 999px;
  background: rgba(245, 181, 68, 0.17);
  color: #f7c76a;
  font-weight: 600;
}

.ar-flag-quiet {
  background: rgba(120, 170, 205, 0.14);
  color: var(--ar-dim);
  font-weight: 500;
}

.ar-model {
  color: var(--ar-dim);
  font-variant-numeric: tabular-nums;
}

.ar-error {
  margin: 0;
  padding: 6px 8px;
  border-radius: 7px;
  background: rgba(242, 112, 143, 0.14);
  color: #ffb3c4;
  font-size: 10px;
  line-height: 1.45;
}

.ar-fields {
  display: grid;
  gap: 6px;
  margin: 0;
}

.ar-field {
  display: grid;
  grid-template-columns: 82px minmax(0, 1fr);
  gap: 8px;
  align-items: start;
}

.ar-field dt {
  color: var(--ar-dim);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding-top: 1px;
}

.ar-field dd {
  margin: 0;
  color: var(--ar-ink);
  font-size: 11px;
  line-height: 1.5;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.ar-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.ar-tag {
  padding: 1px 7px;
  border-radius: 999px;
  background: rgba(120, 170, 205, 0.15);
  color: var(--ar-dim);
  font-size: 9px;
}

.ar-foot {
  display: flex;
  align-items: center;
  gap: 9px;
  flex-wrap: wrap;
  font-size: 10px;
}

.ar-chunks {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 9px;
  border: 1px solid rgba(72, 160, 200, 0.3);
  border-radius: 999px;
  background: rgba(34, 200, 238, 0.12);
  color: #8fe3f5;
  font: inherit;
  cursor: pointer;
}

.ar-chunks:disabled {
  background: rgba(120, 170, 205, 0.1);
  color: var(--ar-dim);
  cursor: default;
}

.ar-dims {
  margin-inline-start: auto;
  color: var(--ar-dim);
  font-variant-numeric: tabular-nums;
}

html[data-theme='light'] .ar {
  --ar-ink: #0d2136;
  --ar-dim: #4d6a83;

  border-color: rgba(23, 80, 120, 0.2);
  background: #ffffff;
}

html[data-theme='light'] .ar-shot img,
html[data-theme='light'] .ar-shot-missing {
  background: #f2f7fb;
}

html[data-theme='light'] .ar-page { color: #157a57; }
html[data-theme='light'] .ar-flag { color: #8a5a06; }
html[data-theme='light'] .ar-error { color: #98243f; }
html[data-theme='light'] .ar-chunks { color: #0d6b83; }

@media (max-width: 640px) {
  .ar { grid-template-columns: minmax(0, 1fr); }
  .ar-shot img { max-height: 190px; }
  .ar-field { grid-template-columns: minmax(0, 1fr); gap: 2px; }
}
</style>
