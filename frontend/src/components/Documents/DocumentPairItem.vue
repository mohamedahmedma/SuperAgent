<template>
  <article class="pair-item">
    <div class="pair-item-head">
      <span class="pair-item-title">
        <i class="fa-regular fa-file-lines"></i>
        {{ pair.title || 'Untitled entry' }}
      </span>
      <span :class="['pair-badge', pair.paired ? 'is-paired' : 'is-single']">
        <i :class="pair.paired ? 'fa-solid fa-language' : 'fa-solid fa-circle-half-stroke'"></i>
        {{ pair.paired ? 'Both languages' : 'One language' }}
      </span>
    </div>

    <div class="pair-item-slots">
      <div
        v-for="side in sides"
        :key="side.lang"
        :class="['pair-item-slot', { empty: !side.filename }]"
        :dir="side.dir"
      >
        <span class="pair-item-lang">{{ side.label }}</span>

        <template v-if="side.filename">
          <strong class="pair-item-file">{{ side.filename }}</strong>
          <small>{{ side.chunks.toLocaleString() }} chunks</small>
          <button
            type="button"
            class="btn-danger pair-item-delete"
            title="Delete this version"
            :disabled="documentStore.isDeleteActionLocked(side.filename)"
            @click="onDelete(side.filename)"
          >
            <i :class="documentStore.getDeleteButtonIcon(side.filename)"></i>
          </button>
        </template>

        <template v-else>
          <small class="pair-item-missing">Not uploaded</small>
          <button type="button" class="pair-item-add" @click="onFill">
            <i class="fa-solid fa-plus"></i> Add {{ side.short }}
          </button>
        </template>
      </div>
    </div>

    <p v-if="pair.unassigned" class="pair-item-note">
      Indexed outside the bilingual form. It still answers questions — re-upload it through a row
      to pair it with a translation.
    </p>

    <!-- Deletes run through four storage layers and can fail partway, so the per-step
         progress stays visible per FILE rather than per entry: only one side of a pair
         is usually being removed, and saying which matters. -->
    <div
      v-for="job in activeDeleteJobs"
      :key="job.filename"
      :class="['upload-progress', 'delete-progress', { collapsed: job.collapsed }]"
    >
      <button type="button" class="upload-progress-header" @click="onToggleCollapse(job.filename)">
        <span>
          <strong>{{ job.message || 'Delete progress' }}</strong>
          <small>{{ job.filename }}</small>
        </span>
        <span class="upload-toggle">
          {{ job.collapsed ? 'Expand' : 'Collapse' }}
          <i :class="job.collapsed ? 'fa-solid fa-chevron-down' : 'fa-solid fa-chevron-up'"></i>
        </span>
      </button>

      <div v-show="!job.collapsed" class="upload-step-list">
        <div
          v-for="step in job.steps"
          :key="step.key"
          :class="['upload-step', 'upload-step-' + step.status]"
        >
          <div class="upload-step-header">
            <span class="upload-step-label">{{ step.label }}</span>
            <span class="upload-step-percent">{{ step.percent }}%</span>
          </div>
          <div class="upload-step-bar">
            <div class="upload-step-fill" :style="{ width: step.percent + '%' }"></div>
          </div>
          <div v-if="step.message" class="upload-step-message">{{ step.message }}</div>
        </div>
      </div>
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useDocumentStore } from '@/stores/documents';
import type { DocumentPair } from '@/types/document';

const props = defineProps<{ pair: DocumentPair }>();

const documentStore = useDocumentStore();

const sides = computed(() => [
  {
    lang: 'ar',
    label: 'العربية · Arabic',
    short: 'Arabic',
    dir: 'rtl',
    filename: props.pair.filename_ar,
    chunks: props.pair.chunk_count_ar,
  },
  {
    lang: 'en',
    label: 'English',
    short: 'English',
    dir: 'ltr',
    filename: props.pair.filename_en,
    chunks: props.pair.chunk_count_en,
  },
]);

const activeDeleteJobs = computed(() =>
  [props.pair.filename_ar, props.pair.filename_en]
    .filter(Boolean)
    .map((filename) => ({ filename, ...documentStore.deleteJobs[filename] }))
    .filter((job) => job.status)
);

const onFill = () => {
  documentStore.startFillingPair(props.pair);
};

const onToggleCollapse = (filename: string) => {
  documentStore.toggleDeleteJobCollapsed(filename);
};

const onDelete = async (filename: string) => {
  try {
    await documentStore.deleteDocument(filename);
  } catch (error: any) {
    alert(error.message);
  }
};
</script>

<style scoped>
.pair-item {
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  background: var(--surface-soft);
}

.pair-item-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
  gap: 10px;
}

.pair-item-title {
  display: flex;
  overflow: hidden;
  align-items: center;
  color: var(--text);
  font-size: var(--font-body);
  gap: 8px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pair-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 9px;
  border-radius: 999px;
  flex: none;
  font-size: var(--font-micro);
  gap: 6px;
}

.pair-badge.is-paired {
  color: var(--mint-ink);
  background: var(--mint);
}

.pair-badge.is-single {
  color: var(--muted-strong);
  background: var(--surface-strong);
}

.pair-item-slots {
  display: grid;
  gap: 10px;
  grid-template-columns: 1fr 1fr;
}

@media (max-width: 720px) {
  .pair-item-slots {
    grid-template-columns: 1fr;
  }
}

.pair-item-slot {
  display: grid;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  gap: 3px;
}

/* Dashed, not greyed out: a missing side is an available action, not a disabled one. */
.pair-item-slot.empty {
  border-style: dashed;
  background: none;
}

.pair-item-lang {
  color: var(--muted);
  font-size: var(--font-micro);
}

.pair-item-file {
  overflow: hidden;
  color: var(--text);
  font-size: var(--font-small);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pair-item-slot small {
  color: var(--muted);
  font-size: var(--font-micro);
}

.pair-item-missing {
  font-style: italic;
}

.pair-item-add {
  justify-self: start;
  margin-top: 4px;
  padding: 0;
  border: 0;
  color: var(--mint-strong);
  background: none;
  cursor: pointer;
  font: inherit;
  font-size: var(--font-micro);
}

.pair-item-add:hover {
  text-decoration: underline;
}

.pair-item-delete {
  justify-self: start;
  margin-top: 4px;
}

.pair-item-note {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: var(--font-micro);
}
</style>
