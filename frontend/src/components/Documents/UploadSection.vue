<template>
  <section class="upload-section">
    <div class="upload-section-head">
      <span class="upload-title-icon"><i class="fa-solid fa-cloud-arrow-up"></i></span>
      <div>
        <h2>Quick upload</h2>
        <p>One entry, two languages. Upload both versions and each question is answered from the matching one.</p>
      </div>
    </div>

    <input ref="arInputRef" type="file" :accept="ACCEPT" hidden @change="onSelect('ar', $event)" />
    <input ref="enInputRef" type="file" :accept="ACCEPT" hidden @change="onSelect('en', $event)" />

    <div v-if="documentStore.pairTargetId" class="pair-filling-note">
      <i class="fa-solid fa-link"></i>
      Adding to <strong>{{ documentStore.pairTitle }}</strong>
      <button type="button" class="pair-cancel" @click="documentStore.clearPairSelection()">Cancel</button>
    </div>

    <label class="pair-title-field">
      <span>Entry title</span>
      <input
        v-model="documentStore.pairTitle"
        type="text"
        placeholder="e.g. Fees policy 2026"
        :disabled="documentStore.isUploading"
      />
    </label>

    <div class="pair-row">
      <button
        v-for="slot in SLOTS"
        :key="slot.lang"
        type="button"
        :class="['pair-slot', { filled: fileFor(slot.lang) }]"
        :dir="slot.dir"
        @click="triggerSelect(slot.lang)"
        @dragover.prevent
        @drop.prevent="onDrop(slot.lang, $event)"
      >
        <span class="pair-slot-lang">{{ slot.label }}</span>
        <template v-if="fileFor(slot.lang)">
          <strong class="pair-slot-name">{{ fileFor(slot.lang)!.name }}</strong>
          <small>{{ formatFileSize(fileFor(slot.lang)!.size) }}</small>
          <span class="pair-slot-clear" role="button" @click.stop="clearSlot(slot.lang)">
            <i class="fa-solid fa-xmark"></i> Remove
          </span>
        </template>
        <template v-else>
          <span class="pair-slot-icon"><i class="fa-solid fa-arrow-up-from-bracket"></i></span>
          <small>{{ slot.hint }}</small>
        </template>
      </button>
    </div>

    <p class="pair-hint">
      Leave a side empty if that version does not exist — a single-language entry still answers
      questions in both languages.
    </p>

    <div v-if="documentStore.selectedFileAr || documentStore.selectedFileEn" class="selected-file">
      <span class="selected-file-icon"><i class="fa-regular fa-file-lines"></i></span>
      <span class="selected-file-copy">
        <strong>{{ readyLabel }}</strong>
        <small>Waiting to upload</small>
      </span>
      <button
        type="button"
        class="btn-primary"
        :disabled="documentStore.isUploading"
        @click="onUpload"
      >
        <i :class="documentStore.isUploading ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-arrow-up'"></i>
        {{ documentStore.isUploading ? 'Processing' : 'Start upload' }}
      </button>
    </div>

    <div
      v-if="documentStore.uploadSteps.length"
      :class="['upload-progress', { collapsed: documentStore.uploadProgressCollapsed }]"
    >
      <button type="button" class="upload-progress-header" @click="onToggleCollapse">
        <span>
          <strong>{{ documentStore.uploadProgress || 'Upload progress' }}</strong>
          <small>{{ completedSteps }} / {{ documentStore.uploadSteps.length }} stages complete</small>
        </span>
        <span class="upload-toggle">
          {{ documentStore.uploadProgressCollapsed ? 'Expand' : 'Collapse' }}
          <i :class="documentStore.uploadProgressCollapsed ? 'fa-solid fa-chevron-down' : 'fa-solid fa-chevron-up'"></i>
        </span>
      </button>

      <div v-show="!documentStore.uploadProgressCollapsed" class="upload-step-list">
        <div
          v-for="step in documentStore.uploadSteps"
          :key="step.key"
          :class="['upload-step', 'upload-step-' + step.status]"
        >
          <div class="upload-step-header">
            <span class="upload-step-label">
              <i :class="stepIcon(step.status)"></i>
              {{ step.label }}
            </span>
            <span class="upload-step-percent">{{ step.percent }}%</span>
          </div>
          <div class="upload-step-bar">
            <div class="upload-step-fill" :style="{ width: step.percent + '%' }"></div>
          </div>
          <div v-if="step.message" class="upload-step-message">{{ step.message }}</div>
        </div>
      </div>
    </div>

    <div class="upload-pipeline-note">
      <div><span>01</span><p><strong>Structure parsing</strong><small>Detects sections, tables, and pages</small></p></div>
      <div><span>02</span><p><strong>Three-tier chunking</strong><small>Preserves parent-child context</small></p></div>
      <div><span>03</span><p><strong>Hybrid indexing</strong><small>Dense + BM25 written in sync</small></p></div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useDocumentStore } from '@/stores/documents';
import type { UploadStep } from '@/types/document';

type Lang = 'ar' | 'en';

const ACCEPT = '.pdf,.doc,.docx,.xls,.xlsx,.html,.htm';

// The Arabic slot renders RTL so an Arabic filename reads the way it was written.
const SLOTS = [
  { lang: 'ar' as Lang, label: 'العربية · Arabic', dir: 'rtl', hint: 'اختر الملف العربي' },
  { lang: 'en' as Lang, label: 'English', dir: 'ltr', hint: 'Choose the English file' },
];

const documentStore = useDocumentStore();
const arInputRef = ref<HTMLInputElement | null>(null);
const enInputRef = ref<HTMLInputElement | null>(null);

const completedSteps = computed(() =>
  documentStore.uploadSteps.filter((step) => step.status === 'completed').length
);

const fileFor = (lang: Lang) =>
  lang === 'ar' ? documentStore.selectedFileAr : documentStore.selectedFileEn;

const readyLabel = computed(() => {
  const names = [documentStore.selectedFileAr, documentStore.selectedFileEn]
    .filter(Boolean)
    .map((file) => (file as File).name);
  return names.length === 2 ? `${names[0]} + ${names[1]}` : names[0] || '';
});

const triggerSelect = (lang: Lang) => {
  (lang === 'ar' ? arInputRef : enInputRef).value?.click();
};

const setFile = (lang: Lang, file: File) => {
  if (lang === 'ar') documentStore.selectedFileAr = file;
  else documentStore.selectedFileEn = file;
  // A title the admin has not set yet defaults to the first filename's stem, so the
  // common case needs no typing. Never overwrites one they did set.
  if (!documentStore.pairTitle) {
    documentStore.pairTitle = file.name.replace(/\.[^.]+$/, '');
  }
  documentStore.uploadProgress = '';
  documentStore.uploadSteps = documentStore.createUploadSteps();
  documentStore.uploadProgressCollapsed = false;
  documentStore.activeUploadJobId = '';
};

const clearSlot = (lang: Lang) => {
  if (lang === 'ar') documentStore.selectedFileAr = null;
  else documentStore.selectedFileEn = null;
};

const onSelect = (lang: Lang, event: Event) => {
  const files = (event.target as HTMLInputElement).files;
  if (files?.length) setFile(lang, files[0]);
  // Cleared so re-choosing the same file fires `change` again.
  (event.target as HTMLInputElement).value = '';
};

const onDrop = (lang: Lang, event: DragEvent) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) setFile(lang, file);
};

const onUpload = async () => {
  try {
    await documentStore.uploadPair();
  } catch (error: any) {
    alert('Failed to upload document: ' + error.message);
  }
};

const onToggleCollapse = () => {
  documentStore.uploadProgressCollapsed = !documentStore.uploadProgressCollapsed;
};

const formatFileSize = (bytes: number) => {
  if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
};

const stepIcon = (status: UploadStep['status']) => {
  if (status === 'completed') return 'fa-solid fa-check';
  if (status === 'running') return 'fa-solid fa-spinner fa-spin';
  if (status === 'failed') return 'fa-solid fa-xmark';
  return 'fa-solid fa-circle';
};
</script>

<style scoped>
.pair-title-field {
  display: block;
  margin-top: 15px;
}

.pair-title-field span {
  display: block;
  margin-bottom: 6px;
  color: var(--muted);
  font-size: var(--font-caption);
}

.pair-title-field input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  color: var(--text);
  background: var(--surface-soft);
  font: inherit;
}

/* Two equal columns so neither language reads as the primary one. Collapses to
   stacked slots on a narrow screen rather than squeezing two unusable targets. */
.pair-row {
  display: grid;
  gap: 12px;
  margin-top: 12px;
  grid-template-columns: 1fr 1fr;
}

@media (max-width: 720px) {
  .pair-row {
    grid-template-columns: 1fr;
  }
}

.pair-slot {
  display: grid;
  min-height: 128px;
  padding: 16px;
  border: 1px dashed rgba(168, 246, 209, 0.25);
  border-radius: 15px;
  color: var(--muted);
  background: var(--surface-soft);
  cursor: pointer;
  place-items: center;
  align-content: center;
  gap: 6px;
  text-align: center;
  transition: border-color 180ms ease, background 180ms ease;
}

.pair-slot:hover {
  border-color: var(--mint-strong);
  background: var(--surface-hover);
}

.pair-slot.filled {
  border-style: solid;
  border-color: var(--mint-strong);
}

.pair-slot-lang {
  color: var(--text-soft);
  font-size: var(--font-caption);
  letter-spacing: 0.04em;
}

.pair-slot-icon {
  font-size: 18px;
}

/* Long filenames must not widen the grid column. */
.pair-slot-name {
  max-width: 100%;
  overflow: hidden;
  color: var(--text);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pair-slot-clear {
  color: var(--muted);
  font-size: var(--font-micro);
  text-decoration: underline;
}

.pair-slot-clear:hover {
  color: var(--danger);
}

.pair-hint {
  margin: 10px 0 0;
  color: var(--muted);
  font-size: var(--font-micro);
}

.pair-filling-note {
  display: flex;
  align-items: center;
  margin-top: 15px;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  color: var(--text-soft);
  background: var(--surface-strong);
  font-size: var(--font-caption);
  gap: 8px;
}

.pair-cancel {
  margin-left: auto;
  border: 0;
  color: var(--muted);
  background: none;
  cursor: pointer;
  font: inherit;
  text-decoration: underline;
}
</style>
