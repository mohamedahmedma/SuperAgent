<template>
  <section class="upload-section">
    <div class="upload-section-head">
      <span class="upload-title-icon"><i class="fa-solid fa-cloud-arrow-up"></i></span>
      <div>
        <h2>Quick upload</h2>
        <p>Parses structure, splits into three-tier chunks, and writes to the hybrid index.</p>
      </div>
    </div>

    <input
      ref="fileInputRef"
      type="file"
      accept=".pdf,.doc,.docx,.xls,.xlsx,.html,.htm"
      hidden
      @change="onFileSelect"
    />

    <button
      type="button"
      class="upload-dropzone"
      @click="triggerFileSelect"
      @dragover.prevent
      @drop.prevent="onFileDrop"
    >
      <span class="dropzone-icon"><i class="fa-solid fa-arrow-up-from-bracket"></i></span>
      <strong>{{ documentStore.selectedFile ? documentStore.selectedFile.name : 'Drag and drop a file here' }}</strong>
      <span>
        {{ documentStore.selectedFile
          ? formatFileSize(documentStore.selectedFile.size)
          : 'or click to choose a PDF, Word, Excel, or HTML file' }}
      </span>
    </button>

    <div v-if="documentStore.selectedFile" class="selected-file">
      <span class="selected-file-icon"><i class="fa-regular fa-file-lines"></i></span>
      <span class="selected-file-copy">
        <strong>{{ documentStore.selectedFile.name }}</strong>
        <small>{{ formatFileSize(documentStore.selectedFile.size) }} · Waiting to upload</small>
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

const documentStore = useDocumentStore();
const fileInputRef = ref<HTMLInputElement | null>(null);

const completedSteps = computed(() =>
  documentStore.uploadSteps.filter((step) => step.status === 'completed').length
);

const triggerFileSelect = () => {
  fileInputRef.value?.click();
};

const setSelectedFile = (file: File) => {
  documentStore.selectedFile = file;
  documentStore.uploadProgress = '';
  documentStore.uploadSteps = documentStore.createUploadSteps();
  documentStore.uploadProgressCollapsed = false;
  documentStore.activeUploadJobId = '';
};

const onFileSelect = (event: Event) => {
  const files = (event.target as HTMLInputElement).files;
  if (files?.length) setSelectedFile(files[0]);
};

const onFileDrop = (event: DragEvent) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) setSelectedFile(file);
};

const onUpload = async () => {
  try {
    await documentStore.uploadDocument();
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
