<template>
  <div class="settings-panel">
    <header class="settings-header">
      <div>
        <span class="panel-eyebrow">Agent memory</span>
        <h1>Knowledge Base</h1>
        <p>Manage the documents, indexes, and data sources Agent can search.</p>
      </div>
      <button
        type="button"
        class="settings-refresh-btn"
        :disabled="documentStore.documentsLoading"
        @click="onRefresh"
      >
        <i class="fa-solid fa-rotate" :class="{ 'fa-spin': documentStore.documentsLoading }"></i>
        Refresh data
      </button>
    </header>

    <section class="settings-stats">
      <article>
        <span>Total documents</span>
        <strong>{{ documentStore.documents.length }}</strong>
        <small>Current knowledge space</small>
      </article>
      <article>
        <span>Searchable chunks</span>
        <strong>{{ totalChunks.toLocaleString() }}</strong>
        <small>Milvus leaf chunks</small>
      </article>
      <article>
        <span>Index status</span>
        <strong>{{ documentStore.documentsLoading ? 'Syncing' : 'Normal' }}</strong>
        <small>{{ documentStore.isUploading ? 'Processing new document' : 'Service connected' }}</small>
      </article>
      <article>
        <span>Supported formats</span>
        <strong>5</strong>
        <small>PDF · Word · Excel · HTML</small>
      </article>
    </section>

    <div class="settings-grid">
      <section class="documents-section">
        <div class="documents-section-head">
          <div>
            <h2>All documents</h2>
            <p>{{ filteredDocuments.length }} files available for Agent to search</p>
          </div>
          <label class="document-search">
            <i class="fa-solid fa-magnifying-glass"></i>
            <input v-model="searchQuery" type="search" placeholder="Search document names…" />
          </label>
        </div>

        <div class="document-table-head">
          <span>Name</span>
          <span>Chunks</span>
          <span>Status</span>
          <span></span>
        </div>

        <div v-if="documentStore.documentsLoading" class="loading-indicator">
          <span class="loading-orb"><i class="fa-solid fa-spinner fa-spin"></i></span>
          <strong>Syncing knowledge base</strong>
          <p>Reading document and chunk stats from Milvus.</p>
        </div>

        <div v-else-if="filteredDocuments.length === 0" class="empty-documents">
          <span class="empty-icon"><i class="fa-regular fa-folder-open"></i></span>
          <h3>{{ searchQuery ? 'No matching documents' : 'Your knowledge base is empty' }}</h3>
          <p>{{ searchQuery ? 'Try a different keyword.' : 'Upload your first file on the right to get Agent started.' }}</p>
        </div>

        <div v-else class="documents-list">
          <DocumentItem
            v-for="doc in filteredDocuments"
            :key="doc.filename"
            :doc="doc"
          />
        </div>
      </section>

      <UploadSection />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import UploadSection from './UploadSection.vue';
import DocumentItem from './DocumentItem.vue';
import { useDocumentStore } from '@/stores/documents';

const documentStore = useDocumentStore();
const searchQuery = ref('');

const totalChunks = computed(() => documentStore.documents.reduce(
  (total, document) => total + Number(document.chunk_count || 0),
  0
));

const filteredDocuments = computed(() => {
  const query = searchQuery.value.trim().toLowerCase();
  if (!query) return documentStore.documents;
  return documentStore.documents.filter((document) =>
    document.filename.toLowerCase().includes(query)
    || document.file_type.toLowerCase().includes(query)
  );
});

const onRefresh = async () => {
  try {
    await documentStore.loadDocuments();
  } catch (error: any) {
    alert(error.message);
  }
};

onMounted(onRefresh);

onUnmounted(() => {
  documentStore.stopAllDeleteJobPolling();
});
</script>
