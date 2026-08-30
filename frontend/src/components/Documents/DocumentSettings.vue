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
        <span>Total entries</span>
        <strong>{{ documentStore.pairs.length }}</strong>
        <small>{{ pairedCount }} in both languages</small>
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
            <p>{{ filteredPairs.length }} entries · {{ pairedCount }} in both languages</p>
          </div>
          <label class="document-search">
            <i class="fa-solid fa-magnifying-glass"></i>
            <input v-model="searchQuery" type="search" placeholder="Search document names…" />
          </label>
        </div>

        <div v-if="documentStore.documentsLoading" class="loading-indicator">
          <span class="loading-orb"><i class="fa-solid fa-spinner fa-spin"></i></span>
          <strong>Syncing knowledge base</strong>
          <p>Reading document and chunk stats from Milvus.</p>
        </div>

        <div v-else-if="filteredPairs.length === 0" class="empty-documents">
          <span class="empty-icon"><i class="fa-regular fa-folder-open"></i></span>
          <h3>{{ searchQuery ? 'No matching documents' : 'Your knowledge base is empty' }}</h3>
          <p>{{ searchQuery ? 'Try a different keyword.' : 'Upload your first file on the right to get Agent started.' }}</p>
        </div>

        <div v-else class="documents-list">
          <DocumentPairItem
            v-for="pair in filteredPairs"
            :key="pair.pair_id || pair.filename_en || pair.filename_ar"
            :pair="pair"
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
import DocumentPairItem from './DocumentPairItem.vue';
import { useDocumentStore } from '@/stores/documents';

const documentStore = useDocumentStore();
const searchQuery = ref('');

const totalChunks = computed(() => documentStore.documents.reduce(
  (total, document) => total + Number(document.chunk_count || 0),
  0
));

// Searches the title and BOTH filenames: an admin looking for the Arabic version by
// its own name should not have to know what the entry was titled.
const filteredPairs = computed(() => {
  const query = searchQuery.value.trim().toLowerCase();
  if (!query) return documentStore.pairs;
  return documentStore.pairs.filter((pair) =>
    [pair.title, pair.filename_ar, pair.filename_en]
      .some((field) => (field || '').toLowerCase().includes(query))
  );
});

const pairedCount = computed(() => documentStore.pairs.filter((pair) => pair.paired).length);

const onRefresh = async () => {
  try {
    // Both: the pair list drives the UI, and `documents` still feeds the chunk-count
    // stat and the delete-job bookkeeping keyed by filename.
    await Promise.all([documentStore.loadPairs(), documentStore.loadDocuments()]);
  } catch (error: any) {
    alert(error.message);
  }
};

onMounted(onRefresh);

onUnmounted(() => {
  documentStore.stopAllDeleteJobPolling();
});
</script>
