import { defineStore } from 'pinia';
import api from '@/utils/api';
import type { DocumentItem, UploadStep, ActiveDeleteJob, DeleteStep } from '@/types/document';

export const useDocumentStore = defineStore('documents', {
  state: () => ({
    documents: [] as DocumentItem[],
    documentsLoading: false,
    selectedFile: null as File | null,
    isUploading: false,
    uploadProgress: '',
    uploadSteps: [] as UploadStep[],
    uploadProgressCollapsed: false,
    activeUploadJobId: '',
    uploadPollTimer: null as any,
    deleteJobs: {} as Record<string, ActiveDeleteJob>,
    deletePollTimers: {} as Record<string, any>,
    deleteRemoveTimers: {} as Record<string, any>,
  }),

  actions: {
    createUploadSteps(): UploadStep[] {
      return [
        { key: 'upload', label: 'Document upload', percent: 0, status: 'pending', message: '' },
        { key: 'cleanup', label: 'Clean up old version', percent: 0, status: 'pending', message: '' },
        { key: 'parse', label: 'Parse & chunk', percent: 0, status: 'pending', message: '' },
        { key: 'parent_store', label: 'Store parent chunks', percent: 0, status: 'pending', message: '' },
        { key: 'vector_store', label: 'Vectorize & store', percent: 0, status: 'pending', message: '' },
      ];
    },

    createDeleteSteps(): DeleteStep[] {
      return [
        { key: 'prepare', label: 'Prepare deletion', percent: 0, status: 'pending', message: '' },
        { key: 'bm25', label: 'Sync BM25 stats', percent: 0, status: 'pending', message: '' },
        { key: 'milvus', label: 'Delete vector data', percent: 0, status: 'pending', message: '' },
        { key: 'parent_store', label: 'Delete parent chunks', percent: 0, status: 'pending', message: '' },
      ];
    },

    updateUploadStep(key: string, percent: number, status: UploadStep['status'] = 'running', message = '') {
      if (!this.uploadSteps.length) {
        this.uploadSteps = this.createUploadSteps();
      }
      const idx = this.uploadSteps.findIndex((step) => step.key === key);
      if (idx === -1) return;
      this.uploadSteps[idx] = {
        ...this.uploadSteps[idx],
        percent: Math.max(0, Math.min(100, Math.round(percent || 0))),
        status,
        message,
      };
    },

    mergeDocumentsWithActiveDeletes(nextDocuments: DocumentItem[]): DocumentItem[] {
      const merged = Array.isArray(nextDocuments) ? [...nextDocuments] : [];
      Object.keys(this.deleteJobs).forEach((filename) => {
        const job = this.deleteJobs[filename];
        if (!job || job.status === 'failed') return;
        const exists = merged.some((doc) => doc.filename === filename);
        if (!exists) {
          const currentDoc = this.documents.find((doc) => doc.filename === filename);
          if (currentDoc) {
            merged.push(currentDoc);
          }
        }
      });
      return merged;
    },

    async loadDocuments() {
      this.documentsLoading = true;
      try {
        const response = await api.get('/documents');
        this.documents = this.mergeDocumentsWithActiveDeletes(response.data.documents || []);
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || 'Failed to load document list';
        throw new Error(errMsg);
      } finally {
        this.documentsLoading = false;
      }
    },

    async uploadDocument() {
      if (!this.selectedFile) {
        throw new Error('Please select a file first');
      }

      this.isUploading = true;
      this.uploadProgress = 'Uploading...';
      this.uploadSteps = this.createUploadSteps();
      this.uploadProgressCollapsed = false;
      this.updateUploadStep('upload', 0, 'running', 'Preparing upload');

      const formData = new FormData();
      formData.append('file', this.selectedFile);

      try {
        const response = await api.post('/documents/upload/async', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            if (!progressEvent.total) return;
            const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            this.updateUploadStep('upload', percent, 'running', `Uploaded ${percent}%`);
          },
        });

        const data = response.data;
        this.updateUploadStep('upload', 100, 'completed', 'Document upload complete');
        this.uploadProgress = data.message;
        this.activeUploadJobId = data.job_id;
        this.startUploadJobPolling(data.job_id);
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || 'Upload failed';
        this.updateUploadStep('upload', 100, 'failed', errMsg);
        this.uploadProgress = 'Upload failed: ' + errMsg;
        this.isUploading = false;
        throw new Error(errMsg);
      }
    },

    syncUploadJob(job: any) {
      this.activeUploadJobId = job.job_id;
      this.uploadProgress = job.message || '';
      if (Array.isArray(job.steps)) {
        this.uploadSteps = job.steps.map((step: any) => ({
          key: step.key,
          label: step.label,
          percent: step.percent,
          status: step.status,
          message: step.message || '',
        }));
      }
      if (job.status === 'completed') {
        this.uploadProgressCollapsed = true;
      }
    },

    startUploadJobPolling(jobId: string) {
      this.stopUploadJobPolling();

      const poll = async () => {
        try {
          const response = await api.get(`/documents/upload/jobs/${encodeURIComponent(jobId)}`);
          const job = response.data;
          this.syncUploadJob(job);

          if (job.status === 'completed') {
            this.stopUploadJobPolling();
            this.isUploading = false;
            this.selectedFile = null;
            await this.loadDocuments();
          } else if (job.status === 'failed') {
            this.stopUploadJobPolling();
            this.isUploading = false;
          }
        } catch (error: any) {
          this.uploadProgress = 'Failed to check progress: ' + (error.response?.data?.detail || error.message);
          this.stopUploadJobPolling();
          this.isUploading = false;
        }
      };

      poll();
      this.uploadPollTimer = setInterval(poll, 1000);
    },

    stopUploadJobPolling() {
      if (this.uploadPollTimer) {
        clearInterval(this.uploadPollTimer);
        this.uploadPollTimer = null;
      }
    },

    isDeletingDocument(filename: string): boolean {
      const job = this.deleteJobs[filename];
      return !!(job && job.status === 'running');
    },

    isDeleteActionLocked(filename: string): boolean {
      const job = this.deleteJobs[filename];
      return !!(job && (job.status === 'running' || job.status === 'completed'));
    },

    getDeleteButtonIcon(filename: string): string {
      const job = this.deleteJobs[filename];
      if (job?.status === 'running') return 'fas fa-spinner fa-spin';
      if (job?.status === 'completed') return 'fas fa-check';
      return 'fas fa-trash';
    },

    setDeleteJob(filename: string, nextJob: Partial<ActiveDeleteJob>) {
      this.deleteJobs = {
        ...this.deleteJobs,
        [filename]: {
          ...(this.deleteJobs[filename] || {
            status: 'running',
            message: '',
            collapsed: false,
            steps: this.createDeleteSteps(),
          }),
          ...nextJob,
        },
      };
    },

    syncDeleteJob(filename: string, job: any) {
      const current = this.deleteJobs[filename] || {};
      this.setDeleteJob(filename, {
        jobId: job.job_id,
        status: job.status,
        message: job.message || '',
        collapsed: job.status === 'completed' ? true : Boolean(current.collapsed),
        steps: Array.isArray(job.steps)
          ? job.steps.map((step: any) => ({
              key: step.key,
              label: step.label,
              percent: step.percent,
              status: step.status,
              message: step.message || '',
            }))
          : this.createDeleteSteps(),
      });
    },

    async deleteDocument(filename: string) {
      if (this.isDeletingDocument(filename)) {
        return;
      }
      if (!confirm(`Delete document "${filename}"? This will also remove all related vectors from Milvus.`)) {
        return;
      }

      this.clearDeleteRemovalTimer(filename);
      this.setDeleteJob(filename, {
        status: 'running',
        message: 'Submitting delete job...',
        collapsed: false,
        steps: this.createDeleteSteps().map((step) =>
          step.key === 'prepare'
            ? { ...step, percent: 1, status: 'running' as const, message: 'Submitting delete job' }
            : step
        ),
      });

      try {
        const response = await api.delete(`/documents/delete/async/${encodeURIComponent(filename)}`);
        const data = response.data;
        this.setDeleteJob(filename, {
          jobId: data.job_id,
          status: 'running',
          message: data.message || `Deleting ${filename}`,
          collapsed: false,
        });
        this.startDeleteJobPolling(filename, data.job_id);
      } catch (error: any) {
        const errMsg = error.response?.data?.detail || error.message || 'Delete request failed';
        this.setDeleteJob(filename, {
          status: 'failed',
          message: 'Failed to delete document: ' + errMsg,
          collapsed: false,
          steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps(),
        });
      }
    },

    startDeleteJobPolling(filename: string, jobId: string) {
      this.stopDeleteJobPolling(filename);

      const poll = async () => {
        try {
          const response = await api.get(`/documents/delete/jobs/${encodeURIComponent(jobId)}`);
          const job = response.data;
          this.syncDeleteJob(filename, job);

          if (job.status === 'completed') {
            this.stopDeleteJobPolling(filename);
            this.scheduleDeletedDocumentRemoval(filename);
          } else if (job.status === 'failed') {
            this.stopDeleteJobPolling(filename);
          }
        } catch (error: any) {
          const errMsg = error.response?.data?.detail || error.message || 'Query failed';
          this.setDeleteJob(filename, {
            status: 'failed',
            message: 'Failed to check delete progress: ' + errMsg,
            collapsed: false,
            steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps(),
          });
          this.stopDeleteJobPolling(filename);
        }
      };

      poll();
      this.deletePollTimers = {
        ...this.deletePollTimers,
        [filename]: setInterval(poll, 1000),
      };
    },

    stopDeleteJobPolling(filename: string) {
      const timer = this.deletePollTimers[filename];
      if (!timer) return;
      clearInterval(timer);
      const { [filename]: _, ...rest } = this.deletePollTimers;
      this.deletePollTimers = rest;
    },

    stopAllDeleteJobPolling() {
      Object.keys(this.deletePollTimers).forEach((filename) => this.stopDeleteJobPolling(filename));
    },

    clearDeleteRemovalTimer(filename: string) {
      const timer = this.deleteRemoveTimers[filename];
      if (!timer) return;
      clearTimeout(timer);
      const { [filename]: _, ...rest } = this.deleteRemoveTimers;
      this.deleteRemoveTimers = rest;
    },

    scheduleDeletedDocumentRemoval(filename: string) {
      this.clearDeleteRemovalTimer(filename);
      const timer = setTimeout(async () => {
        this.documents = this.documents.filter((doc) => doc.filename !== filename);
        const { [filename]: _job, ...jobs } = this.deleteJobs;
        const { [filename]: _timer, ...timers } = this.deleteRemoveTimers;
        this.deleteJobs = jobs;
        this.deleteRemoveTimers = timers;
        await this.loadDocuments();
      }, 3000);
      this.deleteRemoveTimers = {
        ...this.deleteRemoveTimers,
        [filename]: timer,
      };
    },

    toggleDeleteJobCollapsed(filename: string) {
      const job = this.deleteJobs[filename];
      if (!job) return;
      this.setDeleteJob(filename, { collapsed: !job.collapsed });
    },
  },
});
export type { DocumentItem };
