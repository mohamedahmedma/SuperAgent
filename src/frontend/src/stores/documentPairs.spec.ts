import { createPinia, setActivePinia } from 'pinia';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { useDocumentStore } from './documents';
import api from '@/utils/api';

vi.mock('@/utils/api', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
  },
}));

// A real File, not a shape cast to one: FormData coerces any non-Blob value to a
// string, so a stub would be appended as "[object Object]" and every assertion about
// which file went into which slot would pass vacuously.
const asFile = (name: string) => new File(['x'], name);

const readForm = (form: FormData) => ({
  ar: (form.get('file_ar') as File | null)?.name ?? null,
  en: (form.get('file_en') as File | null)?.name ?? null,
  title: form.get('title'),
  pair_id: form.get('pair_id'),
});

describe('bilingual document upload', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
    vi.mocked(api.post).mockResolvedValue({
      data: { job_id: 'job_pair_1', filename: 'fees_ar.docx, fees_en.docx', message: 'Uploaded' },
    });
  });

  afterEach(() => {
    useDocumentStore().stopUploadJobPolling();
    vi.restoreAllMocks();
  });

  it('sends both language slots in one request', async () => {
    const store = useDocumentStore();
    store.selectedFileAr = asFile('fees_ar.docx');
    store.selectedFileEn = asFile('fees_en.docx');
    store.pairTitle = 'Fees policy';

    await store.uploadPair();

    const [url, form] = vi.mocked(api.post).mock.calls[0];
    expect(url).toBe('/documents/upload/pair');
    expect(readForm(form as FormData)).toMatchObject({
      ar: 'fees_ar.docx',
      en: 'fees_en.docx',
      title: 'Fees policy',
    });
  });

  it('accepts a single language, because a one-sided entry is a normal entry', async () => {
    const store = useDocumentStore();
    store.selectedFileEn = asFile('bus_en.docx');

    await store.uploadPair();

    expect(readForm(vi.mocked(api.post).mock.calls[0][1] as FormData)).toMatchObject({
      ar: null,
      en: 'bus_en.docx',
    });
  });

  it('refuses an upload with neither side chosen', async () => {
    const store = useDocumentStore();
    await expect(store.uploadPair()).rejects.toThrow(/Arabic file, an English file, or both/);
    expect(api.post).not.toHaveBeenCalled();
  });

  it('carries the pair id when filling in a missing language', async () => {
    const store = useDocumentStore();
    store.startFillingPair({
      pair_id: 'p_abc',
      title: 'Fees policy',
      filename_ar: '',
      filename_en: 'fees_en.docx',
      paired: false,
      chunk_count_ar: 0,
      chunk_count_en: 12,
      unassigned: false,
    });
    store.selectedFileAr = asFile('fees_ar.docx');

    await store.uploadPair();

    expect(readForm(vi.mocked(api.post).mock.calls[0][1] as FormData)).toMatchObject({
      pair_id: 'p_abc',
      title: 'Fees policy',
      ar: 'fees_ar.docx',
    });
  });

  it('clears the row once the upload is accepted', async () => {
    const store = useDocumentStore();
    store.selectedFileAr = asFile('fees_ar.docx');
    store.pairTitle = 'Fees policy';
    store.pairTargetId = 'p_abc';

    await store.uploadPair();

    expect(store.selectedFileAr).toBeNull();
    expect(store.selectedFileEn).toBeNull();
    expect(store.pairTitle).toBe('');
    expect(store.pairTargetId).toBe('');
    expect(store.activeUploadJobId).toBe('job_pair_1');
  });

  it('keeps the chosen files when the upload is rejected, so they can be re-sent', async () => {
    const store = useDocumentStore();
    vi.mocked(api.post).mockRejectedValue({
      response: { data: { detail: 'fees_en.docx: uploaded as English but 96% Arabic script.' } },
    });
    store.selectedFileAr = asFile('fees_ar.docx');
    store.selectedFileEn = asFile('fees_en.docx');

    await expect(store.uploadPair()).rejects.toThrow(/96% Arabic script/);

    expect(store.selectedFileAr).not.toBeNull();
    expect(store.selectedFileEn).not.toBeNull();
    expect(store.isUploading).toBe(false);
    expect(store.uploadSteps.find((step) => step.key === 'upload')).toMatchObject({
      status: 'failed',
    });
  });

  it('loads the pair list', async () => {
    const store = useDocumentStore();
    vi.mocked(api.get).mockResolvedValue({
      data: {
        pairs: [
          {
            pair_id: 'p_abc',
            title: 'Fees policy',
            filename_ar: 'fees_ar.docx',
            filename_en: 'fees_en.docx',
            paired: true,
            chunk_count_ar: 40,
            chunk_count_en: 38,
            unassigned: false,
          },
        ],
      },
    });

    await store.loadPairs();

    expect(api.get).toHaveBeenCalledWith('/documents/pairs');
    expect(store.pairs).toHaveLength(1);
    expect(store.pairs[0].paired).toBe(true);
    expect(store.pairsLoading).toBe(false);
  });
});
