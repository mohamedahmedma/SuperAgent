import { createPinia, setActivePinia } from 'pinia';
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { useDocumentStore } from './documents';
import api from '@/utils/api';

vi.mock('@/utils/api', () => ({
  default: { get: vi.fn(), post: vi.fn(), delete: vi.fn() },
}));

const chunk = (overrides: Record<string, any> = {}) => ({
  chunk_id: 'c1',
  parent_chunk_id: '',
  root_chunk_id: '',
  chunk_level: 3,
  chunk_idx: 0,
  page_number: 1,
  modality: 'text',
  text: 'the fees are 105,000 EGP',
  char_count: 24,
  asset_ids: [],
  matched: false,
  ...overrides,
});

const payload = (overrides: Record<string, any> = {}) => ({
  filename: 'kb.docx',
  total: 1,
  returned: 1,
  match_count: 0,
  query: '',
  truncated: false,
  chunks: [chunk()],
  ...overrides,
});

describe('the chunk inspector store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('opens on one file and asks the backend for that file', async () => {
    (api.get as any).mockResolvedValue({ data: payload() });
    const store = useDocumentStore();

    await store.openInspector('fees_en.docx');

    expect(api.get).toHaveBeenCalledWith('/documents/fees_en.docx/chunks', { params: {} });
    expect(store.inspecting).toBe('fees_en.docx');
    expect(store.chunkList?.chunks).toHaveLength(1);
  });

  it('encodes a filename that would otherwise break the path', async () => {
    (api.get as any).mockResolvedValue({ data: payload() });
    const store = useDocumentStore();

    await store.openInspector('رسوم 2026/الرسوم.docx');

    expect(api.get).toHaveBeenCalledWith(
      `/documents/${encodeURIComponent('رسوم 2026/الرسوم.docx')}/chunks`,
      { params: {} },
    );
  });

  it('sends the filter as a query parameter rather than folding it here', async () => {
    // Folding lives on the server so there is ONE implementation of it. A second one in
    // TypeScript would drift from the first time either was edited, and the filter would
    // quietly stop agreeing with retrieval.
    (api.get as any).mockResolvedValue({ data: payload({ query: 'المدرسة' }) });
    const store = useDocumentStore();
    store.inspecting = 'kb.docx';
    store.chunkQuery = 'المدرسة';

    await store.loadChunks();

    expect(api.get).toHaveBeenCalledWith('/documents/kb.docx/chunks', {
      params: { q: 'المدرسة' },
    });
  });

  it('omits the parameter entirely when nothing is typed', async () => {
    (api.get as any).mockResolvedValue({ data: payload() });
    const store = useDocumentStore();
    store.inspecting = 'kb.docx';

    await store.loadChunks();

    expect(api.get).toHaveBeenCalledWith('/documents/kb.docx/chunks', { params: {} });
  });

  it('ignores a response for a query the box no longer holds', async () => {
    // Typing produces overlapping requests. The last one TYPED must win, not the last
    // one to arrive, or the list settles on a query the user has already moved past.
    const store = useDocumentStore();
    store.inspecting = 'kb.docx';
    store.chunkQuery = 'slow';

    let release: (value: any) => void = () => {};
    (api.get as any).mockReturnValueOnce(new Promise((resolve) => { release = resolve; }));
    const inFlight = store.loadChunks();

    store.chunkQuery = 'fast';
    release({ data: payload({ query: 'slow', chunks: [chunk({ chunk_id: 'stale' })] }) });
    await inFlight;

    expect(store.chunkList).toBeNull();
  });

  it('ignores a response for a document that is no longer open', async () => {
    const store = useDocumentStore();
    store.inspecting = 'first.docx';

    let release: (value: any) => void = () => {};
    (api.get as any).mockReturnValueOnce(new Promise((resolve) => { release = resolve; }));
    const inFlight = store.loadChunks();

    store.inspecting = 'second.docx';
    release({ data: payload({ filename: 'first.docx' }) });
    await inFlight;

    expect(store.chunkList).toBeNull();
  });

  it('reports a failure instead of showing the previous document', async () => {
    (api.get as any).mockResolvedValueOnce({ data: payload() });
    const store = useDocumentStore();
    await store.openInspector('kb.docx');

    (api.get as any).mockRejectedValueOnce({ response: { data: { detail: 'Milvus is down' } } });
    await store.loadChunks();

    expect(store.chunksError).toBe('Milvus is down');
    expect(store.chunkList).toBeNull();
  });

  it('clears the previous document when another is opened', async () => {
    (api.get as any).mockResolvedValue({ data: payload() });
    const store = useDocumentStore();
    await store.openInspector('first.docx');
    store.chunkQuery = 'fees';

    await store.openInspector('second.docx');

    expect(store.chunkQuery).toBe('');
    expect(store.inspecting).toBe('second.docx');
  });

  it('closing leaves nothing behind for the next open to show', async () => {
    (api.get as any).mockResolvedValue({ data: payload() });
    const store = useDocumentStore();
    await store.openInspector('kb.docx');

    store.closeInspector();

    expect(store.inspecting).toBe('');
    expect(store.chunkList).toBeNull();
    expect(store.chunkQuery).toBe('');
    expect(store.chunksError).toBe('');
  });

  it('does nothing at all when no document is open', async () => {
    const store = useDocumentStore();
    await store.loadChunks();
    expect(api.get).not.toHaveBeenCalled();
  });
});
