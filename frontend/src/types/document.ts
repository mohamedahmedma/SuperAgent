export interface DocumentItem {
  filename: string;
  file_type: string;
  chunk_count: number;
}

/**
 * One knowledge-base entry: the same document in up to two languages.
 *
 * `paired` is what retrieval keys on — only when BOTH sides are present does a
 * question in one language stop seeing the other. A row with a single side is a
 * complete entry that answers questions in either language, so the UI must not
 * present it as incomplete.
 */
export interface DocumentPair {
  pair_id: string;
  title: string;
  filename_ar: string;
  filename_en: string;
  paired: boolean;
  chunk_count_ar: number;
  chunk_count_en: number;
  /** Indexed outside the pairing system — uploaded before it existed, or via the
   *  single-file route. Still answers questions; offered so it can be filed. */
  unassigned: boolean;
}

/**
 * One indexed chunk, as the admin inspector shows it.
 *
 * Mirrors `backend/schemas/documents.py:ChunkInfo` field for field. Everything is read
 * out of the stores rather than recomputed, because the point of the view is to show
 * what retrieval will actually see.
 */
export interface ChunkInfo {
  chunk_id: string;
  /** Empty on a level-1 chunk, which is how the tree finds its roots. */
  parent_chunk_id: string;
  root_chunk_id: string;
  /** 1 and 2 come from Postgres, 3 from Milvus — only leaves are vectorised. */
  chunk_level: number;
  chunk_idx: number;
  page_number: number;
  modality: string;
  text: string;
  char_count: number;
  asset_ids: string[];
  /** Whether the filter matched this chunk. The server MARKS rather than removes, so the
   *  document view keeps its whole structure while the hits light up inside it. */
  matched: boolean;
}

export interface DocumentChunkList {
  filename: string;
  /** Chunks the document has, and how many are in this response. */
  total: number;
  returned: number;
  /** How many of the returned chunks the filter matched. */
  match_count: number;
  query: string;
  truncated: boolean;
  chunks: ChunkInfo[];
}

/** A chunk with its children attached, for the tree view. */
export interface ChunkNode extends ChunkInfo {
  children: ChunkNode[];
}

export interface UploadStep {
  key: string;
  label: string;
  percent: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message: string;
  /**
   * A nested bar inside this step. Figure extraction reports here: it runs inside the
   * `parse` step, so it cannot be a step of its own without appearing to finish while
   * its parent is still going. `subTotal` of 0 means there is nothing to draw.
   */
  subLabel?: string;
  subDone?: number;
  subTotal?: number;
}

export interface UploadJob {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  message: string;
  steps: UploadStep[];
}

export interface DeleteStep {
  key: string;
  label: string;
  percent: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message: string;
}

export interface DeleteJob {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  message: string;
  steps: DeleteStep[];
}

export interface ActiveDeleteJob {
  jobId?: string;
  status: 'running' | 'completed' | 'failed';
  message: string;
  collapsed: boolean;
  steps: DeleteStep[];
}
