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

/**
 * One image of a document and what extraction made of it.
 *
 * Mirrors `backend/schemas/documents.py:AssetInfo`. Deliberately NOT `AssetReference`,
 * which is the public contract a chat client consumes: this carries the model, its
 * confidence and the error behind a failure, which is what an admin needs to judge an
 * extraction and what a parent should never be shown.
 */
export interface AssetInfo {
  asset_id: string;
  sha256: string;
  page_number: number;
  status: string;
  role: string;
  tier: string;
  /** Whether this image produces a retrievable chunk at all. */
  indexable: boolean;
  caption: string;
  description: string;
  transcription: string;
  tags: string[];
  model_used: string;
  confidence: number;
  /** Set by the extractor on every run — a low vision confidence, or no text at all. */
  needs_review: boolean;
  error: string;
  width: number;
  height: number;
  byte_size: number;
  content_type: string;
  /** Authenticated GET returns the bytes, so the reviewer can compare what was read
   *  against the picture it claims to describe. */
  url: string;
  /** The chunks this image produced. The link is stored the other way round and
   *  inverted by the server. */
  chunk_ids: string[];
}

export interface DocumentAssetList {
  filename: string;
  assets: AssetInfo[];
  total: number;
  needs_review_count: number;
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
