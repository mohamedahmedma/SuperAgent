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

export interface UploadStep {
  key: string;
  label: string;
  percent: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  message: string;
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
