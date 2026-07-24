<template>
  <div v-if="msg.ragTrace" class="message-meta">
    <details class="reasoning-details">
      <summary>Retrieval process</summary>
      <div class="reasoning-content">
        <div class="trace-line">
          Tool: {{ msg.ragTrace.tool_used ? msg.ragTrace.tool_name : 'Not used' }}
        </div>
        <div v-if="msg.ragTrace.retrieval_stage" class="trace-line">
          Retrieval stage: {{ msg.ragTrace.retrieval_stage }}
        </div>
        <div v-if="msg.ragTrace.retrieval_status" class="trace-line">
          Retrieval status: {{ formatRetrievalStatus(msg.ragTrace.retrieval_status) }}
        </div>
        <div v-if="msg.ragTrace.hitl_resumed" class="trace-line">
          HITL resumed: Yes
          <span v-if="msg.ragTrace.hitl_answer">(Follow-up: {{ msg.ragTrace.hitl_answer }})</span>
        </div>
        <div v-if="msg.ragTrace.hitl_resume_strategy" class="trace-line">
          HITL resume strategy: {{ formatHitlResumeStrategy(msg.ragTrace.hitl_resume_strategy) }}
        </div>
        <div v-if="msg.ragTrace.evidence_relevance || msg.ragTrace.evidence_answerability" class="trace-line">
          Evidence score:
          Relevance {{ msg.ragTrace.evidence_relevance || '—' }} /
          Answerability {{ msg.ragTrace.evidence_answerability || '—' }}
          <span v-if="msg.ragTrace.evidence_confidence !== null && msg.ragTrace.evidence_confidence !== undefined">
            / Confidence {{ Number(msg.ragTrace.evidence_confidence).toFixed(2) }}
          </span>
        </div>
        <div v-if="msg.ragTrace.evidence_ambiguity && msg.ragTrace.evidence_ambiguity !== 'none'" class="trace-line">
          Ambiguity type: {{ msg.ragTrace.evidence_ambiguity }}
        </div>
        <div v-if="msg.ragTrace.hitl_prompt" class="trace-line">
          HITL prompt: {{ msg.ragTrace.hitl_prompt }}
        </div>
        <div v-if="msg.ragTrace.hitl_options && msg.ragTrace.hitl_options.length" class="trace-line">
          HITL options: {{ msg.ragTrace.hitl_options.join(' / ') }}
        </div>
        <div v-if="msg.ragTrace.route" class="trace-line">
          Routing decision: {{ msg.ragTrace.route }}
        </div>
        <div v-if="msg.ragTrace.retrieval_pipeline" class="trace-line">
          Retrieval pipeline: {{ msg.ragTrace.retrieval_pipeline }}
        </div>
        <div v-if="msg.ragTrace.retrieval_mode" class="trace-line">
          Retrieval mode: {{ msg.ragTrace.retrieval_mode }}
        </div>
        <div v-if="msg.ragTrace.candidate_k !== null && msg.ragTrace.candidate_k !== undefined" class="trace-line">
          {{ formatCandidateKLabel(msg.ragTrace) }}
        </div>
        <div v-if="hasRetrievalFunnel(msg.ragTrace)" class="trace-line trace-funnel">
          Retrieval funnel: Milvus recall {{ msg.ragTrace.recall_count ?? '—' }}
          → merged {{ msg.ragTrace.post_merge_candidate_count ?? '—' }}
          → rerank input {{ msg.ragTrace.candidate_count ?? '—' }}
          → output top{{ msg.ragTrace.retrieval_top_k ?? (msg.ragTrace.retrieved_chunks || []).length }}
          ({{ (msg.ragTrace.retrieved_chunks || []).length }} chunks)
        </div>
        <div v-if="msg.ragTrace.leaf_retrieve_level" class="trace-line">
          Leaf recall level: L{{ msg.ragTrace.leaf_retrieve_level }}
        </div>
        <div v-if="msg.ragTrace.auto_merge_enabled !== null && msg.ragTrace.auto_merge_enabled !== undefined" class="trace-line">
          Auto-merging enabled: {{ msg.ragTrace.auto_merge_enabled ? 'Yes' : 'No' }}
        </div>
        <div v-if="msg.ragTrace.auto_merge_applied !== null && msg.ragTrace.auto_merge_applied !== undefined" class="trace-line">
          Auto-merging applied: {{ msg.ragTrace.auto_merge_applied ? 'Yes' : 'No' }}
        </div>
        <div v-if="msg.ragTrace.auto_merge_threshold" class="trace-line">
          Merge threshold: {{ msg.ragTrace.auto_merge_threshold }}
        </div>
        <div v-if="msg.ragTrace.auto_merge_replaced_chunks" class="trace-line">
          Merged/replaced chunks: {{ msg.ragTrace.auto_merge_replaced_chunks }}
        </div>
        <div v-if="msg.ragTrace.auto_merge_steps" class="trace-line">
          Merge rounds: {{ msg.ragTrace.auto_merge_steps }}
        </div>
        <div v-if="msg.ragTrace.rerank_enabled !== null && msg.ragTrace.rerank_enabled !== undefined" class="trace-line">
          Rerank configured: {{ msg.ragTrace.rerank_enabled ? 'Yes' : 'No' }}
        </div>
        <div v-if="msg.ragTrace.rerank_applied !== null && msg.ragTrace.rerank_applied !== undefined" class="trace-line">
          Rerank applied: {{ msg.ragTrace.rerank_applied ? 'Yes' : 'No' }}
        </div>
        <div v-if="msg.ragTrace.rerank_model" class="trace-line">
          Rerank model: {{ msg.ragTrace.rerank_model }}
        </div>
        <div v-if="msg.ragTrace.rerank_error" class="trace-line">
          Rerank status: {{ msg.ragTrace.rerank_error }}
        </div>
        <div v-if="msg.ragTrace.rewrite_method" class="trace-line">
          Query rewrite method: {{ formatRewriteMethod(msg.ragTrace.rewrite_method) }}
        </div>
        <div v-if="msg.ragTrace.step_back_question" class="trace-line">
          Step-back question: {{ msg.ragTrace.step_back_question }}
        </div>
        <div v-if="msg.ragTrace.hyde_document" class="trace-line">
          HyDE hypothetical document: {{ msg.ragTrace.hyde_document }}
        </div>
        <div v-if="msg.ragTrace.rewritten_query" class="trace-line">
          Rewritten search query: {{ msg.ragTrace.rewritten_query }}
        </div>

        <!-- Complexity routing info -->
        <div v-if="msg.ragTrace.complexity" class="trace-line trace-complexity">
          Question complexity:
          <span :class="msg.ragTrace.complexity === 'complex' ? 'tag-complex' : 'tag-simple'">
            {{ msg.ragTrace.complexity === 'complex' ? 'Complex' : 'Simple' }}
          </span>
          <span v-if="msg.ragTrace.complexity_reason" class="trace-detail">({{ msg.ragTrace.complexity_reason }})</span>
        </div>

        <div v-if="msg.ragTrace.sub_questions && msg.ragTrace.sub_questions.length" class="trace-line">
          <div class="trace-sub-questions">
            <div class="trace-sub-qs-header">Sub-question breakdown ({{ msg.ragTrace.sub_questions.length }}):</div>
            <ul class="trace-sub-qs-list">
              <li v-for="(sq, sqIdx) in msg.ragTrace.sub_questions" :key="sqIdx" class="trace-sub-q-item">
                <span class="trace-sub-q-index">{{ sqIdx + 1 }}.</span> {{ sq }}
              </li>
            </ul>
          </div>
        </div>
        <div v-if="msg.ragTrace.sub_agent_count" class="trace-line">
          Parallel sub-agent count: {{ msg.ragTrace.sub_agent_count }}
        </div>
        <div v-if="msg.ragTrace.synthesis_merged_count" class="trace-line">
          Synthesis merged document count: {{ msg.ragTrace.synthesis_merged_count }}
        </div>

        <!-- Sub-agent retrieval details (collapsible) -->
        <div v-if="msg.ragTrace.sub_traces && msg.ragTrace.sub_traces.length" class="trace-sub-traces">
          <details class="sub-traces-details">
            <summary class="sub-traces-title">
              <i class="fas fa-layer-group"></i> Sub-agent retrieval details ({{ msg.ragTrace.sub_traces.length }})
            </summary>
            <div v-for="(st, stIdx) in msg.ragTrace.sub_traces" :key="stIdx" class="sub-trace-block">
              <div class="sub-trace-header">
                Sub-question {{ stIdx + 1 }}: {{ msg.ragTrace.sub_questions?.[stIdx] || st.query || '—' }}
              </div>
              <div v-if="st.retrieval_stage" class="trace-line">Retrieval stage: {{ st.retrieval_stage }}</div>
              <div v-if="st.retrieval_mode" class="trace-line">Retrieval mode: {{ st.retrieval_mode }}</div>
              <div v-if="st.route" class="trace-line">Routing decision: {{ st.route }}</div>
              <div v-if="st.retrieved_chunks && st.retrieved_chunks.length" class="sub-trace-chunks">
                {{ st.retrieved_chunks.length }} chunks retrieved
              </div>
            </div>
          </details>
        </div>

        <div v-if="msg.ragTrace.initial_retrieved_chunks && msg.ragTrace.initial_retrieved_chunks.length" class="sources">
          <div class="sources-title">Initial retrieval results</div>
          <ul class="sources-list">
            <li v-for="(chunk, sIndex) in msg.ragTrace.initial_retrieved_chunks" :key="sIndex" class="source-item">
              <div class="source-title-line">
                <span class="source-file">{{ chunk.filename }}</span>
                <span v-if="chunk.page_number" class="source-page">(Page {{ chunk.page_number }})</span>
              </div>
              <div class="source-meta-line">
                <span class="source-page">RRF rank: #{{ chunk.rrf_rank || (sIndex + 1) }}</span>
                <span v-if="chunk.rerank_score !== null && chunk.rerank_score !== undefined" class="source-page">
                  Rerank score: {{ Number(chunk.rerank_score).toFixed(4) }}
                </span>
              </div>
              <div v-if="chunk.text" class="source-excerpt">{{ chunk.text }}</div>
            </li>
          </ul>
        </div>

        <div v-if="msg.ragTrace.rewrite_retrieved_chunks && msg.ragTrace.rewrite_retrieved_chunks.length" class="sources">
          <div class="sources-title">Retrieval results after rewrite</div>
          <ul class="sources-list">
            <li v-for="(chunk, sIndex) in msg.ragTrace.rewrite_retrieved_chunks" :key="sIndex" class="source-item">
              <div class="source-title-line">
                <span class="source-file">{{ chunk.filename }}</span>
                <span v-if="chunk.page_number" class="source-page">(Page {{ chunk.page_number }})</span>
              </div>
              <div class="source-meta-line">
                <span class="source-page">RRF rank: #{{ chunk.rrf_rank || (sIndex + 1) }}</span>
                <span v-if="chunk.rerank_score !== null && chunk.rerank_score !== undefined" class="source-page">
                  Rerank score: {{ Number(chunk.rerank_score).toFixed(4) }}
                </span>
              </div>
              <div v-if="chunk.text" class="source-excerpt">{{ chunk.text }}</div>
            </li>
          </ul>
        </div>
      </div>
    </details>
  </div>
</template>

<script setup lang="ts">
import type { Message, RagTrace } from '@/types/chat';

defineProps<{
  msg: Message;
}>();

const formatCandidateKLabel = (trace: RagTrace) => {
  if (!trace || trace.candidate_k === null || trace.candidate_k === undefined) {
    return '';
  }
  const k = trace.candidate_k;
  if (trace.candidate_k_config_error) {
    return `Milvus candidate pool: ${k} (${trace.candidate_k_config_error}, fell back to multiplier calculation)`;
  }
  if (trace.candidate_k_source === 'env') {
    return `Milvus candidate pool: ${k} (env var RETRIEVAL_CANDIDATE_K)`;
  }
  const multiplier = trace.retrieval_candidate_multiplier;
  if (multiplier !== null && multiplier !== undefined) {
    return `Milvus candidate pool: ${k} (top_k × ${multiplier})`;
  }
  return `Milvus candidate pool: ${k}`;
};

const hasRetrievalFunnel = (trace: RagTrace) => {
  if (!trace) {
    return false;
  }
  return trace.recall_count !== null && trace.recall_count !== undefined
    || trace.post_merge_candidate_count !== null && trace.post_merge_candidate_count !== undefined
    || trace.candidate_count !== null && trace.candidate_count !== undefined;
};

const formatRetrievalStatus = (status: string) => {
  const labels: Record<string, string> = {
    answerable: 'Answerable',
    partial: 'Partial evidence',
    needs_rewrite: 'Needs rewrite',
    needs_clarification: 'Needs clarification',
    needs_scope_selection: 'Needs scope selection',
    no_knowledge: 'No knowledge available',
  };
  return labels[status] || status;
};

const formatHitlResumeStrategy = (strategy: string) => {
  const labels: Record<string, string> = {
    targeted_retrieval: 'Targeted retrieval based on user follow-up',
  };
  return labels[strategy] || strategy;
};

const formatRewriteMethod = (method: string) => {
  const labels: Record<string, string> = {
    step_back: 'Step-back',
    hyde: 'HyDE',
  };
  return labels[method] || method;
};
</script>
