<template>
  <aside class="knowledge-context">
    <div class="context-header">
      <div>
        <span class="panel-eyebrow">Live evidence</span>
        <h2>Knowledge Trail</h2>
      </div>
      <span :class="['context-status', { running: isRunning }]">
        <i :class="isRunning ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-circle-check'"></i>
        {{ statusLabel }}
      </span>
    </div>

    <div v-if="!latestMessage" class="context-empty">
      <span class="context-empty-icon"><i class="fa-solid fa-wand-magic-sparkles"></i></span>
      <h3>Waiting for a question</h3>
      <p>Retrieval steps, evidence confidence, and citation sources will appear here in real time.</p>
    </div>

    <template v-else>
      <section class="context-card run-card">
        <div class="context-card-heading">
          <div>
            <strong>This agent run</strong>
            <small>{{ shortSessionId }}</small>
          </div>
          <span v-if="totalDuration">{{ totalDuration }}</span>
        </div>

        <div class="run-timeline">
          <div v-for="(step, index) in runSteps" :key="step.key + index" class="run-step">
            <span :class="['run-step-dot', { active: isRunning && index === runSteps.length - 1 }]">
              <i :class="isRunning && index === runSteps.length - 1 ? 'fa-solid fa-ellipsis' : 'fa-solid fa-check'"></i>
            </span>
            <span class="run-step-copy">
              <strong>{{ step.label }}</strong>
              <small v-if="step.detail">{{ step.detail }}</small>
            </span>
            <span v-if="step.time" class="run-step-time">{{ step.time }}</span>
          </div>
        </div>
      </section>

      <section class="context-card confidence-card">
        <div
          class="confidence-ring"
          :style="{ '--confidence': (confidence ?? 0) + '%' }"
        >
          <strong>{{ confidence === null ? '—' : confidence + '%' }}</strong>
        </div>
        <div>
          <strong>Evidence confidence</strong>
          <p>{{ confidenceDescription }}</p>
        </div>
      </section>

      <div class="context-section-heading">
        <strong>Cited sources</strong>
        <span>{{ sources.length ? sources.length + ' chunks' : 'No citations yet' }}</span>
      </div>

      <div v-if="sources.length" class="context-sources">
        <button
          v-for="(source, index) in sources.slice(0, 5)"
          :key="source.filename + index"
          type="button"
          class="context-source-card"
          @click="onSourceClick(index)"
        >
          <span class="context-file-icon"><i class="fa-regular fa-file-lines"></i></span>
          <span class="context-source-copy">
            <strong>{{ source.filename }}</strong>
            <small>{{ sourceLocation(source, index) }}</small>
          </span>
          <span class="context-source-score">{{ formatScore(source.rerank_score) }}</span>
        </button>
      </div>

      <div v-else class="context-no-sources">
        <i class="fa-solid fa-route"></i>
        <span>{{ trace?.tool_used === false ? "This was a direct answer — the knowledge base wasn't used." : 'Waiting for citable evidence.' }}</span>
      </div>
    </template>
  </aside>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useChatStore } from '@/stores/chat';
import type { RetrievedChunk } from '@/types/chat';

interface RunStepView {
  key: string;
  label: string;
  detail?: string;
  time?: string;
}

const emit = defineEmits<{
  (e: 'cite-click', msgIndex: number, chunkIndex: number): void;
}>();

const chatStore = useChatStore();

const latestMessageIndex = computed(() => {
  for (let index = chatStore.messages.length - 1; index >= 0; index -= 1) {
    const message = chatStore.messages[index];
    if (!message.isUser && !message.isHitlRequest) return index;
  }
  return -1;
});

const latestMessage = computed(() => {
  const index = latestMessageIndex.value;
  return index >= 0 ? chatStore.messages[index] : null;
});

const trace = computed(() => latestMessage.value?.ragTrace || null);
const sources = computed(() => trace.value?.retrieved_chunks || []);
const isRunning = computed(() => Boolean(latestMessage.value?.isThinking || chatStore.isViewingStreamingSession));
const shortSessionId = computed(() => chatStore.sessionId.replace('session_', '').slice(-8));

const statusLabel = computed(() => {
  if (isRunning.value) return 'Running';
  if (latestMessage.value?.isHitlRequest) return 'Waiting for input';
  return latestMessage.value ? 'Completed' : 'Standing by';
});

const runSteps = computed<RunStepView[]>(() => {
  const streamedSteps = latestMessage.value?.ragSteps || [];
  if (streamedSteps.length) {
    return streamedSteps.slice(-6).map((step, index) => ({
      key: step.key || String(index),
      label: step.label,
      detail: step.detail,
      time: step.elapsed_ms != null ? formatMilliseconds(step.elapsed_ms) : '',
    }));
  }

  const currentTrace = trace.value;
  if (!currentTrace) {
    return [{
      key: 'answer',
      label: isRunning.value ? 'Connecting to Agent' : 'Direct answer completed',
      detail: isRunning.value ? 'Understanding the question and choosing a tool' : 'No retrieval trace was produced this time',
    }];
  }

  const result: RunStepView[] = [];
  if (currentTrace.complexity) {
    result.push({
      key: 'complexity',
      label: 'Intent & complexity check',
      detail: currentTrace.complexity === 'complex' ? 'Complex question · multi-step processing' : 'Simple question · fast path',
    });
  }
  if (currentTrace.sub_agent_count) {
    result.push({
      key: 'sub-agents',
      label: 'Parallel sub-question processing',
      detail: currentTrace.sub_agent_count + ' agents',
    });
  }
  if (currentTrace.tool_used || currentTrace.retrieval_mode) {
    result.push({
      key: 'retrieve',
      label: currentTrace.retrieval_mode || 'Knowledge base retrieval',
      detail: formatRetrievalFunnel(currentTrace),
    });
  }
  if (currentTrace.rerank_applied || currentTrace.rerank_enabled) {
    result.push({
      key: 'rerank',
      label: 'Evidence reranking',
      detail: currentTrace.rerank_model || 'Rerank',
    });
  }
  result.push({
    key: 'synthesis',
    label: 'Evidence synthesis',
    detail: sources.value.length ? sources.value.length + ' sources aligned' : 'Answer generation complete',
  });
  return result;
});

const confidence = computed<number | null>(() => {
  const raw = trace.value?.evidence_confidence;
  if (raw === null || raw === undefined || Number.isNaN(Number(raw))) return null;
  const normalized = Number(raw) <= 1 ? Number(raw) * 100 : Number(raw);
  return Math.max(0, Math.min(100, Math.round(normalized)));
});

const confidenceDescription = computed(() => {
  if (confidence.value === null) {
    return sources.value.length ? 'Citations found — waiting for an evidence score.' : 'No evidence score is available for this answer.';
  }
  if (confidence.value >= 85) return 'Sources are highly relevant, with no obvious conflicts between them.';
  if (confidence.value >= 60) return "Evidence is generally usable — it's worth checking the original citations too.";
  return 'Evidence support is weak — use with caution and verify further.';
});

const totalDuration = computed(() => {
  const total = (latestMessage.value?.ragSteps || []).reduce(
    (sum, step) => sum + Number(step.elapsed_ms || 0),
    0
  );
  return total > 0 ? formatMilliseconds(total) : '';
});

const formatMilliseconds = (milliseconds: number) => {
  if (milliseconds < 1000) return Math.round(milliseconds) + 'ms';
  return (milliseconds / 1000).toFixed(1) + 's';
};

const formatRetrievalFunnel = (currentTrace: any) => {
  const recall = currentTrace.recall_count;
  const output = currentTrace.retrieved_chunks?.length;
  if (recall != null && output != null) return recall + ' → ' + output + ' chunks';
  return currentTrace.retrieval_pipeline || 'Hybrid recall';
};

const sourceLocation = (source: RetrievedChunk, index: number) => {
  const parts: string[] = [];
  if (source.page_number) parts.push('Page ' + source.page_number);
  parts.push('RRF #' + (source.rrf_rank || index + 1));
  return parts.join(' · ');
};

const formatScore = (score?: number | null) => {
  if (score === null || score === undefined) return '—';
  return Number(score).toFixed(2);
};

const onSourceClick = (index: number) => {
  if (latestMessageIndex.value < 0) return;
  emit('cite-click', latestMessageIndex.value, index + 1);
};
</script>
