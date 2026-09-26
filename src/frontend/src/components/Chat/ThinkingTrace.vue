<template>
  <div class="message-content thinking-content">
    <div class="thinking-header">
      <div class="thinking-dots">
        <span class="tdot"></span>
        <span class="tdot"></span>
        <span class="tdot"></span>
      </div>
      <span v-if="!msg.ragSteps || !msg.ragSteps.length" class="thinking-text">Thinking...</span>
      <span v-else class="thinking-text">{{ msg.ragSteps[msg.ragSteps.length - 1].label }}</span>
      <span class="thinking-elapsed">{{ elapsedSeconds }}s elapsed</span>
    </div>

    <div v-if="waitingHint" class="thinking-hint">{{ waitingHint }}</div>
    
    <div v-if="msg.ragSteps && msg.ragSteps.length" class="thinking-trace-lines">
      <template v-for="(grp, gIdx) in msg._groupedSteps" :key="grp.group || `main-${gIdx}`">
        <!-- Sub-agent group: collapsible with title -->
        <div v-if="grp.group" class="step-group">
          <div class="step-group-header" @click="toggleGroup(gIdx)">
            <span class="step-group-arrow" :class="{ collapsed: grp.collapsed }">▶</span>
            <span class="step-group-label"><i class="fa-solid fa-code-branch"></i> Sub-question: {{ grp.label }}</span>
            <span class="step-group-count">{{ grp.steps.length }} steps</span>
          </div>
          <div v-show="!grp.collapsed" class="step-group-body">
            <div v-for="(step, sIdx) in grp.steps" :key="sIdx" class="thinking-trace-line">
              <span class="thinking-trace-icon">{{ step.icon || '▶' }}</span>
              <span class="thinking-trace-label">{{ step.label }}</span>
              <span v-if="step.detail" class="thinking-trace-detail">{{ step.detail }}</span>
              <span v-if="step.elapsed_ms != null" class="thinking-trace-time">{{ formatElapsed(step.elapsed_ms) }}</span>
            </div>
          </div>
        </div>
        
        <!-- Regular step: shown directly -->
        <template v-else>
          <div v-for="(step, sIdx) in grp.steps" :key="'s' + gIdx + '-' + sIdx" class="thinking-trace-line">
            <span class="thinking-trace-icon">{{ step.icon || '▶' }}</span>
            <span class="thinking-trace-label">{{ step.label }}</span>
            <span v-if="step.detail" class="thinking-trace-detail">{{ step.detail }}</span>
            <span v-if="step.elapsed_ms != null" class="thinking-trace-time">{{ formatElapsed(step.elapsed_ms) }}</span>
          </div>
        </template>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { useChatStore } from '@/stores/chat';
import type { Message } from '@/types/chat';

const props = defineProps<{
  msg: Message;
  msgIndex: number;
}>();

const chatStore = useChatStore();
const elapsedSeconds = ref(0);
let timer: ReturnType<typeof setInterval> | null = null;

const updateElapsed = () => {
  const startedAt = props.msg.thinkingStartedAt || Date.now();
  elapsedSeconds.value = Math.max(Math.floor((Date.now() - startedAt) / 1000), 0);
};

const waitingHint = computed(() => {
  if (elapsedSeconds.value >= 15) {
    return 'The upstream model or retrieval service is responding slowly. You can keep waiting or stop the response at any time.';
  }
  if (elapsedSeconds.value >= 10) {
    return 'Still working on it — retrieval and evidence evaluation for complex questions can take longer.';
  }
  return '';
});

onMounted(() => {
  updateElapsed();
  timer = setInterval(updateElapsed, 1000);
});

onUnmounted(() => {
  if (timer) clearInterval(timer);
});

const formatElapsed = (elapsedMs: number) => `${(elapsedMs / 1000).toFixed(1)}s`;

const toggleGroup = (groupIndex: number) => {
  chatStore.toggleStepGroup(props.msgIndex, groupIndex);
};
</script>
