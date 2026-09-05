<template>
  <div
    v-if="!msg.isHitlRequest && !msg.isHitlAnswer"
    :class="['message', msg.isUser ? 'user-message' : 'bot-message']"
  >
    <div v-if="!msg.isUser" class="message-avatar" aria-hidden="true"><BrandLogo size="sm" /></div>

    <div class="message-column">
      <div v-if="!msg.isUser" class="message-author">
        <span>Aurexis School Assistant</span>
        <small v-if="showAdvanced && msg.ragTrace?.retrieved_chunks?.length">
          Cited {{ msg.ragTrace.retrieved_chunks.length }} sources
        </small>
      </div>

      <template v-if="msg.isUser">
        <MessageContent :text="msg.text" :is-user="true" :msg-index="msgIndex" />
      </template>

      <template v-else>
        <div v-if="msg.hitlResumeText" class="hitl-resume-note">
          <i class="fa-solid fa-rotate-right"></i>
          <span>Added: {{ msg.hitlResumeText }} — continuing the original flow</span>
        </div>

        <ThinkingTrace
          v-if="msg.isThinking && !msg.text"
          :msg="msg"
          :msg-index="msgIndex"
        />

        <template v-else>
          <MessageContent
            :text="msg.text"
            :is-user="false"
            :msg-index="msgIndex"
            @cite-click="onCiteClick"
          />
          <MessageAssets :assets="msg.assets" />
          <References
            v-if="showAdvanced"
            ref="referencesRef"
            :msg="msg"
            :msg-index="msgIndex"
            @cite-click="onCiteClick"
          />
          <RetrievalTraceDetails v-if="showAdvanced" :msg="msg" />
        </template>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import BrandLogo from '@/components/BrandLogo.vue';
import { ref } from 'vue';
import MessageAssets from './MessageAssets.vue';
import MessageContent from './MessageContent.vue';
import ThinkingTrace from './ThinkingTrace.vue';
import References from './References.vue';
import RetrievalTraceDetails from './RetrievalTraceDetails.vue';
import type { Message } from '@/types/chat';

defineProps<{
  msg: Message;
  msgIndex: number;
  showAdvanced?: boolean;
}>();

const emit = defineEmits<{
  (e: 'cite-click', msgIndex: number, chunkIndex: number): void;
}>();

const referencesRef = ref<InstanceType<typeof References> | null>(null);

const openReferences = () => {
  referencesRef.value?.openDetails();
};

defineExpose({ openReferences });

const onCiteClick = (msgIndex: number, chunkIndex: number) => {
  emit('cite-click', msgIndex, chunkIndex);
};
</script>
