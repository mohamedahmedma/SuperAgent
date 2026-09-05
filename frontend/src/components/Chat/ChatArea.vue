<template>
  <div :class="['chat-workspace', { 'advanced-mode': showAdvanced }]">
    <section class="chat-area">
      <header class="chat-header">
        <div class="header-info">
          <h1>{{ sessionTitle }}</h1>
        </div>
      </header>

      <div class="chat-container" ref="chatContainerRef" @scroll.passive="onScroll">
        <WelcomeScreen v-if="chatStore.messages.length === 0" />

        <div v-if="chatStore.isLoadingOlderMessages" class="older-messages-status">
          <i class="fa-solid fa-spinner fa-spin"></i>
          <span>Loading earlier messages…</span>
        </div>

        <div
          v-else-if="chatStore.messages.length && !chatStore.canLoadOlderMessages && hasPaged"
          class="older-messages-status older-messages-start"
        >
          <span>Beginning of this conversation</span>
        </div>

        <MessageItem
          v-for="(msg, index) in chatStore.messages"
          :key="index"
          :msg="msg"
          :msg-index="index"
          :show-advanced="showAdvanced"
          :ref="(el) => { if (el) messageItemRefs[index] = el; }"
          @cite-click="scrollToChunk"
        />
      </div>

      <ChatInput />
    </section>

    <KnowledgeContextPanel v-if="showAdvanced" @cite-click="scrollToChunk" />
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUpdate, onMounted, ref, watch } from 'vue';
import WelcomeScreen from './WelcomeScreen.vue';
import MessageItem from './MessageItem.vue';
import ChatInput from './ChatInput.vue';
import KnowledgeContextPanel from './KnowledgeContextPanel.vue';
import { useChatStore } from '@/stores/chat';
import { useSessionStore } from '@/stores/sessions';

const chatStore = useChatStore();
const sessionStore = useSessionStore();
const showAdvanced = computed(() => false);

const chatContainerRef = ref<HTMLDivElement | null>(null);
const messageItemRefs = ref<any[]>([]);

const sessionTitle = computed(() => {
  const session = sessionStore.sessions.find((item) => item.session_id === chatStore.sessionId);
  if (session?.title) return session.title;
  const firstUserMessage = chatStore.messages.find((message) => message.isUser && message.text.trim());
  if (!firstUserMessage) return 'New conversation';
  const text = firstUserMessage.text.trim();
  return text.length > 28 ? text.slice(0, 28) + '…' : text;
});

onBeforeUpdate(() => {
  messageItemRefs.value = [];
});

const hasPaged = computed(() => !!chatStore.pagingBySession[chatStore.sessionId]);
const isPrepending = ref(false);
const LOAD_OLDER_THRESHOLD_PX = 120;

const scrollToBottom = () => {
  if (chatContainerRef.value) {
    chatContainerRef.value.scrollTop = chatContainerRef.value.scrollHeight;
  }
};

const onScroll = async () => {
  const container = chatContainerRef.value;
  if (!container || isPrepending.value) return;
  if (container.scrollTop > LOAD_OLDER_THRESHOLD_PX) return;
  if (!chatStore.canLoadOlderMessages) return;

  const sessionId = chatStore.sessionId;
  const previousBottomOffset = container.scrollHeight - container.scrollTop;
  isPrepending.value = true;

  try {
    await chatStore.loadOlderMessages(sessionId);
    await nextTick();
    if (chatContainerRef.value && chatStore.sessionId === sessionId) {
      chatContainerRef.value.scrollTop =
        chatContainerRef.value.scrollHeight - previousBottomOffset;
    }
  } catch (error: any) {
    console.warn('Could not load earlier messages:', error?.message || error);
  } finally {
    await nextTick();
    isPrepending.value = false;
  }
};

const scrollToChunk = async (msgIndex: number, chunkIndex: number) => {
  const msgItem = messageItemRefs.value[msgIndex];
  if (!msgItem) return;

  msgItem.openReferences();
  await nextTick();

  const chunkEl = document.getElementById('chunk-' + msgIndex + '-' + chunkIndex);
  if (chunkEl) {
    chunkEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    chunkEl.classList.add('highlight-chunk');
    window.setTimeout(() => chunkEl.classList.remove('highlight-chunk'), 2000);
  }
};

watch(
  () => chatStore.messages,
  () => {
    if (isPrepending.value) return;
    nextTick(scrollToBottom);
  },
  { deep: true }
);

watch(
  () => chatStore.sessionId,
  () => nextTick(scrollToBottom)
);

onMounted(scrollToBottom);
</script>

<style scoped>
.older-messages-status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 0 14px;
  font-size: 0.78rem;
  color: var(--text-muted, #8a8a8a);
}
.older-messages-start { opacity: 0.7; }
</style>
