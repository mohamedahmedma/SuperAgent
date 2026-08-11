<template>
  <div class="chat-workspace">
    <section class="chat-area">
      <header class="chat-header">
        <div class="header-info">
          <h1>{{ sessionTitle }}</h1>
          <span class="header-status-line">
            <span class="status-dot"></span>
            <span>{{ generationStatus }}</span>
            <span>·</span>
            <span>Context synced</span>
          </span>
        </div>
        <div class="chat-header-actions">
          <button type="button" title="History" aria-label="Open conversation history" @click="openHistory">
            <i class="fa-solid fa-clock-rotate-left"></i>
          </button>
          <button type="button" title="Clear current chat" aria-label="Clear current chat" @click="chatStore.handleClearChat">
            <i class="fa-regular fa-trash-can"></i>
          </button>
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
          :ref="(el) => { if (el) messageItemRefs[index] = el; }"
          @cite-click="scrollToChunk"
        />
      </div>

      <ChatInput />
    </section>

    <KnowledgeContextPanel @cite-click="scrollToChunk" />
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

const generationStatus = computed(() => {
  if (chatStore.isViewingStreamingSession) return 'Mew is generating a response';
  if (chatStore.currentPendingHitl) return 'Waiting for your input';
  return 'Mew is online';
});

onBeforeUpdate(() => {
  messageItemRefs.value = [];
});

// True once a session has been loaded from the server, so the "beginning of this
// conversation" marker does not appear above a brand new chat.
const hasPaged = computed(() => !!chatStore.pagingBySession[chatStore.sessionId]);

// Set while older messages are being spliced in above the viewport. The auto-scroll
// watcher below honours it: without that, growing the list upwards would be read as new
// content and throw the reader back down to the newest message.
const isPrepending = ref(false);

// Far enough from the top that the next batch is usually in place before the user
// reaches it, close enough that it is not fetched speculatively.
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
  // Anchor on the distance from the BOTTOM, which the prepend does not change. Restoring
  // scrollTop directly would put the reader wherever the newly inserted messages pushed
  // the one they were looking at.
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

const openHistory = async () => {
  chatStore.activeNav = 'history';
  sessionStore.showHistorySidebar = true;
  try {
    await sessionStore.fetchSessions();
    chatStore.mergeCachedSessionsIntoHistory();
  } catch (error: any) {
    alert(error.message);
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

.older-messages-start {
  opacity: 0.7;
}
</style>
