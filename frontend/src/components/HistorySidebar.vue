<template>
  <div v-if="sessionStore.showHistorySidebar" class="history-backdrop" @click.self="closeHistory">
    <aside class="history-sidebar">
      <div class="history-header">
        <div>
          <span class="panel-eyebrow">Conversation memory</span>
          <h2>Conversation History</h2>
        </div>
        <button type="button" class="close-btn" aria-label="Close conversation history" @click="closeHistory">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <div class="history-summary">
        <span><strong>{{ sessionStore.sessions.length }}</strong> sessions</span>
        <button type="button" @click="refreshSessions">
          <i class="fa-solid fa-rotate" :class="{ 'fa-spin': refreshing }"></i>
          Refresh
        </button>
      </div>

      <div class="history-list">
        <div v-if="sessionStore.sessions.length === 0" class="empty-history">
          <span class="empty-icon"><i class="fa-regular fa-comments"></i></span>
          <h3>No history yet</h3>
          <p>Start a new conversation and Mew will save it here for you.</p>
        </div>

        <article
          v-for="session in sessionStore.sessions"
          :key="session.session_id"
          :class="['history-item', { active: session.session_id === chatStore.sessionId }]"
        >
          <button type="button" class="session-body" @click="onLoadSession(session.session_id)">
            <span class="session-state-dot" aria-hidden="true"></span>
            <span class="session-info">
              <strong class="session-title">{{ session.title || 'Untitled session' }}</strong>
              <span class="session-meta">
                <span>{{ session.message_count }} messages</span>
                <span v-if="session.isStreaming" class="session-status">Generating</span>
                <span>{{ formatDate(session.updated_at) }}</span>
              </span>
            </span>
          </button>
          <button
            type="button"
            class="history-delete-btn"
            title="Delete session"
            aria-label="Delete session"
            @click.stop="onDeleteSession(session.session_id)"
          >
            <i class="fa-regular fa-trash-can"></i>
          </button>
        </article>
      </div>
    </aside>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useChatStore } from '@/stores/chat';
import { useSessionStore } from '@/stores/sessions';

const chatStore = useChatStore();
const sessionStore = useSessionStore();
const refreshing = ref(false);

const closeHistory = () => {
  sessionStore.showHistorySidebar = false;
  if (chatStore.activeNav === 'history') {
    chatStore.activeNav = 'newChat';
  }
};

const refreshSessions = async () => {
  refreshing.value = true;
  try {
    await sessionStore.fetchSessions();
    chatStore.mergeCachedSessionsIntoHistory();
  } catch (error: any) {
    alert(error.message);
  } finally {
    refreshing.value = false;
  }
};

const onLoadSession = async (sessionId: string) => {
  try {
    await chatStore.loadSession(sessionId);
  } catch (error: any) {
    alert('Failed to load session: ' + error.message);
  }
};

const onDeleteSession = async (sessionId: string) => {
  if (chatStore.streamingSessionId === sessionId) {
    alert('This session is still generating a response. Stop it or wait for it to finish before deleting.');
    return;
  }

  const sessionLabel = sessionStore.sessions.find((session) => session.session_id === sessionId)?.title || sessionId;
  if (!confirm('Delete session "' + sessionLabel + '"?')) {
    return;
  }

  try {
    await sessionStore.deleteSession(sessionId);
    delete chatStore.messagesBySession[sessionId];
    if (chatStore.sessionId === sessionId) {
      chatStore.handleNewChat();
    } else {
      chatStore.mergeCachedSessionsIntoHistory();
    }
  } catch (error: any) {
    alert('Failed to delete session: ' + error.message);
  }
};

const formatDate = (value: string) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Just now';
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};
</script>
