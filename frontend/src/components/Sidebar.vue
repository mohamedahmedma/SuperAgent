<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="logo-icon" aria-hidden="true">
        <i class="fa-solid fa-cat"></i>
      </div>
      <div class="brand-copy">
        <h1>Mew Assistant</h1>
        <span>Knowledge Copilot</span>
      </div>
    </div>

    <div class="workspace-switcher">
      <span class="workspace-orb" aria-hidden="true"></span>
      <span class="workspace-copy">
        <strong>SuperMew Knowledge Space</strong>
        <small>{{ workspaceMeta }}</small>
      </span>
      <i class="fa-solid fa-chevron-down" aria-hidden="true"></i>
    </div>

    <nav class="sidebar-nav" aria-label="Main navigation">
      <button
        type="button"
        :class="['nav-btn', { active: chatStore.activeNav === 'newChat' }]"
        aria-label="Chat"
        @click="onNewChat"
      >
        <i class="fa-regular fa-message"></i>
        <span>Chat</span>
      </button>
      <button
        type="button"
        :class="['nav-btn', { active: chatStore.activeNav === 'history' }]"
        aria-label="History"
        @click="onHistory"
      >
        <i class="fa-solid fa-clock-rotate-left"></i>
        <span>History</span>
        <small v-if="sessionStore.sessions.length" class="nav-count">
          {{ sessionStore.sessions.length }}
        </small>
      </button>
      <button
        v-if="authStore.isAdmin"
        type="button"
        :class="['nav-btn', { active: chatStore.activeNav === 'settings' }]"
        aria-label="Knowledge base"
        @click="onSettings"
      >
        <i class="fa-regular fa-bookmark"></i>
        <span>Knowledge base</span>
      </button>
    </nav>

    <template v-if="authStore.isAuthenticated">
      <div class="sidebar-section-label">Recent sessions</div>
      <div class="sidebar-recents">
        <button
          v-for="session in recentSessions"
          :key="session.session_id"
          type="button"
          :class="['recent-session', { active: session.session_id === chatStore.sessionId }]"
          @click="onLoadSession(session.session_id)"
        >
          <span class="recent-dot" aria-hidden="true"></span>
          <span class="recent-copy">
            <strong>{{ session.title || 'Untitled session' }}</strong>
            <small>
              {{ session.isStreaming ? 'Generating' : session.message_count + ' messages' }}
              · {{ formatRelativeTime(session.updated_at) }}
            </small>
          </span>
        </button>

        <div v-if="!recentSessions.length" class="recent-empty">
          No conversations yet — ask Mew a question.
        </div>
      </div>
    </template>

    <div class="sidebar-bottom">
      <div class="theme-control">
        <span class="theme-control-label">
          <i :class="theme === 'light' ? 'fa-regular fa-sun' : 'fa-regular fa-moon'"></i>
          <span>{{ theme === 'light' ? 'Light mode' : 'Dark mode' }}</span>
        </span>
        <ThemeToggle :theme="theme" @toggle="$emit('toggle-theme')" />
      </div>

      <div v-if="authStore.isAuthenticated" class="user-panel">
        <span class="user-avatar">{{ userInitials }}</span>
        <span class="user-copy">
          <strong>{{ authStore.currentUser?.username }}</strong>
          <small>{{ roleLabel }}</small>
        </span>
        <span class="user-actions">
          <button type="button" title="Clear current chat" aria-label="Clear current chat" @click="chatStore.handleClearChat">
            <i class="fa-regular fa-trash-can"></i>
          </button>
          <button type="button" title="Log out" aria-label="Log out" @click="onLogout">
            <i class="fa-solid fa-arrow-right-from-bracket"></i>
          </button>
        </span>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue';
import ThemeToggle from '@/components/ThemeToggle.vue';
import { useAuthStore } from '@/stores/auth';
import { useChatStore } from '@/stores/chat';
import { useSessionStore } from '@/stores/sessions';

defineProps<{
  theme: 'dark' | 'light';
}>();

defineEmits<{
  (e: 'toggle-theme'): void;
}>();

const authStore = useAuthStore();
const chatStore = useChatStore();
const sessionStore = useSessionStore();

const recentSessions = computed(() => sessionStore.sessions.slice(0, 4));

const workspaceMeta = computed(() => {
  if (!authStore.isAuthenticated) return 'Log in to connect your private knowledge';
  return (sessionStore.sessions.length || 0) + ' sessions · Private';
});

const roleLabel = computed(() => authStore.currentUser?.role === 'admin' ? 'Administrator' : 'Standard user');

const userInitials = computed(() => {
  const name = authStore.currentUser?.username || 'ME';
  return name.slice(0, 2).toUpperCase();
});

const refreshSessions = async () => {
  if (!authStore.isAuthenticated) return;
  try {
    await sessionStore.fetchSessions();
    chatStore.mergeCachedSessionsIntoHistory();
  } catch (error) {
    console.warn('Failed to load conversation history', error);
  }
};

watch(
  () => authStore.isAuthenticated,
  (isAuthenticated) => {
    if (isAuthenticated) refreshSessions();
  },
  { immediate: true }
);

const onNewChat = () => {
  chatStore.handleNewChat();
};

const onHistory = async () => {
  chatStore.activeNav = 'history';
  sessionStore.showHistorySidebar = !sessionStore.showHistorySidebar;
  if (sessionStore.showHistorySidebar) {
    try {
      await sessionStore.fetchSessions();
      chatStore.mergeCachedSessionsIntoHistory();
    } catch (error: any) {
      alert(error.message);
    }
  }
};

const onSettings = () => {
  if (!authStore.isAdmin) {
    alert('Only administrators can access document management');
    return;
  }
  chatStore.activeNav = 'settings';
  sessionStore.showHistorySidebar = false;
};

const onLoadSession = async (sessionId: string) => {
  try {
    await chatStore.loadSession(sessionId);
  } catch (error: any) {
    alert('Failed to load session: ' + error.message);
  }
};

const onLogout = () => {
  sessionStore.showHistorySidebar = false;
  authStore.handleLogout();
};

const formatRelativeTime = (value: string) => {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return 'Just now';
  const diffMinutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60000));
  if (diffMinutes < 1) return 'Just now';
  if (diffMinutes < 60) return diffMinutes + ' minutes ago';
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return diffHours + ' hours ago';
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return diffDays + ' days ago';
  return new Date(value).toLocaleDateString();
};
</script>
