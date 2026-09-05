<template>
  <AuthPanel
    v-if="!authStore.isAuthenticated"
    :theme="theme"
    @toggle-theme="toggleTheme"
  />

  <div v-else class="app-page">
    <div class="app-wrapper">
      <Sidebar :theme="theme" @toggle-theme="toggleTheme" />
      <main class="main-content">
        <DocumentSettings v-if="chatStore.activeNav === 'settings'" />
        <HistorySidebar />
        <ChatArea v-show="chatStore.activeNav !== 'settings'" />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue';
import Sidebar from '@/components/Sidebar.vue';
import AuthPanel from '@/components/AuthPanel.vue';
import HistorySidebar from '@/components/HistorySidebar.vue';
import ChatArea from '@/components/Chat/ChatArea.vue';
import DocumentSettings from '@/components/Documents/DocumentSettings.vue';
import { useAuthStore } from '@/stores/auth';
import { useChatStore } from '@/stores/chat';
import { useSessionStore } from '@/stores/sessions';

const authStore = useAuthStore();
const chatStore = useChatStore();
const sessionStore = useSessionStore();

type Theme = 'dark' | 'light';
const storedTheme = localStorage.getItem('superagent-theme');
const theme = ref<Theme>(storedTheme === 'light' ? 'light' : 'dark');

const applyTheme = (nextTheme: Theme) => {
  document.documentElement.dataset.theme = nextTheme;
  document.documentElement.style.colorScheme = nextTheme;
  localStorage.setItem('superagent-theme', nextTheme);
};

const toggleTheme = () => {
  const nextTheme: Theme = theme.value === 'dark' ? 'light' : 'dark';
  const startViewTransition = (document as any).startViewTransition?.bind(document);
  if (!startViewTransition || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    theme.value = nextTheme;
    return;
  }
  document.documentElement.classList.add('aurexis-theme-transition');
  const transition = startViewTransition(() => { theme.value = nextTheme; });
  transition.finished.finally(() => {
    document.documentElement.classList.remove('aurexis-theme-transition');
  });
};

watch(theme, applyTheme, { immediate: true });

watch(
  () => authStore.currentUser?.username || null,
  (username, previousUsername) => {
    if (username === previousUsername) return;
    chatStore.resetWorkspace();
    sessionStore.$reset();
  }
);

const handleUnauthorized = () => {
  authStore.handleLogout();
  alert('Your session has expired, please log in again');
};

onMounted(async () => {
  window.addEventListener('unauthorized', handleUnauthorized);
  if (authStore.token) {
    try { await authStore.fetchMe(); }
    catch (_) { authStore.handleLogout(); }
  }
});

onUnmounted(() => window.removeEventListener('unauthorized', handleUnauthorized));
</script>
