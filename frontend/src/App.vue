<template>
  <AuthPanel
    v-if="!authStore.isAuthenticated"
    :theme="theme"
    @toggle-theme="toggleTheme"
  />

  <div v-else class="app-page" :class="{ 'desktop-sidebar-collapsed': desktopSidebarCollapsed }">
    <div class="app-wrapper">
      <div
        class="mobile-sidebar-shell"
        :class="{ 'is-open': mobileSidebarOpen }"
      >
        <Sidebar :theme="theme" @toggle-theme="toggleTheme" />
      </div>

      <button
        v-if="mobileSidebarOpen"
        class="mobile-sidebar-backdrop"
        type="button"
        aria-label="Close navigation"
        @click="closeMobileSidebar"
      ></button>
      <button
        class="mobile-menu-button"
        type="button"
        aria-label="Open navigation"
        :aria-expanded="mobileSidebarOpen"
        @click="openMobileSidebar"
      >
        <span class="ax-sidebar-glyph" aria-hidden="true"></span>
      </button>

<button
        class="desktop-sidebar-toggle"
        type="button"
        :aria-expanded="!desktopSidebarCollapsed"
        :aria-label="desktopSidebarCollapsed ? 'Open sidebar' : 'Close sidebar'"
        @click="toggleDesktopSidebar"
      >
        <span class="ax-sidebar-glyph" aria-hidden="true"></span>
      </button>

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

const mobileSidebarOpen = ref(false);

const openMobileSidebar = () => {
  mobileSidebarOpen.value = true;
};

const closeMobileSidebar = () => {
  mobileSidebarOpen.value = false;
};

const desktopSidebarCollapsed = ref(
  localStorage.getItem('aurexis-desktop-sidebar-collapsed') === '1'
);

const toggleDesktopSidebar = () => {
  desktopSidebarCollapsed.value = !desktopSidebarCollapsed.value;
  localStorage.setItem(
    'aurexis-desktop-sidebar-collapsed',
    desktopSidebarCollapsed.value ? '1' : '0'
  );
};

// AUREXIS_HISTORY_MOBILE_DRAWER_SYNC_V11
watch(
  () => sessionStore.showHistorySidebar,
  (isOpen) => {
    if (isOpen) {
      mobileSidebarOpen.value = false;
    }
  }
);

// AUREXIS_KNOWLEDGE_MOBILE_DRAWER_SYNC_V11_1
watch(
  () => chatStore.activeNav,
  (activeNav) => {
    if (activeNav === 'settings') {
      mobileSidebarOpen.value = false;
    }
  }
);

// AUREXIS_LOGO_TOGGLE_V11_10
const handleAurexisDesktopLogoToggle = (event: MouseEvent) => {
  if (window.innerWidth < 900) return;

  const target = event.target as HTMLElement | null;
  if (!target) return;

  const header = target.closest('.sidebar .sidebar-header');
  if (!header) return;

  event.preventDefault();
  event.stopPropagation();

  toggleDesktopSidebar();
};

onMounted(() => {
  document.addEventListener(
    'click',
    handleAurexisDesktopLogoToggle,
    true
  );
});

onUnmounted(() => {
  document.removeEventListener(
    'click',
    handleAurexisDesktopLogoToggle,
    true
  );
});
</script>
