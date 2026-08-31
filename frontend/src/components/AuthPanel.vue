<template>
  <section class="auth-page">
    <div class="auth-showcase">
      <div class="auth-showcase-badge">
        <i class="fa-solid fa-sparkles"></i>
        <span>SuperAgent Knowledge Copilot</span>
      </div>
      <h2>Every piece of knowledge<br />deserves a clear echo.</h2>
      <p>
        Hybrid retrieval, parallel agents, evidence reranking, and traceable
        citations — now all in one workspace.
      </p>
      <div class="auth-feature-list">
        <div>
          <i class="fa-solid fa-magnifying-glass-chart"></i>
          <span><strong>Hybrid RAG</strong><small>Dense + BM25 + Rerank</small></span>
        </div>
        <div>
          <i class="fa-solid fa-diagram-project"></i>
          <span><strong>Parallel Agents</strong><small>Automatically breaks down and synthesizes complex questions</small></span>
        </div>
        <div>
          <i class="fa-regular fa-file-lines"></i>
          <span><strong>Trusted Citations</strong><small>Every answer aligned to its source evidence</small></span>
        </div>
      </div>
    </div>

    <div class="auth-panel">
      <div class="auth-panel-heading">
        <span class="auth-mini-logo"><i class="fa-solid fa-robot"></i></span>
        <div>
          <span class="auth-eyebrow">{{ audience === 'parent' ? 'أهلًا بك' : 'Welcome back' }}</span>
          <h1>{{ audience === 'parent' ? 'الدخول لأولياء الأمور' : 'Log in to Agent Assistant' }}</h1>
        </div>
      </div>

      <!-- Parents first, and by default. They are the many; staff are the few, and the
           few already know where their password box is. -->
      <div class="auth-audience" role="tablist">
        <button
          type="button"
          role="tab"
          :aria-selected="audience === 'parent'"
          :class="{ active: audience === 'parent' }"
          @click="showParent"
        >
          <i class="fa-solid fa-user-group"></i> ولي أمر
        </button>
        <button
          type="button"
          role="tab"
          :aria-selected="audience === 'staff'"
          :class="{ active: audience === 'staff' }"
          @click="showStaff"
        >
          <i class="fa-solid fa-briefcase"></i> Staff
        </button>
      </div>

      <WhatsAppLogin v-if="audience === 'parent'" />

      <template v-else>
      <p class="auth-description">
        Enter your private knowledge space and pick up where you left off.
      </p>

      <form class="auth-form" @submit.prevent="onSubmit">
        <label class="form-field">
          <span>Username</span>
          <span class="field-input">
            <i class="fa-regular fa-user"></i>
            <input v-model="authStore.authForm.username" type="text" autocomplete="username" placeholder="Enter your username" />
          </span>
        </label>

        <label class="form-field">
          <span>Password</span>
          <span class="field-input">
            <i class="fa-solid fa-lock"></i>
            <input
              v-model="authStore.authForm.password"
              type="password"
              autocomplete="current-password"
              placeholder="Enter your password"
            />
          </span>
        </label>

        <button class="auth-submit" type="submit" :disabled="authStore.authLoading">
          <span>{{ authStore.authLoading ? 'Connecting...' : 'Enter workspace' }}</span>
          <i :class="authStore.authLoading ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-arrow-right'"></i>
        </button>
      </form>

      <!-- No sign-up link, because there is no sign-up. Staff accounts are created by an
           administrator through the identity service; saying so is better than leaving
           somebody hunting for a button that used to be here. -->
      <p class="auth-footnote">Staff accounts are created by an administrator.</p>
      </template>

      <p class="auth-footnote">By logging in, you acknowledge that AI output should go through the necessary human review.</p>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import WhatsAppLogin from '@/components/WhatsAppLogin.vue';
import { useAuthStore } from '@/stores/auth';

const authStore = useAuthStore();

/**
 * Which door this visitor came to.
 *
 * Parents by default: they outnumber staff by a few hundred to one, they have no password
 * and never will, and a screen that opens on a password box tells them they are in the
 * wrong place. Staff are one tap away and already know what they are looking for.
 */
const audience = ref<'parent' | 'staff'>('parent');

const showParent = () => {
  audience.value = 'parent';
};

const showStaff = () => {
  // Abandon any half-finished verification, so its poller does not keep running behind a
  // panel nobody is looking at.
  authStore.resetWhatsApp();
  audience.value = 'staff';
};

const onSubmit = async () => {
  try {
    await authStore.handleAuthSubmit();
  } catch (error: any) {
    alert(error.message);
  }
};
</script>

<style scoped>
.auth-audience {
  display: flex;
  gap: 0.5rem;
  margin: 0.75rem 0 1rem;
}

.auth-audience button {
  flex: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  padding: 0.55rem 0.75rem;
  border-radius: 0.6rem;
  border: 1px solid rgba(127, 127, 127, 0.35);
  background: transparent;
  font: inherit;
  cursor: pointer;
  opacity: 0.7;
}

.auth-audience button.active {
  opacity: 1;
  border-color: currentColor;
  background: rgba(127, 127, 127, 0.12);
  font-weight: 600;
}
</style>
