<template>
  <section class="auth-page">
    <div class="auth-showcase">
      <div class="auth-showcase-badge">
        <i class="fa-solid fa-sparkles"></i>
        <span>SuperMew Knowledge Copilot</span>
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
        <span class="auth-mini-logo"><i class="fa-solid fa-cat"></i></span>
        <div>
          <span class="auth-eyebrow">{{ authStore.authMode === 'login' ? 'Welcome back' : 'Create account' }}</span>
          <h1>{{ authStore.authMode === 'login' ? 'Log in to Mew Assistant' : 'Sign up for Mew Assistant' }}</h1>
        </div>
      </div>
      <p class="auth-description">
        {{ authStore.authMode === 'login'
          ? 'Enter your private knowledge space and pick up where you left off.'
          : 'Create an account to start chatting and save your conversation history.' }}
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
              :autocomplete="authStore.authMode === 'login' ? 'current-password' : 'new-password'"
              placeholder="Enter your password"
            />
          </span>
        </label>

        <label v-if="authStore.authMode === 'register'" class="form-field">
          <span>Account role</span>
          <span class="field-input">
            <i class="fa-regular fa-id-badge"></i>
            <select v-model="authStore.authForm.role">
              <option value="user">Standard user</option>
              <option value="admin">Administrator</option>
            </select>
          </span>
        </label>

        <label
          v-if="authStore.authMode === 'register' && authStore.authForm.role === 'admin'"
          class="form-field"
        >
          <span>Admin invite code</span>
          <span class="field-input">
            <i class="fa-solid fa-key"></i>
            <input
              v-model="authStore.authForm.admin_code"
              type="password"
              autocomplete="off"
              placeholder="Enter the admin invite code"
            />
          </span>
        </label>

        <button class="auth-submit" type="submit" :disabled="authStore.authLoading">
          <span>{{ authStore.authLoading ? 'Connecting...' : (authStore.authMode === 'login' ? 'Enter workspace' : 'Create account') }}</span>
          <i :class="authStore.authLoading ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-arrow-right'"></i>
        </button>
      </form>

      <div class="auth-divider"><span>or</span></div>
      <button class="auth-switch" type="button" @click="toggleAuthMode">
        {{ authStore.authMode === 'login' ? "Don't have an account? Create one" : 'Already have an account? Back to login' }}
      </button>
      <p class="auth-footnote">By logging in, you acknowledge that AI output should go through the necessary human review.</p>
    </div>
  </section>
</template>

<script setup lang="ts">
import { useAuthStore } from '@/stores/auth';

const authStore = useAuthStore();

const toggleAuthMode = () => {
  authStore.authMode = authStore.authMode === 'login' ? 'register' : 'login';
};

const onSubmit = async () => {
  try {
    await authStore.handleAuthSubmit();
  } catch (error: any) {
    alert(error.message);
  }
};
</script>
