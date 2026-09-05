<template>
  <button
    ref="buttonRef"
    class="theme-toggle"
    type="button"
    role="switch"
    :aria-checked="theme === 'light'"
    :aria-label="theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode'"
    @click="handleToggle"
  >
    <i class="fa-regular fa-sun theme-toggle-icon theme-toggle-sun" aria-hidden="true"></i>
    <i class="fa-regular fa-moon theme-toggle-icon theme-toggle-moon" aria-hidden="true"></i>
    <span class="theme-toggle-thumb" aria-hidden="true"></span>
  </button>
</template>

<script setup lang="ts">
import { ref } from 'vue';

defineProps<{
  theme: 'dark' | 'light';
}>();

const emit = defineEmits<{
  (e: 'toggle', origin?: { x: number; y: number }): void;
}>();

const buttonRef = ref<HTMLButtonElement | null>(null);

const handleToggle = () => {
  const rect = buttonRef.value?.getBoundingClientRect();

  emit(
    'toggle',
    rect
      ? {
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2,
        }
      : undefined
  );
};
</script>
