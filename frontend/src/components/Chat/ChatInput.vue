<template>
  <div class="input-area-wrapper">
    <div v-if="chatStore.currentPendingHitl" class="hitl-panel">
      <div class="hitl-panel-header">
        <span class="hitl-icon"><i class="fa-solid fa-circle-question"></i></span>
        <span>
          <strong>Just need a bit more from you</strong>
          <small>Agent will continue the original search based on your choice</small>
        </span>
      </div>
      <div class="hitl-panel-prompt">{{ chatStore.currentPendingHitl.prompt }}</div>
      <div
        v-if="chatStore.currentPendingHitl.options && chatStore.currentPendingHitl.options.length"
        class="hitl-options"
      >
        <button
          v-for="option in chatStore.currentPendingHitl.options"
          :key="option"
          type="button"
          class="hitl-option"
          @click="selectHitlOption(option)"
        >
          {{ option }}
        </button>
      </div>
    </div>

    <div :class="['input-area', { 'hitl-active': chatStore.currentPendingHitl }]">
      <button
        class="attach-btn"
        type="button"
        title="Attachments aren't supported yet"
        aria-label="Chat attachments unavailable"
        disabled
      >
        <i class="fa-solid fa-paperclip"></i>
      </button>

      <textarea
        ref="textareaRef"
        v-model="chatStore.userInput"
        class="chat-input-textarea"
        :placeholder="chatStore.inputPlaceholder"
        :disabled="chatStore.isInputLocked"
        rows="1"
        @keydown="handleKeyDown"
        @compositionstart="handleCompositionStart"
        @compositionend="handleCompositionEnd"
        @input="autoResize"
      ></textarea>

      <button
        v-if="chatStore.isViewingStreamingSession"
        type="button"
        class="send-btn stop-btn"
        title="Stop response"
        aria-label="Stop response"
        @click="chatStore.handleStop"
      >
        <i class="fa-solid fa-stop"></i>
      </button>

      <button
        v-else
        type="button"
        class="send-btn"
        :disabled="chatStore.isLoading"
        :title="chatStore.isLoading ? 'A response is already being generated' : 'Send'"
        aria-label="Send message"
        @click="onSend"
      >
        <i class="fa-regular fa-paper-plane"></i>
      </button>
    </div>

    <div class="input-footer">
      <span>AI-generated content may contain errors — verify important conclusions against the cited sources.</span>
      <span><kbd>Enter</kbd> to send · <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref } from 'vue';
import { useChatStore } from '@/stores/chat';

const chatStore = useChatStore();
const textareaRef = ref<HTMLTextAreaElement | null>(null);
const isComposing = ref(false);

const handleCompositionStart = () => {
  isComposing.value = true;
};

const handleCompositionEnd = () => {
  isComposing.value = false;
};

const handleKeyDown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && !event.shiftKey && !isComposing.value) {
    event.preventDefault();
    onSend();
  }
};

const autoResize = () => {
  if (!textareaRef.value) return;
  textareaRef.value.style.height = 'auto';
  textareaRef.value.style.height = Math.min(textareaRef.value.scrollHeight, 140) + 'px';
};

const resetTextareaHeight = () => {
  if (textareaRef.value) textareaRef.value.style.height = 'auto';
};

const focusTextarea = async () => {
  await nextTick();
  textareaRef.value?.focus();
  autoResize();
};

const selectHitlOption = async (option: string) => {
  chatStore.selectHitlOption(option);
  await focusTextarea();
};

const onSend = async () => {
  const text = chatStore.userInput.trim();
  if (!text || chatStore.isLoading || isComposing.value) return;
  await chatStore.handleSend();
  await nextTick();
  resetTextareaHeight();
};
</script>
