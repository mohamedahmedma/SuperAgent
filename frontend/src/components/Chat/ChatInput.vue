<template>
  <div class="input-area-wrapper">
    <div v-if="chatStore.currentPendingHitl" class="hitl-panel">
      <div class="hitl-panel-header">
        <span class="hitl-icon"><i class="fa-solid fa-circle-question"></i></span>
        <span>
          <strong>Just need a bit more from you</strong>
          <small>Aurexis will continue the original search based on your choice</small>
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

    <div :class="['input-area', { 'hitl-active': chatStore.currentPendingHitl, 'is-recording': isRecording }]">
      <button
        class="attach-btn"
        type="button"
        title="Attachments aren't supported yet"
        aria-label="Chat attachments unavailable"
        disabled
      >
        <i class="fa-solid fa-paperclip"></i>
      </button>

      <button v-if="!isRecording" class="voice-btn" type="button"
        :title="isRecording ? 'Stop recording' : 'Record voice message'" :aria-label="isRecording ? 'Stop recording' : 'Record voice message'"
        @click="toggleRecording">
        <i class="fa-solid fa-microphone"></i>
      </button>

      <div v-if="isRecording" class="voice-recording" @pointermove="trackGesture" @pointerup="finishGesture">
        <button type="button" class="voice-cancel" aria-label="Cancel recording" @click="cancelRecording"><i class="fa-solid fa-trash"></i></button>
        <span class="record-dot"></span><strong>{{ recordingTime }}</strong>
        <span class="voice-hint">← Slide to cancel · Slide up to lock</span>
        <button type="button" class="voice-cancel" @click="cancelRecording"><i class="fa-solid fa-trash"></i></button>
        <button type="button" class="voice-send" aria-label="Finish and send recording" @click="stopRecording()"><i class="fa-solid fa-paper-plane"></i></button>
      </div>

      <textarea v-if="!isRecording" data-gramm="false" data-gramm_editor="false" spellcheck="false"
        ref="textareaRef"
        v-model="chatStore.userInput"
        class="chat-input-textarea" :placeholder="language === 'ar' ? 'اكتب رسالتك إلى أوركسيس...' : 'Say something to Aurexis...'"
        :disabled="chatStore.isInputLocked"
        rows="1"
        @keydown="handleKeyDown"
        @compositionstart="handleCompositionStart"
        @compositionend="handleCompositionEnd"
        @input="autoResize"
      ></textarea>

      <button
        v-if="!isRecording && chatStore.isViewingStreamingSession"
        type="button"
        class="send-btn stop-btn"
        title="Stop response"
        aria-label="Stop response"
        @click="chatStore.handleStop"
      >
        <i class="fa-solid fa-stop"></i>
      </button>

      <button
        v-else-if="!isRecording"
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
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref } from 'vue';
import { useChatStore } from '@/stores/chat';

defineProps<{ language: 'en' | 'ar' }>();
const chatStore = useChatStore();
const textareaRef = ref<HTMLTextAreaElement | null>(null);
const isComposing = ref(false);
const recorder = ref<MediaRecorder | null>(null);
const audioChunks = ref<Blob[]>([]);
const isRecording = ref(false);
const startedAt = ref(0);
const elapsed = ref(0);
const locked = ref(false);
const startPoint = ref<{ x: number; y: number } | null>(null);
let timer: number | undefined;
const recordingTime = computed(() => `00:${String(elapsed.value).padStart(2, '0')}`);
const stopRecording = (discard = false) => {
  window.clearInterval(timer); isRecording.value = false;
  const active = recorder.value;
  if (!active || active.state === 'inactive') return;
  active.ondataavailable = (event) => { if (!discard && event.data.size) audioChunks.value.push(event.data); };
  active.onstop = () => { if (!discard && audioChunks.value.length) chatStore.userInput += (chatStore.userInput ? ' ' : '') + '[Voice recording attached]'; audioChunks.value = []; };
  active.stop();
};
const toggleRecording = async (event: PointerEvent) => {
  if (isRecording.value) return stopRecording();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder.value = new MediaRecorder(stream); audioChunks.value = []; elapsed.value = 0; startedAt.value = Date.now(); startPoint.value = { x: event.clientX, y: event.clientY }; locked.value = false; isRecording.value = true; recorder.value.start();
    timer = window.setInterval(() => { elapsed.value = Math.min(60, Math.floor((Date.now() - startedAt.value) / 1000)); if (elapsed.value >= 60) stopRecording(); }, 250);
  } catch { alert('Please allow microphone access to record a voice message.'); }
};
const trackGesture = (event: PointerEvent) => { if (!startPoint.value || locked.value) return; if (startPoint.value.y - event.clientY > 60) locked.value = true; };
const finishGesture = (event: PointerEvent) => { if (!startPoint.value || locked.value) return; if (startPoint.value.x - event.clientX > 100) cancelRecording(); };
const cancelRecording = () => stopRecording(true);
onBeforeUnmount(() => stopRecording(true));

const handleCompositionStart = () => { isComposing.value = true; };
const handleCompositionEnd = () => { isComposing.value = false; };

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
