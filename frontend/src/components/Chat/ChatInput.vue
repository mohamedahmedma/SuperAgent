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

    <div :class="['input-area', { 'hitl-active': chatStore.currentPendingHitl, 'is-recording': isRecording, 'is-paused': isPaused }]">
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
        <span class="voice-hint">{{ recordingLimitReached ? '1 minute reached — send or delete' : isPaused ? 'Recording paused' : 'Recording voice message' }}</span>
        <button v-if="!recordingLimitReached" type="button" class="voice-pause" :aria-label="isPaused ? 'Resume recording' : 'Pause recording'" @click="toggleRecordingPause"><i :class="isPaused ? 'fa-solid fa-play' : 'fa-solid fa-pause'"></i></button>
        <button type="button" class="voice-cancel" @click="cancelRecording"><i class="fa-solid fa-trash"></i></button>
        <button type="button" class="voice-send" aria-label="Finish and send recording" @click="stopRecording(false, true)"><i class="fa-solid fa-paper-plane"></i></button>
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
const isPaused = ref(false);
const recordingLimitReached = ref(false);
const locked = ref(false);
const startPoint = ref<{ x: number; y: number } | null>(null);
let timer: number | undefined;
const recordingTime = computed(() => `00:${String(elapsed.value).padStart(2, '0')}`);
const voiceAttachmentLabel = '[Voice recording attached]';

const stopRecording = (discard = false, sendImmediately = false) => {
  window.clearInterval(timer); isRecording.value = false; isPaused.value = false; recordingLimitReached.value = false;
  const active = recorder.value;
  if (!active || active.state === 'inactive') return;
  active.ondataavailable = (event) => { if (!discard && event.data.size) audioChunks.value.push(event.data); };
  active.onstop = async () => {
    active.stream.getTracks().forEach((track) => track.stop());
    if (!discard && audioChunks.value.length) {
      const blob = new Blob(audioChunks.value, { type: active.mimeType || 'audio/webm' });
      chatStore.queueVoiceMessage({
        url: URL.createObjectURL(blob),
        duration: elapsed.value,
        mimeType: blob.type,
      });
      chatStore.userInput += (chatStore.userInput ? ' ' : '') + voiceAttachmentLabel;
      if (sendImmediately) await onSend();
    }
    audioChunks.value = [];
    recorder.value = null;
  };
  active.stop();
};
const toggleRecording = async (event: PointerEvent) => {
  if (isRecording.value) return stopRecording();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder.value = new MediaRecorder(stream); audioChunks.value = []; elapsed.value = 0; isPaused.value = false; recordingLimitReached.value = false; startedAt.value = Date.now(); startPoint.value = { x: event.clientX, y: event.clientY }; locked.value = false; isRecording.value = true; recorder.value.start();
    timer = window.setInterval(() => {
      if (!isPaused.value) elapsed.value = Math.min(60, elapsed.value + 1);
      if (elapsed.value >= 60) {
        const active = recorder.value;
        if (active?.state === 'recording') active.pause();
        isPaused.value = true;
        recordingLimitReached.value = true;
        window.clearInterval(timer);
      }
    }, 1000);
  } catch { alert('Please allow microphone access to record a voice message.'); }
};
const toggleRecordingPause = () => {
  const active = recorder.value;
  if (!active || recordingLimitReached.value) return;
  if (active.state === 'recording') { active.pause(); isPaused.value = true; }
  else if (active.state === 'paused') { active.resume(); isPaused.value = false; }
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
