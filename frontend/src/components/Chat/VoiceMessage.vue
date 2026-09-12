<template>
  <section class="voice-message" :class="{ 'is-playing': isPlaying }" aria-label="Voice message">
    <button
      class="voice-message-play"
      type="button"
      :aria-label="isPlaying ? 'Pause voice message' : 'Play voice message'"
      @click="togglePlayback"
    >
      <i :class="isPlaying ? 'fa-solid fa-pause' : 'fa-solid fa-play'" aria-hidden="true"></i>
    </button>

    <div class="voice-message-track">
      <button class="voice-message-wave" type="button" aria-label="Seek within voice message" @click="seekAtPosition">
        <span v-for="bar in 24" :key="bar" :class="{ 'is-heard': bar / 24 <= progress }"></span>
      </button>
      <div class="voice-message-meta">
        <span class="voice-message-label"><i class="fa-solid fa-microphone" aria-hidden="true"></i> Voice message</span>
        <time>{{ formatTime(isPlaying ? currentTime : safeDuration) }}</time>
      </div>
    </div>

    <button class="voice-message-speed" type="button" :aria-label="`Playback speed ${playbackRate} times`" @click="cycleSpeed">×{{ playbackRate }}</button>

    <audio ref="audioRef" :src="voice.url" preload="metadata" @loadedmetadata="syncDuration" @timeupdate="syncTime" @ended="handleEnded"></audio>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue';
import type { VoiceMessage } from '@/types/chat';

const props = defineProps<{ voice: VoiceMessage }>();
const audioRef = ref<HTMLAudioElement | null>(null);
const isPlaying = ref(false);
const currentTime = ref(0);
const mediaDuration = ref(0);
const playbackRate = ref(1);

const safeDuration = computed(() => Math.max(mediaDuration.value || props.voice.duration || 0, 0.1));

const formatTime = (seconds: number) => {
  const wholeSeconds = Math.max(0, Math.floor(seconds));
  return `${Math.floor(wholeSeconds / 60)}:${String(wholeSeconds % 60).padStart(2, '0')}`;
};

const togglePlayback = async () => {
  const audio = audioRef.value;
  if (!audio) return;
  if (audio.paused) {
    await audio.play();
    isPlaying.value = true;
  } else {
    audio.pause();
    isPlaying.value = false;
  }
};

const progress = computed(() => Math.min(currentTime.value / safeDuration.value, 1));

const seekAtPosition = (event: MouseEvent) => {
  const audio = audioRef.value;
  if (!audio) return;
  const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
  audio.currentTime = ((event.clientX - rect.left) / rect.width) * safeDuration.value;
  currentTime.value = audio.currentTime;
};

const cycleSpeed = () => {
  const speeds = [1, 1.5, 2];
  playbackRate.value = speeds[(speeds.indexOf(playbackRate.value) + 1) % speeds.length];
  if (audioRef.value) audioRef.value.playbackRate = playbackRate.value;
};

const syncDuration = () => {
  mediaDuration.value = audioRef.value?.duration || 0;
  if (audioRef.value) audioRef.value.playbackRate = playbackRate.value;
};
const syncTime = () => { currentTime.value = audioRef.value?.currentTime || 0; };
const handleEnded = () => { isPlaying.value = false; currentTime.value = 0; };

onBeforeUnmount(() => audioRef.value?.pause());
</script>
