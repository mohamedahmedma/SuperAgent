<template>
  <div :class="['chat-workspace', { 'advanced-mode': showAdvanced }]">
    <section class="chat-area">
      <header class="chat-header">
        <div class="header-info">
          <h1>{{ sessionTitle }}</h1>
        </div>
        <button
          v-if="chatStore.messages.length === 0"
          type="button"
          class="services-launcher"
          :aria-expanded="servicesOpen"
          aria-controls="school-services-popover"
          :aria-label="language === 'ar' ? 'فتح الخدمات السريعة' : 'Open quick services'"
          @click="servicesOpen = !servicesOpen"
        >
          <i class="fa-solid fa-bolt" aria-hidden="true"></i>
          <span>{{ language === 'ar' ? 'خدمات سريعة' : 'Quick services' }}</span>
        </button>
      </header>

      <div :class="['chat-container', { 'has-messages': chatStore.messages.length > 0 }]" ref="chatContainerRef" @scroll.passive="onScroll">
        <div v-if="chatStore.messages.length === 0" class="parent-dashboard">
          <WelcomeScreen :language="language" @use-question="useSuggestedQuestion" />

          <Transition name="services-pop">
          <aside v-if="servicesOpen" id="school-services-popover" class="parent-services" aria-label="School services">
            <div class="parent-services-hero">
              <div class="robot-frame" aria-hidden="true"><AurexisRobot /></div>
              <h2>{{ ui.assistantTitle }}</h2>
              <p>{{ ui.assistantDescription }}</p>
            </div>
            <div class="parent-services-list">
              <h3>{{ ui.servicesTitle }}</h3>
              <button
                v-for="service in localizedServices"
                :key="service.label"
                type="button"
                class="quick-service"
                @click="useSuggestedQuestion(service.question)"
              >
                <span class="quick-service-icon"><i :class="service.icon" aria-hidden="true"></i></span>
                <span>{{ service.label }}</span>
                <i class="fa-solid fa-chevron-right" aria-hidden="true"></i>
              </button>
            </div>
            <div class="parent-tip"><i class="fa-regular fa-lightbulb" aria-hidden="true"></i><span><strong>Helpful tip</strong> Ask in your own words—we’ll guide you.</span></div>
          </aside>
          </Transition>
        </div>

        <div v-if="chatStore.isLoadingOlderMessages" class="older-messages-status">
          <i class="fa-solid fa-spinner fa-spin"></i>
          <span>Loading earlier messages…</span>
        </div>

        <div
          v-else-if="chatStore.messages.length && !chatStore.canLoadOlderMessages && hasPaged"
          class="older-messages-status older-messages-start"
        >
          <span>Beginning of this conversation</span>
        </div>

        <MessageItem
          v-for="(msg, index) in chatStore.messages"
          :key="index"
          :msg="msg"
          :msg-index="index"
          :show-advanced="showAdvanced"
          :ref="(el) => { if (el) messageItemRefs[index] = el; }"
          @cite-click="scrollToChunk"
        />
      </div>

      <ChatInput :language="language" />
    </section>

    <KnowledgeContextPanel v-if="showAdvanced" @cite-click="scrollToChunk" />
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUpdate, onMounted, ref, watch } from 'vue';
import WelcomeScreen from './WelcomeScreen.vue';
import MessageItem from './MessageItem.vue';
import ChatInput from './ChatInput.vue';
import KnowledgeContextPanel from './KnowledgeContextPanel.vue';
import AurexisRobot from '@/components/AurexisRobot.vue';
import { useChatStore } from '@/stores/chat';
import { useSessionStore } from '@/stores/sessions';

const chatStore = useChatStore();
const sessionStore = useSessionStore();
const props = defineProps<{ language: 'en' | 'ar' }>();
const showAdvanced = computed(() => false);

const localizedContent = {
  en: {
    assistantTitle: 'Your school assistant', assistantDescription: 'Answers and support, whenever you need them.', servicesTitle: 'Quick services',
    services: [
      { icon: 'fa-solid fa-chart-column', label: 'Student progress', question: 'How can I check my child’s progress?' },
      { icon: 'fa-regular fa-calendar-check', label: 'Attendance', question: 'How can I check my child’s attendance?' },
      { icon: 'fa-solid fa-book-open', label: 'School timetable', question: 'Where can I find the school timetable?' },
      { icon: 'fa-regular fa-credit-card', label: 'School fees', question: 'How can I find information about school fees?' },
      { icon: 'fa-solid fa-people-group', label: 'Contact the school', question: 'Who should I contact for school support?' },
    ],
  },
  ar: {
    assistantTitle: 'مساعد مدرستك', assistantDescription: 'إجابات ودعم وقتما تحتاج إليهما.', servicesTitle: 'خدمات سريعة',
    services: [
      { icon: 'fa-solid fa-chart-column', label: 'تقدم الطالب', question: 'كيف أتحقق من تقدم طفلي؟' },
      { icon: 'fa-regular fa-calendar-check', label: 'الحضور', question: 'كيف أتحقق من حضور طفلي؟' },
      { icon: 'fa-solid fa-book-open', label: 'جدول المدرسة', question: 'أين أجد جدول المدرسة؟' },
      { icon: 'fa-regular fa-credit-card', label: 'رسوم المدرسة', question: 'كيف أجد معلومات عن رسوم المدرسة؟' },
      { icon: 'fa-solid fa-people-group', label: 'التواصل مع المدرسة', question: 'بمن أتواصل للحصول على دعم المدرسة؟' },
    ],
  },
} as const;
const ui = computed(() => localizedContent[props.language]);
const localizedServices = computed(() => ui.value.services);

const quickServices = [
  { icon: 'fa-solid fa-chart-column', label: 'Student progress', question: 'How can I check my child’s progress?' },
  { icon: 'fa-regular fa-calendar-check', label: 'Attendance', question: 'How can I check my child’s attendance?' },
  { icon: 'fa-solid fa-book-open', label: 'School timetable', question: 'Where can I find the school timetable?' },
  { icon: 'fa-regular fa-credit-card', label: 'School fees', question: 'How can I find information about school fees?' },
  { icon: 'fa-solid fa-people-group', label: 'Contact the school', question: 'Who should I contact for school support?' },
];

const chatContainerRef = ref<HTMLDivElement | null>(null);
const messageItemRefs = ref<any[]>([]);
const servicesOpen = ref(false);

const sessionTitle = computed(() => {
  const session = sessionStore.sessions.find((item) => item.session_id === chatStore.sessionId);
  if (session?.title) return session.title;
  const firstUserMessage = chatStore.messages.find((message) => message.isUser && message.text.trim());
  if (!firstUserMessage) return 'New conversation';
  const text = firstUserMessage.text.trim();
  return text.length > 28 ? text.slice(0, 28) + '…' : text;
});

onBeforeUpdate(() => {
  messageItemRefs.value = [];
});

const hasPaged = computed(() => !!chatStore.pagingBySession[chatStore.sessionId]);
const isPrepending = ref(false);
const LOAD_OLDER_THRESHOLD_PX = 120;

const scrollToBottom = () => {
  if (chatContainerRef.value) {
    chatContainerRef.value.scrollTop = chatContainerRef.value.scrollHeight;
  }
};

const useSuggestedQuestion = async (question: string) => {
  chatStore.userInput = question;
  await nextTick();
  document.querySelector<HTMLTextAreaElement>('.chat-input-textarea')?.focus();
};

const onScroll = async () => {
  const container = chatContainerRef.value;
  if (!container || isPrepending.value) return;
  if (container.scrollTop > LOAD_OLDER_THRESHOLD_PX) return;
  if (!chatStore.canLoadOlderMessages) return;

  const sessionId = chatStore.sessionId;
  const previousBottomOffset = container.scrollHeight - container.scrollTop;
  isPrepending.value = true;

  try {
    await chatStore.loadOlderMessages(sessionId);
    await nextTick();
    if (chatContainerRef.value && chatStore.sessionId === sessionId) {
      chatContainerRef.value.scrollTop =
        chatContainerRef.value.scrollHeight - previousBottomOffset;
    }
  } catch (error: any) {
    console.warn('Could not load earlier messages:', error?.message || error);
  } finally {
    await nextTick();
    isPrepending.value = false;
  }
};

const scrollToChunk = async (msgIndex: number, chunkIndex: number) => {
  const msgItem = messageItemRefs.value[msgIndex];
  if (!msgItem) return;

  msgItem.openReferences();
  await nextTick();

  const chunkEl = document.getElementById('chunk-' + msgIndex + '-' + chunkIndex);
  if (chunkEl) {
    chunkEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    chunkEl.classList.add('highlight-chunk');
    window.setTimeout(() => chunkEl.classList.remove('highlight-chunk'), 2000);
  }
};

watch(
  () => chatStore.messages,
  () => {
    if (isPrepending.value) return;
    nextTick(scrollToBottom);
  },
  { deep: true }
);

watch(
  () => chatStore.sessionId,
  () => nextTick(scrollToBottom)
);

onMounted(scrollToBottom);
</script>

<style scoped>
.older-messages-status {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 0 14px;
  font-size: 0.78rem;
  color: var(--text-muted, #8a8a8a);
}
.older-messages-start { opacity: 0.7; }
</style>
