<template>
  <section class="welcome-screen superagent-welcome">
    <div class="welcome-avatar" aria-hidden="true"><BrandLogo size="xl" /></div>
    <span class="welcome-eyebrow">{{ copy.eyebrow }}</span>
    <h2>{{ copy.heading }}</h2>
    <p>
      <span class="welcome-description-desktop">{{ copy.description }}</span>
      <span class="welcome-description-mobile">{{ language === 'ar' ? 'معرفة مدرستك، جاهزة عندما تحتاجها.' : 'Your school knowledge, ready when you need it.' }}</span>
    </p>

    <section class="suggested-questions" aria-labelledby="question-starters-title">
      <div class="suggested-questions-heading">
        <div>
          <h3 id="question-starters-title">{{ copy.questionsTitle }}</h3>
          <p>{{ copy.questionsHint }}</p>
        </div>
      </div>
      <div class="question-grid">
        <button v-for="question in questions" :key="question.text" type="button" class="question-card" @click="$emit('use-question', question.text)">
          <i :class="question.icon" aria-hidden="true"></i>
          <span>{{ question.text }}</span>
          <i class="fa-solid fa-arrow-up-right-from-square question-arrow" aria-hidden="true"></i>
        </button>
      </div>
    </section>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import BrandLogo from '@/components/BrandLogo.vue';

const props = defineProps<{ language: 'en' | 'ar' }>();
defineEmits<{ (e: 'use-question', question: string): void }>();

const content = {
  en: {
    eyebrow: 'AUREXIS IS READY',
    heading: "Hi, I'm Aurexis.",
    description: 'Aurexis School Assistant searches your school knowledge base while answering, shows its working, and links key conclusions back to the original evidence.',
    questionsTitle: 'Popular parent questions', questionsHint: 'Choose one to get started',
    questions: [
      { icon: 'fa-regular fa-calendar-days', text: 'What events are coming up this week?' },
      { icon: 'fa-solid fa-clipboard-check', text: 'How can I check my child’s attendance?' },
      { icon: 'fa-solid fa-graduation-cap', text: 'When will progress reports be available?' },
      { icon: 'fa-regular fa-circle-question', text: 'Who should I contact for school support?' },
    ],
  },
  ar: {
    eyebrow: 'أوريكسيس جاهز',
    heading: 'مرحبًا، أنا أوريكسيس.',
    description: 'يبحث مساعد أوريكسيس المدرسي في قاعدة معارف مدرستك أثناء الإجابة، ويعرض طريقة عمله، ويربط الاستنتاجات الرئيسية بالأدلة الأصلية.',
    questionsTitle: 'أسئلة أولياء الأمور الشائعة', questionsHint: 'اختر سؤالًا للبدء',
    questions: [
      { icon: 'fa-regular fa-calendar-days', text: 'ما الفعاليات القادمة هذا الأسبوع؟' },
      { icon: 'fa-solid fa-clipboard-check', text: 'كيف أتحقق من حضور طفلي؟' },
      { icon: 'fa-solid fa-graduation-cap', text: 'متى ستكون تقارير التقدم متاحة؟' },
      { icon: 'fa-regular fa-circle-question', text: 'بمن أتواصل للحصول على دعم المدرسة؟' },
    ],
  },
} as const;

const copy = computed(() => content[props.language]);
const questions = computed(() => copy.value.questions);
</script>
