<template>
  <div class="wa-login" dir="rtl" lang="ar">
    <!-- 1 · nothing started ------------------------------------------------>
    <template v-if="wa.status === 'idle'">
      <p class="wa-lead">
        سجّل الدخول برقم هاتفك المسجّل لدى المدرسة. لا حاجة لكلمة مرور.
        <span class="wa-lead-en">Sign in with the number the school has on file. No password needed.</span>
      </p>
      <button class="wa-primary" type="button" :disabled="wa.busy" @click="start">
        <i :class="wa.busy ? 'fa-solid fa-spinner fa-spin' : 'fa-brands fa-whatsapp'"></i>
        <span>{{ wa.busy ? 'جارٍ التحضير…' : 'المتابعة عبر واتساب' }}</span>
      </button>
    </template>

    <!-- 2 · waiting for the parent to send --------------------------------->
    <template v-else-if="wa.status === 'waiting'">
      <ol class="wa-steps">
        <li>اضغط الزر بالأسفل — سيفتح واتساب برسالة جاهزة.</li>
        <!-- Said explicitly. WhatsApp never sends a pre-filled message on the user's
             behalf, and a screen that implies otherwise produces a queue of parents who
             tapped and then waited for something that was never going to happen. -->
        <li><strong>اضغط زر الإرسال داخل واتساب</strong> — لن تُرسل الرسالة تلقائيًا.</li>
        <li>سنرد عليك برمز من ٦ أرقام، اكتبه هنا.</li>
      </ol>

      <a class="wa-primary" :href="wa.link" target="_blank" rel="noopener noreferrer">
        <i class="fa-brands fa-whatsapp"></i>
        <span>افتح واتساب وأرسل الرسالة</span>
      </a>

      <p class="wa-waiting"><i class="fa-solid fa-spinner fa-spin"></i> في انتظار رسالتك…</p>

      <!-- The manual fallback, and it is not decoration: in-app browsers (Instagram,
           Facebook, some Android WebViews) routinely swallow the wa.me handoff and strand
           the parent on a WhatsApp Web login page. Without the number and the text visible
           as copyable plain text, that parent has no way through at all. -->
      <details class="wa-fallback">
        <summary>لم يفتح واتساب؟</summary>
        <p>أرسل هذه الرسالة يدويًا إلى الرقم:</p>
        <p class="wa-copyable">{{ wa.businessNumber }}</p>
        <p class="wa-copyable">{{ wa.message }}</p>
      </details>

      <button class="wa-secondary" type="button" @click="cancel">إلغاء</button>
    </template>

    <!-- 3 · the code has been sent ------------------------------------------>
    <template v-else-if="wa.status === 'code_sent'">
      <p class="wa-lead">
        <template v-if="wa.displayName">أهلًا {{ wa.displayName }} —</template>
        أرسلنا رمزًا من ٦ أرقام إلى واتساب.
      </p>

      <form class="wa-code-form" @submit.prevent="submit">
        <label class="form-field">
          <span>رمز التحقق</span>
          <span class="field-input">
            <i class="fa-solid fa-key"></i>
            <input
              ref="codeInput"
              v-model="wa.code"
              type="text"
              inputmode="numeric"
              autocomplete="one-time-code"
              maxlength="6"
              placeholder="••••••"
              dir="ltr"
            />
          </span>
        </label>

        <p v-if="wa.error" class="wa-error" role="alert">{{ errorText }}</p>

        <button class="wa-primary" type="submit" :disabled="wa.busy || wa.code.trim().length < 6">
          <i :class="wa.busy ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-arrow-left'"></i>
          <span>{{ wa.busy ? 'جارٍ التحقق…' : 'تأكيد' }}</span>
        </button>
      </form>

      <button class="wa-secondary" type="button" @click="cancel">البدء من جديد</button>
    </template>

    <!-- 4 · dead ------------------------------------------------------------>
    <template v-else>
      <p class="wa-error" role="alert">{{ errorText }}</p>
      <button class="wa-primary" type="button" @click="start">
        <i class="fa-solid fa-rotate-right"></i>
        <span>المحاولة مرة أخرى</span>
      </button>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue';
import { useAuthStore } from '@/stores/auth';

const authStore = useAuthStore();
const wa = computed(() => authStore.whatsapp);
const codeInput = ref<HTMLInputElement | null>(null);

/**
 * What to actually say, chosen by identity's own refusal code.
 *
 * Keyed on the code rather than the message because the message is prose written for a
 * developer reading a log — it says which rule was broken, not what the parent should do
 * next. Anything unrecognised falls back to the server's wording rather than to a generic
 * apology, so a refusal added later is still readable instead of silently blank.
 */
const MESSAGES: Record<string, string> = {
  bad_code: 'الرمز غير صحيح. تحقّق من الرسالة وحاول مرة أخرى.',
  too_many_attempts: 'محاولات كثيرة غير صحيحة. ابدأ من جديد للحصول على رمز جديد.',
  expired: 'انتهت صلاحية الطلب. ابدأ من جديد.',
  already_used: 'تم استخدام هذا الرمز بالفعل. ابدأ من جديد.',
  not_ready: 'لم نستلم رسالتك بعد. أرسل الرسالة من واتساب أولًا.',
  not_found: 'انتهت هذه الجلسة. ابدأ من جديد.',
  rejected: 'هذا الرقم غير مسجّل لدى المدرسة. تواصل مع إدارة المدرسة لإضافته.',
};

const errorText = computed(() => MESSAGES[wa.value.errorCode] || wa.value.error);

const start = () => authStore.startWhatsAppLogin();
const cancel = () => authStore.resetWhatsApp();
const submit = () => authStore.submitWhatsAppCode();

// Put the cursor in the code box the moment there is a code to type, so a parent coming
// back from WhatsApp can type straight away rather than hunting for the field.
watch(
  () => wa.value.status,
  async (status) => {
    if (status === 'code_sent') {
      await nextTick();
      codeInput.value?.focus();
    }
  }
);

// A poller that outlives its panel keeps hitting identity every two seconds forever.
onBeforeUnmount(() => authStore.resetWhatsApp());
</script>

<style scoped>
.wa-login {
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
  text-align: right;
}

.wa-lead {
  margin: 0;
  line-height: 1.7;
}

.wa-lead-en {
  display: block;
  margin-top: 0.25rem;
  font-size: 0.85em;
  opacity: 0.65;
  direction: ltr;
  text-align: left;
}

.wa-steps {
  margin: 0;
  padding-inline-start: 1.2rem;
  line-height: 1.9;
}

.wa-primary,
.wa-secondary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.55rem;
  width: 100%;
  padding: 0.8rem 1rem;
  border-radius: 0.7rem;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
  text-decoration: none;
  border: 1px solid transparent;
}

.wa-primary {
  background: #25d366; /* WhatsApp green: parents recognise the button before the words. */
  color: #06251a;
}

.wa-primary[disabled] {
  opacity: 0.6;
  cursor: default;
}

.wa-secondary {
  background: transparent;
  border-color: currentColor;
  opacity: 0.7;
}

.wa-waiting {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  opacity: 0.75;
}

.wa-fallback {
  font-size: 0.9em;
  opacity: 0.85;
}

.wa-fallback summary {
  cursor: pointer;
}

/* Selectable on purpose: this is the escape hatch when the link fails, and a parent has
   to be able to copy both of these by hand. */
.wa-copyable {
  direction: ltr;
  text-align: left;
  user-select: all;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background: rgba(127, 127, 127, 0.12);
  padding: 0.5rem 0.65rem;
  border-radius: 0.5rem;
  margin: 0.35rem 0;
  word-break: break-all;
}

.wa-code-form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.wa-code-form input {
  /* Wide tracking and a monospace face so six digits are read as six digits — a parent is
     copying them off a second screen. */
  letter-spacing: 0.5em;
  text-align: center;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 1.15rem;
}

.wa-error {
  margin: 0;
  color: #d64545;
  line-height: 1.6;
}
</style>
