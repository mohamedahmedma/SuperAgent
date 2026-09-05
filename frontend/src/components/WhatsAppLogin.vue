<template>
  <div class="wa-login" :dir="language === 'ar' ? 'rtl' : 'ltr'" :lang="language">
    <template v-if="wa.status === 'idle'">
      <p class="wa-lead">
        {{
          language === 'ar'
            ? 'سجّل الدخول برقم هاتفك المسجّل لدى المدرسة. لا حاجة لكلمة مرور.'
            : 'Sign in with the number the school has on file. No password needed.'
        }}
      </p>

      <button class="wa-primary" type="button" :disabled="wa.busy" @click="start">
        <i :class="wa.busy ? 'fa-solid fa-spinner fa-spin' : 'fa-brands fa-whatsapp'"></i>
        <span>
          {{
            wa.busy
              ? language === 'ar'
                ? 'جارٍ التحضير…'
                : 'Preparing…'
              : language === 'ar'
                ? 'المتابعة عبر واتساب'
                : 'Continue with WhatsApp'
          }}
        </span>
      </button>
    </template>

    <template v-else-if="wa.status === 'waiting'">
      <ol class="wa-steps">
        <li>
          {{
            language === 'ar'
              ? 'اضغط الزر بالأسفل — سيفتح واتساب برسالة جاهزة.'
              : 'Tap the button below — WhatsApp will open with a prepared message.'
          }}
        </li>
        <li>
          <strong>
            {{
              language === 'ar'
                ? 'اضغط زر الإرسال داخل واتساب'
                : 'Tap Send inside WhatsApp'
            }}
          </strong>
          {{
            language === 'ar'
              ? ' — لن تُرسل الرسالة تلقائيًا.'
              : ' — the message is not sent automatically.'
          }}
        </li>
        <li>
          {{
            language === 'ar'
              ? 'سنرد عليك برمز من ٦ أرقام، اكتبه هنا.'
              : 'We will reply with a 6-digit code. Enter it here.'
          }}
        </li>
      </ol>

      <a class="wa-primary" :href="wa.link" target="_blank" rel="noopener noreferrer">
        <i class="fa-brands fa-whatsapp"></i>
        <span>{{ language === 'ar' ? 'افتح واتساب وأرسل الرسالة' : 'Open WhatsApp and send the message' }}</span>
      </a>

      <p class="wa-waiting">
        <i class="fa-solid fa-spinner fa-spin"></i>
        {{ language === 'ar' ? 'في انتظار رسالتك…' : 'Waiting for your message…' }}
      </p>

      <details class="wa-fallback">
        <summary>{{ language === 'ar' ? 'لم يفتح واتساب؟' : 'WhatsApp did not open?' }}</summary>
        <p>{{ language === 'ar' ? 'أرسل هذه الرسالة يدويًا إلى الرقم:' : 'Send this message manually to:' }}</p>
        <p class="wa-copyable">{{ wa.businessNumber }}</p>
        <p class="wa-copyable">{{ wa.message }}</p>
      </details>

      <button class="wa-secondary" type="button" @click="cancel">
        {{ language === 'ar' ? 'إلغاء' : 'Cancel' }}
      </button>
    </template>

    <template v-else-if="wa.status === 'code_sent'">
      <p class="wa-lead">
        <template v-if="wa.displayName">
          {{ language === 'ar' ? `أهلًا ${wa.displayName} —` : `Welcome ${wa.displayName} —` }}
        </template>
        {{
          language === 'ar'
            ? ' أرسلنا رمزًا من ٦ أرقام إلى واتساب.'
            : ' We sent a 6-digit code to WhatsApp.'
        }}
      </p>

      <form class="wa-code-form" @submit.prevent="submit">
        <label class="form-field">
          <span>{{ language === 'ar' ? 'رمز التحقق' : 'Verification code' }}</span>
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
              @focus="onOtpFocus"
              @blur="onOtpBlur"
            />
          </span>
        </label>

        <p v-if="wa.error" class="wa-error" role="alert">{{ errorText }}</p>

        <button class="wa-primary" type="submit" :disabled="wa.busy || wa.code.trim().length < 6">
          <i :class="wa.busy ? 'fa-solid fa-spinner fa-spin' : language === 'ar' ? 'fa-solid fa-arrow-left' : 'fa-solid fa-arrow-right'"></i>
          <span>
            {{
              wa.busy
                ? language === 'ar'
                  ? 'جارٍ التحقق…'
                  : 'Verifying…'
                : language === 'ar'
                  ? 'تأكيد'
                  : 'Verify'
            }}
          </span>
        </button>
      </form>

      <button class="wa-secondary" type="button" @click="cancel">
        {{ language === 'ar' ? 'البدء من جديد' : 'Start over' }}
      </button>
    </template>

    <template v-else>
      <p class="wa-error" role="alert">{{ errorText }}</p>
      <button class="wa-primary" type="button" @click="start">
        <i class="fa-solid fa-rotate-right"></i>
        <span>{{ language === 'ar' ? 'المحاولة مرة أخرى' : 'Try again' }}</span>
      </button>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue';
import { useAuthStore } from '@/stores/auth';

const props = defineProps<{
  language: 'ar' | 'en';
}>();

const authStore = useAuthStore();
const wa = computed(() => authStore.whatsapp);
const codeInput = ref<HTMLInputElement | null>(null);


// aurexis-otp-privacy-v10-4-2
// OTP is treated as a private secret.
// While the OTP field is active, the robot deliberately looks away.
const notifyRobotOtpPrivacy = (mode: 'idle' | 'away') => {
  window.dispatchEvent(
    new CustomEvent('aurexis-password-privacy', {
      detail: { mode },
    })
  );
};

const onOtpFocus = () => {
  notifyRobotOtpPrivacy('away');
};

const onOtpBlur = () => {
  notifyRobotOtpPrivacy('idle');
};
const MESSAGES_AR: Record<string, string> = {
  bad_code: 'الرمز غير صحيح. تحقّق من الرسالة وحاول مرة أخرى.',
  too_many_attempts: 'محاولات كثيرة غير صحيحة. ابدأ من جديد للحصول على رمز جديد.',
  expired: 'انتهت صلاحية الطلب. ابدأ من جديد.',
  already_used: 'تم استخدام هذا الرمز بالفعل. ابدأ من جديد.',
  not_ready: 'لم نستلم رسالتك بعد. أرسل الرسالة من واتساب أولًا.',
  not_found: 'انتهت هذه الجلسة. ابدأ من جديد.',
  rejected: 'هذا الرقم غير مسجّل لدى المدرسة. تواصل مع إدارة المدرسة لإضافته.',
  not_configured: 'الدخول عبر واتساب غير متاح حاليًا. برجاء التواصل مع إدارة المدرسة.',
};

const MESSAGES_EN: Record<string, string> = {
  bad_code: 'That code is incorrect. Check the WhatsApp message and try again.',
  too_many_attempts: 'Too many incorrect attempts. Start again to request a new code.',
  expired: 'This request has expired. Please start again.',
  already_used: 'This code has already been used. Please start again.',
  not_ready: 'We have not received your message yet. Send the WhatsApp message first.',
  not_found: 'This sign-in session has ended. Please start again.',
  rejected: 'This number is not registered with the school. Contact the school administration.',
  not_configured: 'WhatsApp sign-in is currently unavailable. Please contact the school administration.',
};

const errorText = computed(() => {
  const messages = props.language === 'ar' ? MESSAGES_AR : MESSAGES_EN;
  return messages[wa.value.errorCode] || wa.value.error;
});

const start = () => authStore.startWhatsAppLogin();
const cancel = () => {
  notifyRobotOtpPrivacy('idle');
  authStore.resetWhatsApp();
};
const submit = () => authStore.submitWhatsAppCode();

watch(
  () => wa.value.status,
  async (status) => {
    if (status === 'code_sent') {
      await nextTick();
      codeInput.value?.focus();
      notifyRobotOtpPrivacy('away');
    } else {
      notifyRobotOtpPrivacy('idle');
    }
  }
);

onBeforeUnmount(() => {
  notifyRobotOtpPrivacy('idle');
  authStore.resetWhatsApp();
});
</script>

<style scoped>
.wa-login {
  display: flex;
  flex-direction: column;
  gap: 0.9rem;
  text-align: start;
}

.wa-lead {
  margin: 0;
  line-height: 1.7;
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
  background: #25d366;
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

/* aurexis-whatsapp-exact-00a03c-v10-5-2 */
.wa-button,
button[class*="whatsapp"],
button[class*="WhatsApp"],
.ax-whatsapp-button,
button:has(.fa-whatsapp),
button:has([class*="whatsapp"]) {
  background:#00A03C!important;
  background-image:none!important;
  border-color:#00A03C!important;
  color:#FFFFFF!important;
  opacity:1!important;
  font-size:1.06rem!important;
  font-weight:700!important;
  letter-spacing:.035em!important;
  gap:.72rem!important;
  column-gap:.72rem!important;
  box-shadow:0 10px 24px rgba(0,160,60,.20)!important;
}

.wa-button .fa-whatsapp,
.wa-button [class*="whatsapp"],
button[class*="whatsapp"] .fa-whatsapp,
button[class*="WhatsApp"] .fa-whatsapp,
.ax-whatsapp-button .fa-whatsapp,
button:has(.fa-whatsapp) .fa-whatsapp,
button:has([class*="whatsapp"]) [class*="whatsapp"] {
  font-size:1.34em!important;
  line-height:1!important;
  color:#FFFFFF!important;
  flex:0 0 auto!important;
}

.wa-button:hover,
button[class*="whatsapp"]:hover,
button[class*="WhatsApp"]:hover,
.ax-whatsapp-button:hover,
button:has(.fa-whatsapp):hover,
button:has([class*="whatsapp"]):hover {
  background:#008F36!important;
  background-image:none!important;
  border-color:#008F36!important;
  color:#FFFFFF!important;
  box-shadow:0 12px 28px rgba(0,160,60,.25)!important;
}

.wa-button:active,
button[class*="whatsapp"]:active,
button[class*="WhatsApp"]:active,
.ax-whatsapp-button:active,
button:has(.fa-whatsapp):active,
button:has([class*="whatsapp"]):active {
  background:#007F30!important;
  background-image:none!important;
  border-color:#007F30!important;
  color:#FFFFFF!important;
  transform:translateY(1px)!important;
}

.wa-button:focus-visible,
button[class*="whatsapp"]:focus-visible,
button[class*="WhatsApp"]:focus-visible,
.ax-whatsapp-button:focus-visible,
button:has(.fa-whatsapp):focus-visible,
button:has([class*="whatsapp"]):focus-visible {
  outline:3px solid rgba(0,160,60,.24)!important;
  outline-offset:2px!important;
}
</style>
