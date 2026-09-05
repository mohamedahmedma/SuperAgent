<template>
  <main class="ax-auth-root" :dir="language === 'ar' ? 'rtl' : 'ltr'" data-aurexis-auth-v9="true">
    <div class="ax-auth-controls" dir="ltr">
      <button class="ax-control" type="button" @click="toggleLanguage">
        <i class="fa-solid fa-language"></i>
        <span>{{ language === 'ar' ? 'English' : 'العربية' }}</span>
      </button>

      <button class="ax-control ax-theme" type="button" @click="$emit('toggle-theme')">
        <i :class="theme === 'dark' ? 'fa-regular fa-moon' : 'fa-regular fa-sun'"></i>
        <span>{{ theme === 'dark' ? (language === 'ar' ? 'داكن' : 'Dark') : (language === 'ar' ? 'فاتح' : 'Light') }}</span>
      </button>
    </div>

    <section class="ax-auth-shell">
      <div class="ax-hero">
        <div class="ax-brand-row">
          <BrandLogo size="lg" />
          <div class="ax-brand-copy">
            <strong>AUREXIS</strong>
            <span>SCHOOL ASSISTANT</span>
          </div>
        </div>

        <div class="ax-hero-copy">
          <span class="ax-kicker">
            {{ language === 'ar' ? 'منصة المدرسة الذكية' : 'SMART SCHOOL ACCESS' }}
          </span>

          <h1 v-if="language === 'ar'">
            كل ما يخص طفلك،<br />
            في مكان واحد.
          </h1>
          <h1 v-else>
            Everything about your child,<br />
            in one place.
          </h1>

          <p v-if="language === 'ar'">
            استخدم رقم واتساب المسجّل لدى المدرسة للاستفسار عن الحضور والدرجات
            والمعلومات المدرسية بسهولة وأمان.
          </p>
          <p v-else>
            Use the WhatsApp number registered with the school to ask about
            attendance, grades, and school information securely.
          </p>
        </div>

        <div class="ax-hero-meta">
          <span><i class="fa-solid fa-shield-halved"></i>{{ language === 'ar' ? 'وصول آمن' : 'Secure access' }}</span>
          <span><i class="fa-solid fa-school"></i>{{ language === 'ar' ? 'بيانات المدرسة' : 'School data' }}</span>
          <span><i class="fa-solid fa-bolt"></i>{{ language === 'ar' ? 'دخول سريع' : 'Fast sign in' }}</span>
        </div>
      </div>

      <div class="ax-login-panel">
        <div class="ax-login-heading">
          <BrandLogo size="md" />
          <div>
            <span class="ax-eyebrow">
              {{ audience === 'parent'
                ? (language === 'ar' ? 'أهلًا بك في AUREXIS' : 'WELCOME TO AUREXIS')
                : (language === 'ar' ? 'دخول فريق المدرسة' : 'STAFF ACCESS') }}
            </span>
            <h2>
              {{ audience === 'parent'
                ? (language === 'ar' ? 'الدخول لأولياء الأمور' : 'Parent sign in')
                : (language === 'ar' ? 'تسجيل دخول الموظفين' : 'Staff sign in') }}
            </h2>
          </div>
        </div>

        <div class="ax-tabs" role="tablist">
          <button
            type="button"
            :class="{ active: audience === 'parent' }"
            @click="showParent"
          >
            <i class="fa-solid fa-user-group"></i>
            <span>{{ language === 'ar' ? 'ولي أمر' : 'Parent' }}</span>
          </button>

          <button
            type="button"
            :class="{ active: audience === 'staff' }"
            @click="showStaff"
          >
            <i class="fa-solid fa-briefcase"></i>
            <span>{{ language === 'ar' ? 'الموظفون' : 'Staff' }}</span>
          </button>
        </div>

        <div v-if="audience === 'parent'" class="ax-parent-wrap">
          <WhatsAppLogin :language="language" />
        </div>

        <template v-else>
          <p class="ax-description">
            {{ language === 'ar'
              ? 'استخدم بيانات حساب المدرسة للوصول إلى مساحة العمل الخاصة بك.'
              : 'Use your school account credentials to access your private workspace.' }}
          </p>

          <form class="ax-form" @submit.prevent="onSubmit">
            <label>
              <span>{{ language === 'ar' ? 'اسم المستخدم' : 'Username' }}</span>
              <div class="ax-field">
                <i class="fa-regular fa-user"></i>
                <input
                  v-model="authStore.authForm.username"
                  type="text"
                  autocomplete="username"
                  :placeholder="language === 'ar' ? 'أدخل اسم المستخدم' : 'Enter your username'"
                />
              </div>
            </label>

            <label>
              <span>{{ language === 'ar' ? 'كلمة المرور' : 'Password' }}</span>
              <div class="ax-field">
                <i class="fa-solid fa-lock"></i>
                <input
                  v-model="authStore.authForm.password"
                  :type="showPassword ? 'text' : 'password'"
                  autocomplete="current-password"
                  :placeholder="language === 'ar' ? 'أدخل كلمة المرور' : 'Enter your password'"
                  @focus="onPasswordFocus"
                  @blur="onPasswordBlur"
                />
                <button
                  class="ax-password-toggle"
                  type="button"
                  :aria-label="showPassword
                    ? (language === 'ar' ? 'إخفاء كلمة المرور' : 'Hide password')
                    : (language === 'ar' ? 'إظهار كلمة المرور' : 'Show password')"
                  :title="showPassword
                    ? (language === 'ar' ? 'إخفاء كلمة المرور' : 'Hide password')
                    : (language === 'ar' ? 'إظهار كلمة المرور' : 'Show password')"
                  @mousedown.prevent
                  @click="togglePasswordVisibility"
                >
                  <i :class="showPassword ? 'fa-regular fa-eye-slash' : 'fa-regular fa-eye'"></i>
                </button>
              </div>
            </label>

            <button class="ax-submit" type="submit" :disabled="authStore.authLoading">
              <span>
                {{ authStore.authLoading
                  ? (language === 'ar' ? 'جارٍ الاتصال...' : 'Connecting...')
                  : (language === 'ar' ? 'دخول مساحة العمل' : 'Enter workspace') }}
              </span>
              <i
                :class="authStore.authLoading
                  ? 'fa-solid fa-spinner fa-spin'
                  : (language === 'ar' ? 'fa-solid fa-arrow-left' : 'fa-solid fa-arrow-right')"
              ></i>
            </button>
          </form>
        </template>
      </div>
    </section>

    <footer class="ax-auth-footer">
      <span>
        Powered by
        <a
          href="https://aurexis.cc/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Visit AUREXIS website"
        >AUREXIS</a>
      </span>
    </footer>

    <div class="ax-robot-dock" aria-hidden="true">
      <AurexisRobot />
    </div>
  </main>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import BrandLogo from '@/components/BrandLogo.vue';
import AurexisRobot from '@/components/AurexisRobot.vue';
import WhatsAppLogin from '@/components/WhatsAppLogin.vue';
import { useAuthStore } from '@/stores/auth';

defineProps<{ theme: 'dark' | 'light' }>();
defineEmits<{ (e: 'toggle-theme'): void }>();

const authStore = useAuthStore();

const storedLanguage = localStorage.getItem('superagent-language');
const language = ref<'ar' | 'en'>(storedLanguage === 'en' ? 'en' : 'ar');
const audience = ref<'parent' | 'staff'>('parent');
const showPassword = ref(false);

const toggleLanguage = () => {
  language.value = language.value === 'ar' ? 'en' : 'ar';
  localStorage.setItem('superagent-language', language.value);
};

const showParent = () => {
  audience.value = 'parent';
};

const showStaff = () => {
  authStore.resetWhatsApp();
  audience.value = 'staff';
};

const notifyRobotPasswordState = (mode: 'idle' | 'away' | 'peek') => {
  window.dispatchEvent(
    new CustomEvent('aurexis-password-privacy', {
      detail: { mode },
    })
  );
};

const onPasswordFocus = () => {
  notifyRobotPasswordState(showPassword.value ? 'peek' : 'away');
};

const onPasswordBlur = () => {
  notifyRobotPasswordState('idle');
};

const togglePasswordVisibility = () => {
  showPassword.value = !showPassword.value;
  notifyRobotPasswordState(showPassword.value ? 'peek' : 'away');
};

const onSubmit = async () => {
  try {
    await authStore.handleAuthSubmit();
  } catch (error: any) {
    alert(error.message);
  }
};
</script>

<style>
/* V9 styles are intentionally embedded here so they CANNOT be lost because
   of a missing global CSS import. Signature: AUREXIS_AUTH_V9_SIGNATURE */

.ax-auth-root{
  --bg:#101e2e;
  --bg2:#17273a;
  --panel:#0a1c2d;
  --panel2:#0d2539;
  --line:rgba(69,145,189,.22);
  --line2:rgba(43,180,219,.36);
  --txt:#f7f9fc;
  --muted:#819db7;
  --muted2:#607f9a;
  --cyan:#25c5e4;

  position:fixed;
  inset:0;
  z-index:0;
  display:grid;
  grid-template-rows:1fr auto;
  overflow:auto;
  color:var(--txt);
  background:
    radial-gradient(900px 520px at 7% 0%,rgba(32,126,172,.065),transparent 67%),
    radial-gradient(800px 480px at 100% 100%,rgba(54,92,154,.045),transparent 72%),
    linear-gradient(145deg,var(--bg),var(--bg2));
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}

.ax-auth-root *,
.ax-auth-root *::before,
.ax-auth-root *::after{box-sizing:border-box}

.ax-auth-controls{
  position:fixed;
  top:22px;
  left:24px;
  z-index:30;
  display:flex;
  gap:8px;
}

.ax-control{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  gap:7px;
  min-height:38px;
  padding:8px 12px;
  border:1px solid rgba(91,152,190,.24);
  border-radius:999px;
  color:#9eb5c9;
  background:rgba(7,25,41,.74);
  box-shadow:0 8px 22px rgba(0,0,0,.10);
  backdrop-filter:blur(10px);
  font:inherit;
  font-size:12px;
  font-weight:700;
  cursor:pointer;
}

.ax-auth-shell{
  align-self:center;
  display:grid;
  grid-template-columns:minmax(0,1.08fr) minmax(390px,.78fr);
  width:min(1160px,calc(100% - 64px));
  margin:86px auto 36px;
  overflow:hidden;
  border:1px solid var(--line);
  border-radius:28px;
  background:rgba(7,24,40,.56);
  box-shadow:
    0 30px 80px rgba(0,7,16,.24),
    0 8px 24px rgba(0,7,16,.15),
    inset 0 1px 0 rgba(255,255,255,.025);
}

.ax-hero{
  position:relative;
  display:flex;
  min-height:590px;
  flex-direction:column;
  padding:48px 52px;
  overflow:hidden;
  border-inline-end:1px solid rgba(65,136,179,.15);
  background:
    radial-gradient(520px 330px at 18% 0%,rgba(34,148,188,.075),transparent 68%),
    linear-gradient(150deg,rgba(17,48,73,.82),rgba(7,26,42,.76));
}

.ax-hero::after{
  content:"";
  position:absolute;
  right:-130px;
  bottom:-150px;
  width:390px;
  height:390px;
  border:1px solid rgba(37,197,228,.075);
  border-radius:50%;
  box-shadow:0 0 0 52px rgba(37,197,228,.018),0 0 0 104px rgba(37,197,228,.010);
}

.ax-brand-row{
  position:relative;
  z-index:1;
  display:flex;
  align-items:center;
  gap:14px;
}

.ax-brand-copy{
  display:flex;
  flex-direction:column;
  gap:5px;
}
.ax-brand-copy strong{
  color:#fff;
  font-size:16px;
  font-weight:900;
  line-height:1;
  letter-spacing:.09em;
}
.ax-brand-copy span{
  color:#75a2c1;
  font-size:9px;
  font-weight:850;
  letter-spacing:.18em;
}

.ax-hero-copy{
  position:relative;
  z-index:1;
  max-width:590px;
  margin:auto 0;
}
.ax-kicker{
  display:inline-block;
  margin-bottom:18px;
  color:#2bc2df;
  font-size:11px;
  font-weight:850;
  letter-spacing:.06em;
}
.ax-hero h1{
  margin:0;
  color:#f8fafc;
  font-size:clamp(46px,4.3vw,66px);
  font-weight:900;
  line-height:.98;
  letter-spacing:-.058em;
}
.ax-hero p{
  max-width:530px;
  margin:24px 0 0;
  color:#8aa5be;
  font-size:14px;
  font-weight:450;
  line-height:1.85;
}

.ax-hero-meta{
  position:relative;
  z-index:1;
  display:flex;
  flex-wrap:wrap;
  gap:8px;
}
.ax-hero-meta span{
  display:inline-flex;
  align-items:center;
  gap:7px;
  padding:7px 10px;
  border:1px solid rgba(71,135,174,.15);
  border-radius:999px;
  color:#7998b1;
  background:rgba(5,23,38,.38);
  font-size:10px;
  font-weight:650;
}
.ax-hero-meta i{color:#25bbd8}

.ax-login-panel{
  display:flex;
  min-width:0;
  flex-direction:column;
  justify-content:center;
  padding:48px 46px;
  background:linear-gradient(180deg,rgba(7,24,40,.95),rgba(5,21,35,.97));
}

.ax-login-heading{
  display:flex;
  align-items:center;
  gap:14px;
  margin-bottom:24px;
}
.ax-eyebrow{
  display:block;
  margin-bottom:5px;
  color:#29bfdc;
  font-size:10px;
  font-weight:850;
  letter-spacing:.055em;
}
.ax-login-panel h2{
  margin:0;
  color:#f7f9fb;
  font-size:27px;
  font-weight:850;
  line-height:1.12;
  letter-spacing:-.038em;
}

.ax-tabs{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:8px;
  margin-bottom:22px;
  padding:4px;
  border:1px solid rgba(69,132,171,.17);
  border-radius:14px;
  background:rgba(4,18,31,.58);
}
.ax-tabs button{
  display:flex;
  align-items:center;
  justify-content:center;
  gap:8px;
  min-height:43px;
  border:1px solid transparent;
  border-radius:10px;
  color:#7895ae;
  background:transparent;
  font:inherit;
  font-size:12px;
  font-weight:700;
  cursor:pointer;
}
.ax-tabs button.active{
  color:#edf8fc;
  border-color:rgba(49,164,207,.20);
  background:#0a3651;
}

.ax-description{
  margin:0 0 20px;
  color:#829eb7;
  font-size:13px;
  line-height:1.65;
}

.ax-form{
  display:flex;
  flex-direction:column;
  gap:15px;
}
.ax-form label{
  display:flex;
  flex-direction:column;
  gap:7px;
}
.ax-form label>span{
  color:#8aa3ba;
  font-size:11px;
  font-weight:700;
}
.ax-field{
  position:relative;
  display:flex;
  align-items:center;
}
.ax-field i{
  position:absolute;
  left:14px;
  z-index:1;
  color:#60829d;
}
[dir="rtl"] .ax-field i{left:auto;right:14px}
.ax-field input{
  width:100%;
  min-height:47px;
  padding:10px 14px 10px 40px;
  border:1px solid rgba(65,128,167,.22);
  border-radius:11px;
  outline:0;
  color:#edf3f8;
  background:rgba(4,20,34,.69);
  font:inherit;
  font-size:12px;
}
[dir="rtl"] .ax-field input{padding:10px 40px 10px 14px}
.ax-field input:focus{
  border-color:rgba(37,190,220,.44);
  box-shadow:0 0 0 3px rgba(37,190,220,.055);
}
.ax-field input::placeholder{color:#56758f}

.ax-submit{
  display:flex;
  align-items:center;
  justify-content:center;
  gap:9px;
  min-height:49px;
  margin-top:4px;
  border:0;
  border-radius:12px;
  color:#062333;
  background:linear-gradient(180deg,#2bc8e4,#19a9ca);
  box-shadow:0 9px 24px rgba(26,170,203,.16);
  font:inherit;
  font-size:12px;
  font-weight:800;
  cursor:pointer;
}

/* Force the current WhatsAppLogin component into the same design, despite its scoped CSS. */
.ax-parent-wrap .wa-login{
  display:flex!important;
  flex-direction:column!important;
  gap:14px!important;
  text-align:start!important;
}
.ax-parent-wrap .wa-lead{
  margin:0!important;
  color:#89a4bd!important;
  font-size:13px!important;
  line-height:1.75!important;
}
.ax-parent-wrap .wa-lead-en{
  display:block!important;
  margin-top:4px!important;
  color:#6f8ba4!important;
  font-size:11px!important;
  direction:ltr!important;
  text-align:left!important;
}
.ax-parent-wrap .wa-primary{
  display:inline-flex!important;
  align-items:center!important;
  justify-content:center!important;
  gap:8px!important;
  width:100%!important;
  min-height:49px!important;
  padding:10px 14px!important;
  border:0!important;
  border-radius:12px!important;
  color:#062333!important;
  background:linear-gradient(180deg,#2bc8e4,#19a9ca)!important;
  box-shadow:0 9px 24px rgba(26,170,203,.16)!important;
  font:inherit!important;
  font-size:12px!important;
  font-weight:800!important;
  text-decoration:none!important;
}
.ax-parent-wrap .wa-secondary{
  width:100%!important;
  min-height:42px!important;
  border:1px solid rgba(71,136,175,.22)!important;
  border-radius:11px!important;
  color:#7896af!important;
  background:transparent!important;
}
.ax-parent-wrap .wa-copyable{
  color:#b8c8d6!important;
  background:rgba(4,19,32,.70)!important;
}
.ax-parent-wrap .wa-error{color:#ef9292!important}

.ax-auth-footer{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:18px;
  width:min(1160px,calc(100% - 64px));
  margin:0 auto;
  padding:0 2px 22px;
  color:#58758f;
  font-size:9px;
}
.ax-auth-footer strong{
  color:#8ca6bd;
  letter-spacing:.12em;
}

/* LIGHT */
html[data-theme="light"] .ax-auth-root{
  --txt:#172a3d;
  background:
    radial-gradient(850px 500px at 7% 0%,rgba(37,142,178,.055),transparent 67%),
    linear-gradient(145deg,#d6e1ea,#c3d2de);
}
html[data-theme="light"] .ax-control{
  color:#526f88;
  border-color:rgba(56,108,143,.20);
  background:rgba(188,207,220,.78);
}
html[data-theme="light"] .ax-auth-shell{
  border-color:rgba(53,105,141,.19);
  background:rgba(192,210,222,.58);
  box-shadow:0 28px 72px rgba(45,65,82,.11),inset 0 1px 0 rgba(255,255,255,.48);
}
html[data-theme="light"] .ax-hero{
  border-inline-end-color:rgba(60,109,141,.14);
  background:linear-gradient(150deg,rgba(199,216,227,.90),rgba(182,202,216,.86));
}
html[data-theme="light"] .ax-brand-copy strong,
html[data-theme="light"] .ax-hero h1,
html[data-theme="light"] .ax-login-panel h2{color:#172b3e}
html[data-theme="light"] .ax-hero p,
html[data-theme="light"] .ax-description,
html[data-theme="light"] .ax-parent-wrap .wa-lead{color:#5e7b94!important}
html[data-theme="light"] .ax-login-panel{
  background:linear-gradient(180deg,rgba(203,218,228,.96),rgba(193,210,222,.97));
}
html[data-theme="light"] .ax-tabs{
  border-color:rgba(64,114,145,.15);
  background:rgba(177,198,212,.54);
}
html[data-theme="light"] .ax-tabs button{color:#5e7890}
html[data-theme="light"] .ax-tabs button.active{
  color:#17364d;
  background:#aac8d8;
}
html[data-theme="light"] .ax-field input{
  color:#193248;
  border-color:rgba(57,111,146,.20);
  background:rgba(179,200,214,.70);
}
html[data-theme="light"] .ax-form label>span{color:#5b758d}
html[data-theme="light"] .ax-auth-footer{color:#607990}

@media(max-width:900px){
  .ax-auth-shell{
    grid-template-columns:1fr;
    width:min(640px,calc(100% - 32px));
    margin-top:82px;
  }
  .ax-hero{
    min-height:auto;
    padding:34px 32px;
    border-inline-end:0;
    border-bottom:1px solid rgba(58,137,186,.15);
  }
  .ax-hero-copy{margin:46px 0}
  .ax-login-panel{padding:38px 32px}
  .ax-auth-footer{width:min(640px,calc(100% - 32px))}
}
@media(max-width:560px){
  .ax-auth-controls{top:12px;left:12px}
  .ax-control{min-height:34px;padding:7px 10px;font-size:11px}
  .ax-auth-shell{width:calc(100% - 20px);margin-top:66px;border-radius:20px}
  .ax-hero{padding:28px 22px}
  .ax-hero h1{font-size:36px}
  .ax-login-panel{padding:28px 22px}
  .ax-login-panel h2{font-size:22px}
  .ax-auth-footer{
    width:calc(100% - 24px);
    flex-direction:column;
    align-items:flex-start;
  }
}

/* V10 footer: centered, single line, linked company name */
.ax-auth-footer{
  justify-content:center!important;
  text-align:center!important;
  padding-bottom:20px!important;
  font-size:10px!important;
}
.ax-auth-footer a{
  margin-inline-start:4px!important;
  color:#8ca6bd!important;
  font-weight:850!important;
  letter-spacing:.12em!important;
  text-decoration:none!important;
  transition:color .18s ease,text-shadow .18s ease!important;
}
.ax-auth-footer a:hover,
.ax-auth-footer a:focus-visible{
  color:#2bc8e4!important;
  text-shadow:0 0 12px rgba(43,200,228,.22)!important;
  outline:none!important;
}

/* Restore the original procedural AUREXIS 3D robot at bottom-right. */
.ax-robot-dock{
  position:fixed!important;
  right:18px!important;
  bottom:8px!important;
  z-index:12!important;
  width:250px!important;
  height:320px!important;
  pointer-events:none!important;
  overflow:visible!important;
  opacity:.98!important;
  filter:drop-shadow(0 18px 24px rgba(0,0,0,.18))!important;
}
.ax-robot-dock .aurexis-robot{
  width:100%!important;
  height:100%!important;
  display:block!important;
}
.ax-robot-dock canvas{
  display:block!important;
  width:100%!important;
  height:100%!important;
  background:transparent!important;
}

/* Keep the robot balanced with the auth card at medium widths. */
@media(max-width:1250px){
  .ax-robot-dock{
    width:205px!important;
    height:270px!important;
    right:6px!important;
    bottom:4px!important;
    opacity:.92!important;
  }
}
@media(max-width:1000px){
  .ax-robot-dock{
    width:170px!important;
    height:225px!important;
    right:0!important;
    bottom:0!important;
    opacity:.80!important;
  }
}
@media(max-width:760px){
  .ax-robot-dock{
    display:none!important;
  }
}


/* V10.2 bilingual directional polish */
.ax-auth-root[dir="rtl"] .ax-submit{
  flex-direction:row!important;
}
.ax-auth-root[dir="ltr"] .ax-submit{
  flex-direction:row!important;
}
.ax-auth-root[dir="rtl"] .ax-submit i,
.ax-auth-root[dir="ltr"] .ax-submit i{
  transform:none!important;
}
.ax-auth-root[dir="rtl"] .ax-tabs,
.ax-auth-root[dir="rtl"] .ax-login-heading,
.ax-auth-root[dir="rtl"] .ax-brand-row{
  direction:rtl!important;
}
.ax-auth-root[dir="ltr"] .ax-tabs,
.ax-auth-root[dir="ltr"] .ax-login-heading,
.ax-auth-root[dir="ltr"] .ax-brand-row{
  direction:ltr!important;
}

/* V10.4 password reveal control */
.ax-field:has(.ax-password-toggle) input{
  padding-inline-end:44px!important;
}
[dir="rtl"] .ax-field:has(.ax-password-toggle) input{
  padding-inline-start:44px!important;
  padding-inline-end:40px!important;
}
.ax-password-toggle{
  position:absolute!important;
  right:8px!important;
  z-index:2!important;
  display:grid!important;
  width:32px!important;
  height:32px!important;
  place-items:center!important;
  padding:0!important;
  border:0!important;
  border-radius:9px!important;
  color:#6f8da8!important;
  background:transparent!important;
  box-shadow:none!important;
  font:inherit!important;
  cursor:pointer!important;
  transition:color .18s ease,background .18s ease,transform .18s ease!important;
}
[dir="rtl"] .ax-password-toggle{
  right:auto!important;
  left:8px!important;
}
.ax-password-toggle:hover,
.ax-password-toggle:focus-visible{
  color:#d9f6fb!important;
  background:rgba(37,190,220,.08)!important;
  outline:none!important;
}
.ax-password-toggle:active{
  transform:scale(.94)!important;
}
html[data-theme="light"] .ax-password-toggle{
  color:#58768f!important;
}
html[data-theme="light"] .ax-password-toggle:hover,
html[data-theme="light"] .ax-password-toggle:focus-visible{
  color:#17415a!important;
  background:rgba(45,121,158,.09)!important;
}
</style>
