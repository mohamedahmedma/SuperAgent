import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import './assets/styles/main.css';
import '@fortawesome/fontawesome-free/css/all.min.css';

const app = createApp(App);
const pinia = createPinia();

app.use(pinia);
app.mount('#app');
import './assets/styles/aurexis-final-match-v5.css';
import './assets/styles/aurexis-theme-lights-auth-v6.css';
import './assets/styles/aurexis-fullscreen-v7.css';
import './assets/styles/aurexis-auth-repair-v8.css';
