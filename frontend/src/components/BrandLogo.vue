<template>
  <span :class="['aurexis-logo-object', `aurexis-logo-object--${size}`]" aria-hidden="true">
    <svg viewBox="250 220 780 760" role="img" focusable="false">
      <defs>
        <linearGradient id="ax-main" x1="760" y1="250" x2="290" y2="935" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#18CFFF"/>
          <stop offset=".42" stop-color="#238BFF"/>
          <stop offset=".74" stop-color="#4450FF"/>
          <stop offset="1" stop-color="#7428FF"/>
        </linearGradient>
        <linearGradient id="ax-right" x1="700" y1="575" x2="855" y2="915" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#22CBFF"/>
          <stop offset=".48" stop-color="#2D7DFF"/>
          <stop offset="1" stop-color="#6B28FF"/>
        </linearGradient>
        <linearGradient id="ax-square" x1="890" y1="830" x2="990" y2="940" gradientUnits="userSpaceOnUse">
          <stop offset="0" stop-color="#2C8BFF"/>
          <stop offset="1" stop-color="#6230FF"/>
        </linearGradient>
        <filter id="ax-depth" x="-20%" y="-20%" width="150%" height="160%">
          <feDropShadow dx="6" dy="10" stdDeviation="7" flood-color="#020817" flood-opacity=".52"/>
          <feDropShadow dx="0" dy="0" stdDeviation="3.2" flood-color="#1ABEFF" flood-opacity=".16"/>
        </filter>
      </defs>

      <g filter="url(#ax-depth)">
        <polygon
          points="762,252 285,753 285,936 762,444"
          fill="url(#ax-main)"
        />
        <polygon
          points="719,586 626,684 851,912 851,793 916,791"
          fill="url(#ax-right)"
        />
        <rect x="887" y="829" width="104" height="107" rx="0" fill="url(#ax-square)" />

        <!-- restrained edge highlights only; no circle, no card, no background -->
        <polyline points="762,252 285,753" fill="none" stroke="#66E9FF" stroke-width="3.2" stroke-linecap="square" opacity=".72"/>
        <polyline points="719,586 626,684" fill="none" stroke="#64E6FF" stroke-width="2.5" opacity=".55"/>
        <polyline points="887,829 991,829" fill="none" stroke="#5CCEFF" stroke-width="2.2" opacity=".40"/>
      </g>
    </svg>
  </span>
</template>

<script setup lang="ts">
withDefaults(
  defineProps<{ size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl' }>(),
  { size: 'md' }
);
</script>
