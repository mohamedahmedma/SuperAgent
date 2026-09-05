<template>
  <section class="welcome-screen">
    <div class="welcome-avatar" aria-hidden="true">
      <BrandLogo size="xl" />
    </div>

    <span class="welcome-eyebrow">AUREXIS IS READY</span>

    <h2>Hi, I'm Aurexis.</h2>

    <p>
      Aurexis School Assistant searches your school knowledge base while answering,
      shows its working, and links key conclusions back to the original evidence.
    </p>

    <div class="welcome-capabilities">
      <span><i class="fa-solid fa-magnifying-glass"></i> Hybrid retrieval</span>
      <span><i class="fa-solid fa-diagram-project"></i> Parallel agents</span>
      <span><i class="fa-solid fa-link"></i> Traceable citations</span>
    </div>
  </section>
</template>

<script setup lang="ts">
import BrandLogo from '@/components/BrandLogo.vue';
</script>
