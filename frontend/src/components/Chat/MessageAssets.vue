<template>
  <div v-if="assets.length" class="message-assets">
    <figure v-for="asset in assets" :key="asset.asset_id" class="asset-card">
      <div class="asset-frame">
        <img
          v-if="sources[asset.asset_id]"
          :src="sources[asset.asset_id]"
          :alt="asset.alt_text || asset.caption || 'Figure from the knowledge base'"
          class="asset-image"
          loading="lazy"
          @click="open(asset)"
        />
        <div v-else-if="failed[asset.asset_id]" class="asset-placeholder asset-failed">
          <i class="fa-regular fa-image"></i>
          <span>Image unavailable</span>
        </div>
        <div v-else class="asset-placeholder">
          <i class="fa-solid fa-spinner fa-spin"></i>
        </div>
      </div>
      <figcaption class="asset-caption">
        <span class="asset-title">{{ asset.caption || 'Figure' }}</span>
        <small v-if="asset.source?.filename" class="asset-source">
          <i class="fa-regular fa-file-lines"></i>{{ asset.source.filename }}
        </small>
      </figcaption>
    </figure>
  </div>
</template>

<script setup lang="ts">
/**
 * Renders the images the backend surfaced for an answer.
 *
 * The backend sends structured `AssetReference` descriptors, never markup, so all the
 * presentation decisions live here.
 *
 * Fetching deserves a note: the asset endpoint is authenticated with a Bearer token,
 * and a plain `<img src>` cannot carry a header — it would 401. So each image is
 * fetched with the token and turned into an object URL. Those URLs hold memory until
 * revoked, which is why they are released on unmount.
 */
import { onBeforeUnmount, ref, watch } from 'vue';

import { useAuthStore } from '../../stores/auth';
import type { AssetReference } from '../../types/chat';
import { apiUrl } from '../../utils/api';

const props = defineProps<{ assets?: AssetReference[] }>();

const authStore = useAuthStore();
const sources = ref<Record<string, string>>({});
const failed = ref<Record<string, boolean>>({});
const objectUrls: string[] = [];

const assets = ref<AssetReference[]>([]);

const release = () => {
  objectUrls.forEach((url) => URL.revokeObjectURL(url));
  objectUrls.length = 0;
};

const load = async (asset: AssetReference) => {
  if (sources.value[asset.asset_id] || failed.value[asset.asset_id]) return;

  // Already embedded — nothing to fetch.
  if (asset.mode === 'inline' && asset.inline_data) {
    sources.value = { ...sources.value, [asset.asset_id]: asset.inline_data };
    return;
  }
  if (asset.mode === 'metadata' || !asset.url) {
    failed.value = { ...failed.value, [asset.asset_id]: true };
    return;
  }

  try {
    /* `asset.url` arrives as a path — `/media/{asset_id}`, built by
       `backend/assets/delivery.py` — so a bare `fetch` resolves it against the PAGE's
       origin. On a UI deployed to its own domain that is a request to the static server,
       and every image renders as "Image unavailable" while the answer's text arrives
       perfectly. `apiUrl` sends it where the rest of the API goes, and passes an absolute
       URL through untouched should the backend ever emit one. */
    const response = await fetch(apiUrl(asset.url), {
      headers: { Authorization: `Bearer ${authStore.token}` },
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const url = URL.createObjectURL(await response.blob());
    objectUrls.push(url);
    sources.value = { ...sources.value, [asset.asset_id]: url };
  } catch {
    failed.value = { ...failed.value, [asset.asset_id]: true };
  }
};

const open = (asset: AssetReference) => {
  const src = sources.value[asset.asset_id];
  if (src) window.open(src, '_blank', 'noopener');
};

watch(
  () => props.assets,
  (next) => {
    // Only figures carry a picture worth showing; metadata-only entries would render
    // as an empty frame.
    assets.value = (next || []).filter((a) => a.mode !== 'metadata' || a.caption);
    assets.value.forEach(load);
  },
  { immediate: true, deep: true },
);

onBeforeUnmount(release);
</script>

<style scoped>
.message-assets {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 12px;
}

.asset-card {
  margin: 0;
  max-width: 260px;
  border: 1px solid var(--border-color, #e3e3e3);
  border-radius: 10px;
  overflow: hidden;
  background: var(--card-bg, #fff);
}

.asset-frame {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 120px;
  background: #f6f7f9;
}

.asset-image {
  display: block;
  max-width: 100%;
  max-height: 240px;
  cursor: zoom-in;
}

.asset-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 24px;
  color: #8a8a8a;
  font-size: 0.8rem;
}

.asset-failed {
  color: #9e3a3a;
}

.asset-caption {
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.asset-title {
  font-size: 0.82rem;
  font-weight: 600;
  line-height: 1.3;
}

.asset-source {
  font-size: 0.72rem;
  color: #777;
  display: flex;
  align-items: center;
  gap: 4px;
}

@media (prefers-color-scheme: dark) {
  .asset-card {
    border-color: #3a3a3a;
    background: #1f1f1f;
  }
  .asset-frame {
    background: #171717;
  }
  .asset-source {
    color: #9a9a9a;
  }
}
</style>
