<template>
  <div class="message-parts" @click="onContentClick">
    <template v-for="(part, index) in parts" :key="index">
      <div
        v-if="part.kind === 'prose'"
        class="message-content"
        v-html="part.html"
      ></div>
      <MessageAssets v-else-if="part.asset" :assets="[part.asset]" />
    </template>
  </div>
</template>

<script setup lang="ts">
/**
 * The answer's prose, with each anchored figure rendered at the point it was anchored.
 *
 * The backend leaves `<!--figure:{asset_id}-->` where the model wrote `[FIGURE n]`, so
 * the picture belongs inside the flow of the answer rather than in a block underneath
 * it. Splitting here is what places it: each stretch of prose is parsed on its own and
 * the matching asset is rendered between two of them.
 *
 * An anchor with no matching asset renders NOTHING, and that is load-bearing twice over.
 * It is the client half of "an invented marker costs nothing" — the backend deletes a
 * marker it cannot resolve, and anything that slips past is silent here too. It also
 * covers the gap during streaming, where the resolved text arrives on `content_replace`
 * one event before `assets` does: for that moment the anchor has no asset, and the
 * reader sees prose rather than a placeholder that flashes.
 */
import { computed } from 'vue';

import MessageAssets from './MessageAssets.vue';
import type { AssetReference } from '@/types/chat';
import { parseMarkdown, escapeHtml, splitFigureAnchors } from '@/utils/markdown';

const props = defineProps<{
  text: string;
  isUser: boolean;
  msgIndex?: number | null;
  assets?: AssetReference[];
}>();

const emit = defineEmits<{
  (e: 'cite-click', msgIndex: number, chunkIndex: number): void;
}>();

type RenderedPart =
  | { kind: 'prose'; html: string }
  | { kind: 'figure'; asset: AssetReference | null };

const parts = computed<RenderedPart[]>(() => {
  // A user's own message is escaped, never parsed, and cannot carry an anchor.
  if (props.isUser) {
    return [{ kind: 'prose', html: escapeHtml(props.text) }];
  }
  const byId = new Map((props.assets || []).map((asset) => [asset.asset_id, asset]));
  return splitFigureAnchors(props.text).map((part) =>
    part.kind === 'prose'
      ? { kind: 'prose' as const, html: parseMarkdown(part.text, props.msgIndex) }
      : { kind: 'figure' as const, asset: byId.get(part.assetId) || null },
  );
});

const onContentClick = (e: MouseEvent) => {
  const citeRef = (e.target as HTMLElement).closest('.cite-ref');
  if (!citeRef) return;
  const msgIndexStr = citeRef.getAttribute('data-msg-index');
  const chunkIndexStr = citeRef.getAttribute('data-chunk-index');
  if (msgIndexStr !== null && chunkIndexStr !== null) {
    emit('cite-click', Number(msgIndexStr), Number(chunkIndexStr));
  }
};
</script>

<style scoped>
/* An anchored figure sits in the flow of the prose, so it needs room underneath as well
   as above. MessageAssets carries only `margin-top`, which is all the trailing block at
   the end of a message ever needed. Scoped, so that trailing block is untouched. */
.message-parts :deep(.message-assets) {
  margin-bottom: 12px;
}
</style>
