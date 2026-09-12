<template>
  <div class="message-parts" @click="onContentClick">
    <template v-for="(part, index) in parts" :key="index">
      <div
        v-if="part.kind === 'prose'"
        class="message-content"
        v-html="part.html"
      ></div>
      <AnswerBlockView v-else-if="part.kind === 'block'" :block="part.block" />
      <MessageAssets v-else-if="part.kind === 'figure' && part.asset" :assets="[part.asset]" />
    </template>
  </div>
</template>

<script setup lang="ts">
/**
 * The answer's prose, with each anchored figure rendered at the point it was anchored, and
 * each record a tool rendered drawn in the place it was appended.
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
 *
 * A record — a timetable, a term's marks — follows the same idea from the other side. The
 * backend appends each one under the prose after `<!--record-block-->`, and may send the
 * same record as data naming that marker's index. With the data, the record is DRAWN
 * there; without it — an older server, a kind this client does not draw, a block that
 * failed its check — the text after the marker is printed as markdown, exactly as before.
 * Either way the reader has the record; the data only changes how it looks.
 */
import { computed } from 'vue';

import AnswerBlockView from './blocks/AnswerBlock.vue';
import MessageAssets from './MessageAssets.vue';
import type { AnswerBlock, AssetReference } from '@/types/chat';
import { parseMarkdown, escapeHtml, splitFigureAnchors, splitRecordBlocks } from '@/utils/markdown';

const props = defineProps<{
  text: string;
  isUser: boolean;
  msgIndex?: number | null;
  assets?: AssetReference[];
  answerBlocks?: AnswerBlock[];
}>();

const emit = defineEmits<{
  (e: 'cite-click', msgIndex: number, chunkIndex: number): void;
}>();

type RenderedPart =
  | { kind: 'prose'; html: string }
  | { kind: 'figure'; asset: AssetReference | null }
  | { kind: 'block'; block: AnswerBlock };

const parts = computed<RenderedPart[]>(() => {
  // A user's own message is escaped, never parsed, and cannot carry an anchor.
  if (props.isUser) {
    return [{ kind: 'prose', html: escapeHtml(props.text) }];
  }
  const assetsById = new Map((props.assets || []).map((asset) => [asset.asset_id, asset]));
  const blocksByIndex = new Map((props.answerBlocks || []).map((block) => [block.index, block]));
  return splitRecordBlocks(props.text).flatMap((segment): RenderedPart[] => {
    if (segment.kind === 'record') {
      const block = blocksByIndex.get(segment.index);
      return block
        ? [{ kind: 'block', block }]
        : [{ kind: 'prose', html: parseMarkdown(segment.text, props.msgIndex) }];
    }
    return splitFigureAnchors(segment.text).map(
      (part): RenderedPart =>
        part.kind === 'prose'
          ? { kind: 'prose', html: parseMarkdown(part.text, props.msgIndex) }
          : { kind: 'figure', asset: assetsById.get(part.assetId) || null },
    );
  });
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
