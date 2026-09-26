<template>
  <component :is="renderers[block.kind]" :block="block" />
</template>

<script setup lang="ts">
/**
 * One record a tool rendered for the reader, drawn by the component registered for its
 * kind.
 *
 * This file is the whole of the frontend's half of a new kind: a data type in
 * types/chat.ts, a reader in utils/answerBlocks.ts, and a component registered below.
 * `readAnswerBlocks` lets through only kinds it has a reader for, and the
 * `Record<AnswerBlockKind, …>` here makes the compiler refuse a kind that has a type but
 * no component — so a block that reaches this point always has something to draw it.
 * Anything else stays the markdown it also travels as.
 */
import type { Component } from 'vue';

import type { AnswerBlock, AnswerBlockKind } from '@/types/chat';
import GradesBlock from './GradesBlock.vue';
import TimetableBlock from './TimetableBlock.vue';
import './answerBlock.css';

defineProps<{ block: AnswerBlock }>();

const renderers: Record<AnswerBlockKind, Component> = {
  timetable: TimetableBlock,
  grades: GradesBlock,
};
</script>
