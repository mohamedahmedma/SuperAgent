/**
 * The tree the inspector draws, as logic rather than as markup.
 *
 * The structure is the claim the view makes: three levels, each leaf under the section it
 * was split from. Getting it wrong is not a cosmetic bug — a flat list presented as a
 * hierarchy tells an admin the corpus is shaped in a way it is not, which is exactly the
 * mistake the endpoint itself made until it learned to read the parent store.
 */
import { describe, expect, it } from 'vitest';
import { buildChunkTree, levelSummary } from './chunkTree';
import type { ChunkInfo } from '@/types/document';

const chunk = (overrides: Partial<ChunkInfo> & { chunk_id: string }): ChunkInfo => ({
  parent_chunk_id: '',
  root_chunk_id: '',
  chunk_level: 3,
  chunk_idx: 0,
  page_number: 1,
  modality: 'text',
  text: '',
  char_count: 0,
  asset_ids: [],
  matched: false,
  ...overrides,
});

const ids = (nodes: { chunk_id: string }[]) => nodes.map((n) => n.chunk_id);

describe('buildChunkTree', () => {
  it('nests each level under the one above it', () => {
    const tree = buildChunkTree([
      chunk({ chunk_id: 'L1', chunk_level: 1 }),
      chunk({ chunk_id: 'L2', chunk_level: 2, parent_chunk_id: 'L1' }),
      chunk({ chunk_id: 'L3', chunk_level: 3, parent_chunk_id: 'L2' }),
    ]);

    expect(ids(tree)).toEqual(['L1']);
    expect(ids(tree[0].children)).toEqual(['L2']);
    expect(ids(tree[0].children[0].children)).toEqual(['L3']);
  });

  it('keeps every sibling, in the order it arrived', () => {
    const tree = buildChunkTree([
      chunk({ chunk_id: 'L1', chunk_level: 1 }),
      chunk({ chunk_id: 'a', parent_chunk_id: 'L1', chunk_idx: 0 }),
      chunk({ chunk_id: 'b', parent_chunk_id: 'L1', chunk_idx: 1 }),
      chunk({ chunk_id: 'c', parent_chunk_id: 'L1', chunk_idx: 2 }),
    ]);

    expect(ids(tree[0].children)).toEqual(['a', 'b', 'c']);
  });

  it('finds a parent that appears after its child', () => {
    // The endpoint sorts parents first today. Depending on that would make the tree
    // break the day anything re-sorted the response.
    const tree = buildChunkTree([
      chunk({ chunk_id: 'L3', parent_chunk_id: 'L1' }),
      chunk({ chunk_id: 'L1', chunk_level: 1 }),
    ]);

    expect(ids(tree)).toEqual(['L1']);
    expect(ids(tree[0].children)).toEqual(['L3']);
  });

  it('treats a chunk whose parent is absent as a root', () => {
    // The ordinary case under a filter: the leaf matched and its parents did not.
    // Dropping it would show an empty tree for a search that found something.
    const tree = buildChunkTree([chunk({ chunk_id: 'orphan', parent_chunk_id: 'missing' })]);

    expect(ids(tree)).toEqual(['orphan']);
  });

  it('shows every leaf when no parent chunk is stored at all', () => {
    // What the whole response looked like before the endpoint read the parent store, and
    // what a deployment with an empty parent_chunks table still gets.
    const tree = buildChunkTree([
      chunk({ chunk_id: 'a', parent_chunk_id: 'gone' }),
      chunk({ chunk_id: 'b', parent_chunk_id: 'gone' }),
    ]);

    expect(ids(tree)).toEqual(['a', 'b']);
  });

  it('does not build a cycle out of a chunk that names itself', () => {
    const tree = buildChunkTree([chunk({ chunk_id: 'self', parent_chunk_id: 'self' })]);

    expect(ids(tree)).toEqual(['self']);
    expect(tree[0].children).toEqual([]);
  });

  it('carries every field through to the node', () => {
    const tree = buildChunkTree([
      chunk({ chunk_id: 'c1', text: 'fees', char_count: 4, asset_ids: ['img-1'], modality: 'figure' }),
    ]);

    expect(tree[0].text).toBe('fees');
    expect(tree[0].char_count).toBe(4);
    expect(tree[0].asset_ids).toEqual(['img-1']);
    expect(tree[0].modality).toBe('figure');
  });

  it('does not mutate what it was given', () => {
    const chunks = [chunk({ chunk_id: 'L1', chunk_level: 1 })];
    buildChunkTree(chunks);

    expect('children' in chunks[0]).toBe(false);
  });

  it('handles several roots', () => {
    const tree = buildChunkTree([
      chunk({ chunk_id: 'A', chunk_level: 1 }),
      chunk({ chunk_id: 'B', chunk_level: 1 }),
      chunk({ chunk_id: 'a1', parent_chunk_id: 'A' }),
    ]);

    expect(ids(tree)).toEqual(['A', 'B']);
    expect(ids(tree[0].children)).toEqual(['a1']);
    expect(tree[1].children).toEqual([]);
  });

  it('handles an empty document', () => {
    expect(buildChunkTree([])).toEqual([]);
  });

  it('holds a deep chain without losing the bottom of it', () => {
    const deep = Array.from({ length: 40 }, (_, i) =>
      chunk({ chunk_id: `n${i}`, parent_chunk_id: i ? `n${i - 1}` : '' }),
    );

    let node = buildChunkTree(deep)[0];
    let depth = 1;
    while (node.children.length) {
      node = node.children[0];
      depth += 1;
    }
    expect(depth).toBe(40);
  });
});

describe('levelSummary', () => {
  it('counts each level in order', () => {
    const summary = levelSummary([
      chunk({ chunk_id: 'a', chunk_level: 3 }),
      chunk({ chunk_id: 'b', chunk_level: 1 }),
      chunk({ chunk_id: 'c', chunk_level: 3 }),
      chunk({ chunk_id: 'd', chunk_level: 2 }),
    ]);

    expect(summary).toBe('L1 1 · L2 1 · L3 2');
  });

  it('is empty for an empty document', () => {
    expect(levelSummary([])).toBe('');
  });
});
