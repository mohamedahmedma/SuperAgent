import type { ChunkInfo, ChunkNode } from '@/types/document';

/**
 * The flat chunk list as the tree the inspector draws.
 *
 * Built in the browser rather than server side because the list view needs the flat form
 * anyway: one shape travels, and the two views cannot then disagree about what came back.
 *
 * A chunk whose parent is not in the list becomes a ROOT rather than disappearing. That
 * is the ordinary case under a filter — the filter matched a leaf and not its parents —
 * and dropping those would show an empty tree for a search that found things. It is also
 * what the whole response looked like before the endpoint learned to read the parent
 * store, so a deployment whose Postgres has no parent chunks still gets a readable list
 * instead of nothing at all.
 *
 * Extracted from the component so it can be tested as logic. The project has no
 * @vue/test-utils, and the alternative convention — reading the component's source and
 * matching it with a regex — cannot tell a working tree from a broken one.
 */
export function buildChunkTree(chunks: ChunkInfo[]): ChunkNode[] {
  const nodes = new Map<string, ChunkNode>();
  // Built in one pass first so a child that appears BEFORE its parent still finds it.
  // Level-then-index ordering happens to put parents first today; relying on that would
  // make the tree depend on how the endpoint happens to sort.
  for (const chunk of chunks) {
    nodes.set(chunk.chunk_id, { ...chunk, children: [] });
  }

  const roots: ChunkNode[] = [];
  for (const chunk of chunks) {
    const node = nodes.get(chunk.chunk_id)!;
    const parent = chunk.parent_chunk_id ? nodes.get(chunk.parent_chunk_id) : undefined;
    // A chunk naming ITSELF as its parent would otherwise build a cycle and hang the
    // renderer. Nothing writes that today; it costs one comparison to make impossible.
    if (parent && parent !== node) parent.children.push(node);
    else roots.push(node);
  }
  return roots;
}

/** "L1 18 · L2 36 · L3 175" — what the document is made of, at a glance. */
export function levelSummary(chunks: ChunkInfo[]): string {
  const counts = new Map<number, number>();
  for (const chunk of chunks) {
    counts.set(chunk.chunk_level, (counts.get(chunk.chunk_level) || 0) + 1);
  }
  return [...counts.entries()]
    .sort(([a], [b]) => a - b)
    .map(([level, count]) => `L${level} ${count}`)
    .join(' · ');
}
