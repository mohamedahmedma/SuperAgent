import { marked } from 'marked';
import hljs from 'highlight.js';

// Customize the code renderer in marked for syntax highlighting
const renderer = new marked.Renderer();
renderer.code = (code, language) => {
  const validLanguage = language && hljs.getLanguage(language) ? language : 'plaintext';
  const highlighted = hljs.highlight(code, { language: validLanguage }).value;
  return `<pre><code class="hljs language-${validLanguage}">${highlighted}</code></pre>`;
};

/**
 * Images written by the MODEL are dropped.
 *
 * The backend never sends markup: a figure reaches the UI as a structured
 * AssetReference and is rendered by MessageAssets, which fetches the bytes with the
 * Bearer token an `<img>` cannot carry. So an image in this text can only have been
 * invented by the model, and it can only ever break — the URL is guessed (often the
 * bare asset_id it was shown for view_figure), it is loaded with no Authorization
 * header, and on a split UI/API deployment it resolves against the UI's own origin.
 * What the user sees is a broken-image glyph sitting next to the caption of a picture
 * the system was about to render properly one element further down.
 *
 * Dropping it costs nothing and closes the other half of the problem: this HTML goes
 * straight into `v-html`, so model-authored tags are worth admitting one at a time.
 */
renderer.image = () => '';

/**
 * Raw HTML written by the model is dropped too, tags only — the text inside them is a
 * separate token and survives, so `<b>bold</b>` still reads "bold".
 *
 * `<img src=...>` is the same failure as above by a second route, and marked stops
 * sanitizing anything as of v5: without this, every tag a model emits reaches `v-html`
 * verbatim. Answers are markdown by contract (the profile prompt says so and nothing
 * downstream renders HTML), so there is nothing here to lose and an `onerror=` to
 * avoid.
 */
renderer.html = () => '';

/**
 * A figure anchor, and the answer split around them.
 *
 * `_resolve_figure_markers` in backend/chat/service.py turns each `[FIGURE n]` the model
 * wrote into `<!--figure:{asset_id}-->` at that point in the prose, so the picture can be
 * rendered where the answer put it rather than as a card underneath the whole message.
 *
 * The comment form is the reason this is safe: `renderer.html` above drops it, so a
 * client that knows nothing about anchors shows clean prose and still gets the pictures
 * from the trailing block. Nothing breaks; the placement is just lost.
 *
 * A capturing `split` rather than `matchAll`: with one group it alternates prose, id,
 * prose, id, prose — every odd index is a capture — which needs no iterator support.
 */
export type AnswerPart =
  | { kind: 'prose'; text: string }
  | { kind: 'figure'; assetId: string };

export function splitFigureAnchors(text: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  (text || '').split(/<!--figure:(.+?)-->/).forEach((piece, index) => {
    if (index % 2 === 1) {
      const assetId = (piece || '').trim();
      if (assetId) parts.push({ kind: 'figure', assetId });
    } else if (piece) {
      parts.push({ kind: 'prose', text: piece });
    }
  });
  return parts;
}

/**
 * Where each tool-rendered record starts in an answer.
 *
 * `_settle_answer_blocks` in backend/chat/service.py appends every record under the prose,
 * each on the line after `<!--record-block-->`. The same record may also have arrived as
 * data (an `AnswerBlock`) naming its marker by `index` — the marker's position here,
 * counted from 0 — in which case it is drawn in that place instead of printed.
 *
 * Like a figure anchor, the marker is an HTML comment that `renderer.html` drops, so a
 * client that never splits on it shows the record as markdown and loses nothing.
 */
export const RECORD_BLOCK_MARKER = '<!--record-block-->';

export type AnswerSegment =
  | { kind: 'prose'; text: string }
  | { kind: 'record'; index: number; text: string };

export function splitRecordBlocks(text: string): AnswerSegment[] {
  const [prose, ...recordTexts] = (text || '').split(RECORD_BLOCK_MARKER);
  const segments: AnswerSegment[] = [];
  if (prose.trim()) segments.push({ kind: 'prose', text: prose });
  recordTexts.forEach((record, index) => {
    segments.push({ kind: 'record', index, text: record.replace(/^\r?\n/, '') });
  });
  return segments;
}

/** The asset ids this answer anchored, in order, deduped. */
export function figureAnchorIds(text: string): string[] {
  const ids: string[] = [];
  splitFigureAnchors(text).forEach((part) => {
    if (part.kind === 'figure' && !ids.includes(part.assetId)) ids.push(part.assetId);
  });
  return ids;
}

marked.use({
  renderer,
  breaks: true,
  gfm: true
});

export function parseMarkdown(text: string, msgIndex?: number | null): string {
  const html = marked.parse(text || '', { async: false }) as string;

  if (msgIndex === undefined || msgIndex === null) {
    return html;
  }

  let inCode = false;
  return html.split(/(<[^>]*>)/).map(part => {
    if (part.startsWith('<')) {
      if (part.startsWith('<code') || part.startsWith('<pre')) inCode = true;
      if (part.startsWith('</code') || part.startsWith('</pre')) inCode = false;
      return part;
    }
    if (!inCode) {
      return part.replace(/\[([\d\s,]+)\]/g, (match: string, p1: string) => {
        const numbers = p1.split(',').map((n: string) => n.trim()).filter((n: string) => /^\d+$/.test(n));
        if (numbers.length === 0) return match;
        return numbers.map(
          (n: string) => `<sup class="cite-ref" data-msg-index="${msgIndex}" data-chunk-index="${n}">[${n}]</sup>`
        ).join('');
      });
    }
    return part;
  }).join('');
}

export function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
