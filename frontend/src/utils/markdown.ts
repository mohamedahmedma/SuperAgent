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
