/**
 * What an assistant answer is allowed to render.
 *
 * The backend never sends markup. A figure reaches the UI as a structured
 * `AssetReference` and is rendered by `MessageAssets`, which fetches the bytes with the
 * Bearer token an `<img>` cannot carry. So an image inside the answer TEXT can only have
 * been written by the model, and it can only ever break: the href is guessed — usually the
 * `asset_id` the knowledge tool showed it so `view_figure` would be callable — it is
 * loaded with no `Authorization` header, and on a UI with its own domain it resolves
 * against the static server rather than the API.
 *
 * The symptom is specific and was worth a day: a broken-image glyph showing the caption of
 * the very picture the system was about to render properly one element below it.
 *
 * These run the real renderer rather than reading the source. `parseMarkdown` touches no
 * DOM (only `escapeHtml` does), so node can execute it.
 */
import { describe, expect, it } from 'vitest';

import { parseMarkdown } from './markdown';

/** The shape the deployed model actually produced, asset_id and all. */
const REAL_ANSWER =
  'تقدر تشوف الصورة هنا:\n\n' +
  '![Sports Wear: All Grades - Unisex](/media/BHCR Knowledge Base-Jan 2026 (4).docx::p0::img5) [2]';

describe('images the model wrote are never rendered', () => {
  it('drops the markdown image that produced the broken glyph', () => {
    const html = parseMarkdown(REAL_ANSWER, 0);
    expect(html).not.toContain('<img');
  });

  it('keeps the words around it', () => {
    expect(parseMarkdown(REAL_ANSWER, 0)).toContain('تقدر تشوف الصورة هنا');
  });

  it('still turns the citation beside it into a clickable marker', () => {
    expect(parseMarkdown(REAL_ANSWER, 0)).toContain('data-chunk-index="2"');
  });

  it.each([
    ['a bare relative path', '![Uniform](/media/kb.docx::p0::img5)'],
    ['a bare asset id', '![Uniform](kb.docx::p0::img5)'],
    ['an absolute API url', '![Uniform](https://api.example.com/media/kb.docx::p0::img5)'],
    ['a title attribute', '![Uniform](/media/x "The PE kit")'],
    ['an inline image mid-sentence', 'The kit ![kit](/media/x) is navy blue.'],
    ['a reference-style image', '![Uniform][ref]\n\n[ref]: /media/x'],
    ['a data uri', '![Uniform](data:image/png;base64,iVBORw0KGgo=)'],
  ])('drops %s', (_label, markdown) => {
    expect(parseMarkdown(markdown, 0)).not.toContain('<img');
  });

  it('drops a raw HTML image, which is the same failure by another route', () => {
    const html = parseMarkdown('<img src="/media/kb.docx::p0::img5" alt="Uniform">', 0);
    expect(html).not.toContain('<img');
  });

  it('drops a tag carrying an event handler, since this html goes into v-html', () => {
    const html = parseMarkdown('<img src=x onerror="alert(1)">', 0);
    expect(html).not.toContain('onerror');
  });
});

describe('everything else about an answer still renders', () => {
  it('renders emphasis', () => {
    expect(parseMarkdown('The kit is **navy blue**.', 0)).toContain('<strong>navy blue</strong>');
  });

  it('renders a bulleted list, which is the shape most answers take', () => {
    const html = parseMarkdown('- Sports shirt\n- Hoodie', 0);
    expect(html).toContain('<li>');
    expect(html).toContain('Hoodie');
  });

  it('renders a link, which is not an image and is safe to follow', () => {
    expect(parseMarkdown('[the office](https://school.example.com)', 0)).toContain('<a href=');
  });

  it('highlights a fenced code block', () => {
    expect(parseMarkdown('```js\nconst a = 1;\n```', 0)).toContain('class="hljs language-js"');
  });

  it('leaves a citation-shaped number inside code alone', () => {
    expect(parseMarkdown('`arr[1]`', 0)).not.toContain('cite-ref');
  });

  it('marks citations up when a message index is given, and not otherwise', () => {
    expect(parseMarkdown('Navy blue. [1]', 3)).toContain('data-msg-index="3"');
    expect(parseMarkdown('Navy blue. [1]')).not.toContain('cite-ref');
  });
});
