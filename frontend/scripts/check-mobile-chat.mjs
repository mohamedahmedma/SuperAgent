// Start Vite on the port below first:  npx vite --port 3100 --strictPort
// Then:  node scripts/check-mobile-chat.mjs <path to playwright/index.js>
// Playwright is not a dependency here, so pass its entry file (not its directory).
// Simulates VisualViewport events; a physical iPhone is still needed for OS keyboard QA.
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const playwright = await import(process.argv[2] ? pathToFileURL(process.argv[2]).href : 'playwright');
// The npm package is CJS: named exports are not always visible through ESM import.
const { chromium } = playwright.chromium ? playwright : playwright.default;
const browser = await chromium.launch({ headless: true });
try {
  for (const size of [{ width: 375, height: 667 }, { width: 390, height: 844 }, { width: 844, height: 390 }]) {
    const page = await browser.newPage({ viewport: size, isMobile: true, hasTouch: true });
    page.on('pageerror', error => console.error('Browser error:', error.message));
    page.on('requestfailed', request => console.error('Request failed:', request.url(), request.failure()?.errorText));
    await page.addInitScript(() => {
      localStorage.setItem('superagent-theme-v2', 'dark');
      localStorage.setItem('superagent-language', 'ar');
      const vv = new EventTarget();
      Object.assign(vv, { height: innerHeight, width: innerWidth, offsetTop: 0, offsetLeft: 0, scale: 1 });
      Object.defineProperty(window, 'visualViewport', { value: vv });
      window.testViewport = (height, offsetTop = 0, scale = 1) => {
        Object.assign(vv, { height, offsetTop, scale });
        vv.dispatchEvent(new Event('resize'));
        vv.dispatchEvent(new Event('scroll'));
      };
    });
    await page.route(/\/sessions(?:[/?]|$)/, route => route.fulfill({ json: [] }));
    await page.goto('http://127.0.0.1:3100', { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('.ax-auth-root', { state: 'attached' });
    await page.evaluate(height => window.testViewport(height), size.height);
    await page.evaluate(async () => {
      const { useAuthStore } = await import('/src/stores/auth.ts');
      const { useChatStore } = await import('/src/stores/chat.ts');
      window.testChat = useChatStore();
      useAuthStore().$patch({ token: 'layout-test-only', currentUser: { username: 'layout-test', role: 'parent' } });
    });
    await page.waitForSelector('.aurexis-chat-viewport .chat-area');
    const keyboardHeight = Math.round(size.height * .55);
    const settle = () => page.waitForTimeout(150);
    const metrics = () => page.evaluate(() => {
      const rect = selector => {
        const r = document.querySelector(selector).getBoundingClientRect();
        return { top: r.top, bottom: r.bottom, height: r.height };
      };
      const list = document.querySelector('.chat-container');
      return { shell: rect('.app-page'), header: rect('.chat-header'), list: rect('.chat-container'),
        composer: rect('.input-area-wrapper'), scrollTop: list.scrollTop, scrollHeight: list.scrollHeight,
        clientHeight: list.clientHeight, overflow: getComputedStyle(list).overflowY,
        padding: parseFloat(getComputedStyle(list).paddingBottom), input: rect('.input-area'),
        inputStyle: ['position','display','height','transform'].map(key => [key,getComputedStyle(document.querySelector('.input-area'))[key]]) };
    });
    const checkLayout = async (height, top) => {
      const m = await metrics();
      assert(Math.abs(m.shell.top - top) <= 1, JSON.stringify(m));
      assert(Math.abs(m.shell.height - height) <= 1, JSON.stringify(m));
      assert(m.list.height > 20, JSON.stringify(m));
      assert(m.list.bottom <= m.composer.top + 1, 'composer overlaps messages');
      assert(m.composer.height >= 64, 'composer does not reserve space for its input');
      assert(m.input.top >= m.composer.top, 'input escapes composer and covers messages');
      assert(Math.abs(m.composer.bottom - (top + height)) <= 1, JSON.stringify(m));
      assert(m.padding >= 24, 'missing bottom message spacing');
      assert.equal(m.overflow, 'auto');
    };
    await settle();
    await checkLayout(size.height, 0);
    await page.evaluate(() => {
      window.testChat.messages = Array.from({ length: 15 }, (_, i) => ({
        isUser: false, text: `رسالة ${i + 1}\n\n` + 'معلومات المدرسة وجدول الحصص. '.repeat(35),
      }));
    });
    await settle();
    await page.locator('.chat-input-textarea').focus();
    await page.evaluate(height => window.testViewport(height, 130), keyboardHeight);
    await settle();
    await checkLayout(keyboardHeight, 130);
    // Scroll upwards while focused; streaming must not pull the reader down again.
    await page.evaluate(() => { document.querySelector('.chat-container').scrollTop -= 350; });
    await settle();
    const before = (await metrics()).scrollTop;
    await page.evaluate(() => { window.testChat.messages.at(-1).text += '\n\n' + 'رد جديد '.repeat(70); });
    await settle();
    const afterStream = await metrics();
    assert(Math.abs(afterStream.scrollTop - before) <= 2, `stream stole the reading position: ${before} -> ${JSON.stringify(afterStream)}`);
    // A real browser wheel event must still scroll the message container.
    const m = await metrics();
    await page.mouse.move(size.width / 2, m.list.top + 15);
    await page.mouse.wheel(0, -100);
    await settle();
    assert((await metrics()).scrollTop < before, 'message container cannot scroll');
    await page.evaluate(() => {
      document.querySelector('.chat-input-textarea').blur();
      window.testChat.isLoading = true;
      window.testChat.streamingSessionId = window.testChat.sessionId;
      window.testViewport(window.visualViewport.height, 60);
    });
    await settle();
    await checkLayout(keyboardHeight, 60);
    await page.evaluate(height => window.testViewport(height, 0), size.height);
    await settle();
    await checkLayout(size.height, 0);
    // Closing then reopening the keyboard must work repeatedly.
    await page.evaluate(height => window.testViewport(height, 80), keyboardHeight);
    await settle();
    await checkLayout(keyboardHeight, 80);
    await page.evaluate(() => { document.documentElement.dataset.theme = 'light'; });
    await checkLayout(keyboardHeight, 80);
    await page.evaluate(() => { window.testChat.activeNav = 'settings'; });
    await settle();
    assert.equal(await page.locator('html').evaluate(el => el.classList.contains('aurexis-chat-viewport')), false);
    console.log(`PASS ${size.width}x${size.height}: viewport offsets, spacing, focused scrolling, streaming, reopen, light mode, cleanup`);
    await page.close();
  }
} finally { await browser.close(); }
