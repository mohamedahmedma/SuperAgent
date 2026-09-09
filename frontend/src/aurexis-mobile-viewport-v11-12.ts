/*
 * AUREXIS mobile visual viewport runtime v11.12
 * Detects the software keyboard without changing desktop behavior.
 */
const installAurexisMobileViewport = () => {
  if (typeof window === 'undefined' || typeof document === 'undefined') return;
  if (document.documentElement.dataset.aurexisViewportV1112 === '1') return;

  document.documentElement.dataset.aurexisViewportV1112 = '1';

  const root = document.documentElement;
  const viewport = window.visualViewport;
  let baseline = Math.max(window.innerHeight, viewport?.height ?? 0);

  const update = () => {
    const vvHeight = viewport?.height ?? window.innerHeight;
    const mobile = window.matchMedia('(max-width: 900px)').matches;

    if (!mobile) {
      root.classList.remove('aurexis-keyboard-open');
      root.style.removeProperty('--aurexis-visual-height');
      baseline = Math.max(window.innerHeight, vvHeight);
      return;
    }

    const active = document.activeElement;
    const editing =
      active instanceof HTMLTextAreaElement ||
      active instanceof HTMLInputElement ||
      (active instanceof HTMLElement && active.isContentEditable);

    if (!editing) baseline = Math.max(baseline, window.innerHeight, vvHeight);

    const keyboardDelta = baseline - vvHeight;
    const keyboardOpen = editing && keyboardDelta > 120;

    root.style.setProperty('--aurexis-visual-height', `${Math.round(vvHeight)}px`);
    root.classList.toggle('aurexis-keyboard-open', keyboardOpen);
  };

  const deferredUpdate = () => {
    requestAnimationFrame(update);
    window.setTimeout(update, 80);
    window.setTimeout(update, 240);
  };

  viewport?.addEventListener('resize', deferredUpdate, { passive: true });
  viewport?.addEventListener('scroll', deferredUpdate, { passive: true });
  window.addEventListener('resize', deferredUpdate, { passive: true });
  window.addEventListener('orientationchange', () => {
    baseline = 0;
    window.setTimeout(() => {
      baseline = Math.max(window.innerHeight, viewport?.height ?? 0);
      update();
    }, 350);
  }, { passive: true });

  document.addEventListener('focusin', deferredUpdate);
  document.addEventListener('focusout', () => {
    window.setTimeout(() => {
      root.classList.remove('aurexis-keyboard-open');
      baseline = Math.max(window.innerHeight, viewport?.height ?? 0);
      update();
    }, 180);
  });

  update();
};

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', installAurexisMobileViewport, { once: true });
} else {
  installAurexisMobileViewport();
}
