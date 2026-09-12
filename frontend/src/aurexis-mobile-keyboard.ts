/** Safari can pan the visual viewport on input focus: height alone is not enough.
 * Keyboard detection only controls safe-area padding; layout never depends on it.
 */
export function installMobileChatViewport(): () => void {
  const root = document.documentElement;
  const viewport = window.visualViewport;
  const mobile = window.matchMedia('(max-width: 900px)');
  let frame = 0;
  let baseline = window.innerHeight;
  let width = window.innerWidth;

  const clear = () => {
    root.classList.remove('aurexis-chat-viewport', 'aurexis-keyboard-open');
    root.style.removeProperty('--aurexis-visual-height');
    root.style.removeProperty('--aurexis-visual-top');
  };
  const update = () => {
    frame = 0;
    if (!mobile.matches) { clear(); return; }
    // Preserve native pinch zoom; do not resize the app around the zoom.
    if (viewport && Math.abs(viewport.scale - 1) > .02) return;
    const height = viewport?.height ?? window.innerHeight;
    if (Math.abs(window.innerWidth - width) > 80) {
      baseline = window.innerHeight;
      width = window.innerWidth;
    }
    baseline = Math.max(baseline, window.innerHeight, height);
    root.style.setProperty('--aurexis-visual-height', `${height}px`);
    root.style.setProperty('--aurexis-visual-top', `${Math.max(0, viewport?.offsetTop ?? 0)}px`);
    root.classList.add('aurexis-chat-viewport');
    root.classList.toggle('aurexis-keyboard-open', baseline - height > 120);
  };
  const schedule = () => {
    if (!frame) frame = window.requestAnimationFrame(update);
  };
  viewport?.addEventListener('resize', schedule);
  viewport?.addEventListener('scroll', schedule);
  window.addEventListener('resize', schedule);
  window.addEventListener('pageshow', schedule);
  mobile.addEventListener('change', schedule);
  update();
  return () => {
    cancelAnimationFrame(frame);
    viewport?.removeEventListener('resize', schedule);
    viewport?.removeEventListener('scroll', schedule);
    window.removeEventListener('resize', schedule);
    window.removeEventListener('pageshow', schedule);
    mobile.removeEventListener('change', schedule);
    clear();
  };
}
