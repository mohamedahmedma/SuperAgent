/*
 * One source of truth for the mobile software keyboard.
 *
 * Some mobile browsers keep the keyboard visible after a textarea is disabled
 * (for example while an answer is streaming).  Focus-based detection then
 * removes the keyboard layout too early.  We detect the keyboard exclusively
 * from the VisualViewport height, which remains correct for that transition.
 */
const installAurexisMobileKeyboard = () => {
  if (typeof window === 'undefined' || typeof document === 'undefined') return

  const root = document.documentElement
  if (root.dataset.aurexisMobileKeyboard === '1') return
  root.dataset.aurexisMobileKeyboard = '1'

  const viewport = window.visualViewport
  let expandedHeight = Math.max(
    window.innerHeight,
    document.documentElement.clientHeight,
    viewport?.height || 0,
  )
  let timer: number | undefined

  const isPhone = () => window.matchMedia('(max-width: 900px)').matches

  const update = () => {
    const visualHeight = Math.round(viewport?.height || window.innerHeight)
    const layoutHeight = Math.max(
      window.innerHeight,
      document.documentElement.clientHeight,
      visualHeight,
    )

    // Only grow the baseline. While the keyboard is displayed some browsers
    // shrink innerHeight too, so the previous full viewport is intentional.
    expandedHeight = Math.max(expandedHeight, layoutHeight)
    const keyboardOpen = isPhone() && expandedHeight - visualHeight > 120

    root.style.setProperty('--aurexis-visual-height', `${visualHeight}px`)
    root.classList.toggle('aurexis-keyboard-open', keyboardOpen)
  }

  const schedule = () => {
    window.clearTimeout(timer)
    requestAnimationFrame(update)
    timer = window.setTimeout(update, 100)
  }

  viewport?.addEventListener('resize', schedule, { passive: true })
  viewport?.addEventListener('scroll', schedule, { passive: true })
  window.addEventListener('resize', schedule, { passive: true })
  window.addEventListener('orientationchange', () => {
    expandedHeight = 0
    window.setTimeout(() => {
      expandedHeight = Math.max(
        window.innerHeight,
        document.documentElement.clientHeight,
        viewport?.height || 0,
      )
      update()
    }, 300)
  }, { passive: true })

  update()
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', installAurexisMobileKeyboard, { once: true })
} else {
  installAurexisMobileKeyboard()
}
