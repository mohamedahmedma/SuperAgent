/*
 AUREXIS Mobile VisualViewport Controller v11.16
 Keeps the mobile chat layout stable while the software keyboard is open.
*/

const installAurexisKeyboardLayoutV1116 = () => {
  if (typeof window === 'undefined' || typeof document === 'undefined') return

  const root = document.documentElement
  if (root.dataset.aurexisKeyboardLayoutV1116 === '1') return
  root.dataset.aurexisKeyboardLayoutV1116 = '1'

  const vv = window.visualViewport
  let stableLayoutHeight = Math.max(
    document.documentElement.clientHeight || 0,
    window.innerHeight || 0,
    vv?.height || 0,
  )

  const isPhone = () => window.matchMedia('(max-width: 900px)').matches

  const isEditing = () => {
    const active = document.activeElement
    return (
      active instanceof HTMLTextAreaElement ||
      active instanceof HTMLInputElement ||
      (active instanceof HTMLElement && active.isContentEditable)
    )
  }

  const visibleBottom = () => {
    const top = vv?.offsetTop || 0
    const height = vv?.height || window.innerHeight
    return top + height
  }

  const measureHeaderBottom = () => {
    const selectors = [
      '.mobile-chat-header',
      '.mobile-topbar',
      '.mobile-top-bar',
      '.chat-header',
    ]

    for (const selector of selectors) {
      const el = document.querySelector<HTMLElement>(selector)
      if (!el) continue

      const style = window.getComputedStyle(el)
      if (style.display === 'none' || style.visibility === 'hidden') continue

      const rect = el.getBoundingClientRect()
      if (rect.height < 28 || rect.height > 140) continue
      if (rect.bottom <= 0 || rect.top > 160) continue

      return Math.round(rect.bottom + 8)
    }

    return 78
  }

  const measureComposerHeight = () => {
    const composer = document.querySelector<HTMLElement>('.input-area-wrapper')
    if (!composer) return 82

    const rect = composer.getBoundingClientRect()
    const measured = Math.round(rect.height)

    if (measured >= 56 && measured <= 120) return measured + 2
    return 82
  }

  const update = () => {
    if (!isPhone()) {
      root.classList.remove('aurexis-keyboard-open')
      root.style.removeProperty('--aurexis-vv-bottom')
      root.style.removeProperty('--aurexis-visible-content-top')
      root.style.removeProperty('--aurexis-composer-height')
      root.style.removeProperty('--aurexis-composer-reserved')

      stableLayoutHeight = Math.max(
        document.documentElement.clientHeight || 0,
        window.innerHeight || 0,
        vv?.height || 0,
      )
      return
    }

    const visualHeight = vv?.height || window.innerHeight
    const currentLayoutHeight = Math.max(
      document.documentElement.clientHeight || 0,
      window.innerHeight || 0,
    )

    if (!isEditing()) {
      stableLayoutHeight = Math.max(
        stableLayoutHeight,
        currentLayoutHeight,
        visualHeight,
      )
    }

    const keyboardDelta = stableLayoutHeight - visualHeight
    const keyboardOpen = isEditing() && keyboardDelta > 120

    root.classList.toggle('aurexis-keyboard-open', keyboardOpen)

    if (!keyboardOpen) return

    const composerHeight = measureComposerHeight()
    const headerBottom = measureHeaderBottom()
    const bottom = Math.round(visibleBottom())

    root.style.setProperty('--aurexis-vv-bottom', `${bottom}px`)
    root.style.setProperty('--aurexis-visible-content-top', `${headerBottom}px`)
    root.style.setProperty('--aurexis-composer-height', `${composerHeight}px`)
    root.style.setProperty(
      '--aurexis-composer-reserved',
      `${composerHeight + 18}px`,
    )
  }

  const refresh = () => {
    requestAnimationFrame(update)
    window.setTimeout(update, 40)
    window.setTimeout(update, 120)
    window.setTimeout(update, 260)
  }

  vv?.addEventListener('resize', refresh, { passive: true })
  vv?.addEventListener('scroll', refresh, { passive: true })

  window.addEventListener('resize', refresh, { passive: true })

  window.addEventListener(
    'orientationchange',
    () => {
      root.classList.remove('aurexis-keyboard-open')
      stableLayoutHeight = 0

      window.setTimeout(() => {
        stableLayoutHeight = Math.max(
          document.documentElement.clientHeight || 0,
          window.innerHeight || 0,
          vv?.height || 0,
        )
        refresh()
      }, 360)
    },
    { passive: true },
  )

  document.addEventListener('focusin', refresh)

  document.addEventListener('focusout', () => {
    window.setTimeout(() => {
      if (!isEditing()) {
        root.classList.remove('aurexis-keyboard-open')
      }
      refresh()
    }, 180)
  })

  update()
}

if (document.readyState === 'loading') {
  document.addEventListener(
    'DOMContentLoaded',
    installAurexisKeyboardLayoutV1116,
    { once: true },
  )
} else {
  installAurexisKeyboardLayoutV1116()
}
