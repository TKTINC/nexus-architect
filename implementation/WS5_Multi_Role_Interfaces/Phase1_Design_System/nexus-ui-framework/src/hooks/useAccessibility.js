import React from 'react'

// Hook for managing focus trap within modals and dialogs
export const useFocusTrap = (isActive = false) => {
  const containerRef = React.useRef(null)
  const previousActiveElement = React.useRef(null)

  React.useEffect(() => {
    if (!isActive || !containerRef.current) return

    const container = containerRef.current
    const focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    )
    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]

    // Store the previously focused element
    previousActiveElement.current = document.activeElement

    // Focus the first element
    if (firstElement) {
      firstElement.focus()
    }

    const handleTabKey = (e) => {
      if (e.key !== 'Tab') return

      if (e.shiftKey) {
        // Shift + Tab
        if (document.activeElement === firstElement) {
          e.preventDefault()
          lastElement?.focus()
        }
      } else {
        // Tab
        if (document.activeElement === lastElement) {
          e.preventDefault()
          firstElement?.focus()
        }
      }
    }

    container.addEventListener('keydown', handleTabKey)

    return () => {
      container.removeEventListener('keydown', handleTabKey)
      // Restore focus to the previously focused element
      if (previousActiveElement.current) {
        previousActiveElement.current.focus()
      }
    }
  }, [isActive])

  return containerRef
}

// Hook for keyboard navigation
export const useKeyboardNavigation = (options = {}) => {
  const {
    onEscape,
    onEnter,
    onSpace,
    onArrowUp,
    onArrowDown,
    onArrowLeft,
    onArrowRight,
    onHome,
    onEnd,
    preventDefault = true
  } = options

  const handleKeyDown = React.useCallback((e) => {
    const handlers = {
      'Escape': onEscape,
      'Enter': onEnter,
      ' ': onSpace,
      'ArrowUp': onArrowUp,
      'ArrowDown': onArrowDown,
      'ArrowLeft': onArrowLeft,
      'ArrowRight': onArrowRight,
      'Home': onHome,
      'End': onEnd
    }

    const handler = handlers[e.key]
    if (handler) {
      if (preventDefault) {
        e.preventDefault()
      }
      handler(e)
    }
  }, [onEscape, onEnter, onSpace, onArrowUp, onArrowDown, onArrowLeft, onArrowRight, onHome, onEnd, preventDefault])

  return { onKeyDown: handleKeyDown }
}

// Hook for managing ARIA attributes
export const useAria = (options = {}) => {
  const {
    role,
    label,
    labelledBy,
    describedBy,
    expanded,
    selected,
    checked,
    disabled,
    hidden,
    live = 'polite',
    atomic = false
  } = options

  const ariaProps = React.useMemo(() => {
    const props = {}

    if (role) props.role = role
    if (label) props['aria-label'] = label
    if (labelledBy) props['aria-labelledby'] = labelledBy
    if (describedBy) props['aria-describedby'] = describedBy
    if (expanded !== undefined) props['aria-expanded'] = expanded
    if (selected !== undefined) props['aria-selected'] = selected
    if (checked !== undefined) props['aria-checked'] = checked
    if (disabled !== undefined) props['aria-disabled'] = disabled
    if (hidden !== undefined) props['aria-hidden'] = hidden
    if (live) props['aria-live'] = live
    if (atomic) props['aria-atomic'] = atomic

    return props
  }, [role, label, labelledBy, describedBy, expanded, selected, checked, disabled, hidden, live, atomic])

  return ariaProps
}

// Hook for screen reader announcements
export const useScreenReader = () => {
  const [announcements, setAnnouncements] = React.useState([])

  const announce = React.useCallback((message, priority = 'polite') => {
    const id = Date.now()
    setAnnouncements(prev => [...prev, { id, message, priority }])

    // Remove announcement after it's been read
    setTimeout(() => {
      setAnnouncements(prev => prev.filter(a => a.id !== id))
    }, 1000)
  }, [])

  const AnnouncementRegion = React.useCallback(() => (
    <div className="sr-only">
      {announcements.map(({ id, message, priority }) => (
        <div key={id} aria-live={priority} aria-atomic="true">
          {message}
        </div>
      ))}
    </div>
  ), [announcements])

  return { announce, AnnouncementRegion }
}

// Hook for managing reduced motion preferences
export const useReducedMotion = () => {
  const [prefersReducedMotion, setPrefersReducedMotion] = React.useState(false)

  React.useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    setPrefersReducedMotion(mediaQuery.matches)

    const handleChange = (e) => {
      setPrefersReducedMotion(e.matches)
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  return prefersReducedMotion
}

// Hook for managing high contrast preferences
export const useHighContrast = () => {
  const [prefersHighContrast, setPrefersHighContrast] = React.useState(false)

  React.useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-contrast: high)')
    setPrefersHighContrast(mediaQuery.matches)

    const handleChange = (e) => {
      setPrefersHighContrast(e.matches)
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  return prefersHighContrast
}

// Hook for managing color scheme preferences
export const useColorScheme = () => {
  const [colorScheme, setColorScheme] = React.useState('light')

  React.useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    setColorScheme(mediaQuery.matches ? 'dark' : 'light')

    const handleChange = (e) => {
      setColorScheme(e.matches ? 'dark' : 'light')
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  return colorScheme
}

// Hook for managing focus visibility
export const useFocusVisible = () => {
  const [isFocusVisible, setIsFocusVisible] = React.useState(false)
  const [hadKeyboardEvent, setHadKeyboardEvent] = React.useState(false)

  React.useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.metaKey || e.altKey || e.ctrlKey) return
      setHadKeyboardEvent(true)
    }

    const handlePointerDown = () => {
      setHadKeyboardEvent(false)
    }

    const handleFocus = (e) => {
      if (hadKeyboardEvent || e.target.matches(':focus-visible')) {
        setIsFocusVisible(true)
      }
    }

    const handleBlur = () => {
      setIsFocusVisible(false)
    }

    document.addEventListener('keydown', handleKeyDown, true)
    document.addEventListener('mousedown', handlePointerDown, true)
    document.addEventListener('pointerdown', handlePointerDown, true)
    document.addEventListener('touchstart', handlePointerDown, true)
    document.addEventListener('focus', handleFocus, true)
    document.addEventListener('blur', handleBlur, true)

    return () => {
      document.removeEventListener('keydown', handleKeyDown, true)
      document.removeEventListener('mousedown', handlePointerDown, true)
      document.removeEventListener('pointerdown', handlePointerDown, true)
      document.removeEventListener('touchstart', handlePointerDown, true)
      document.removeEventListener('focus', handleFocus, true)
      document.removeEventListener('blur', handleBlur, true)
    }
  }, [hadKeyboardEvent])

  return isFocusVisible
}

// Hook for managing roving tabindex
export const useRovingTabIndex = (items = [], orientation = 'horizontal') => {
  const [activeIndex, setActiveIndex] = React.useState(0)

  const handleKeyDown = React.useCallback((e) => {
    const isHorizontal = orientation === 'horizontal'
    const nextKey = isHorizontal ? 'ArrowRight' : 'ArrowDown'
    const prevKey = isHorizontal ? 'ArrowLeft' : 'ArrowUp'

    switch (e.key) {
      case nextKey:
        e.preventDefault()
        setActiveIndex(prev => (prev + 1) % items.length)
        break
      case prevKey:
        e.preventDefault()
        setActiveIndex(prev => (prev - 1 + items.length) % items.length)
        break
      case 'Home':
        e.preventDefault()
        setActiveIndex(0)
        break
      case 'End':
        e.preventDefault()
        setActiveIndex(items.length - 1)
        break
    }
  }, [items.length, orientation])

  const getTabIndex = React.useCallback((index) => {
    return index === activeIndex ? 0 : -1
  }, [activeIndex])

  const setActiveItem = React.useCallback((index) => {
    if (index >= 0 && index < items.length) {
      setActiveIndex(index)
    }
  }, [items.length])

  return {
    activeIndex,
    handleKeyDown,
    getTabIndex,
    setActiveItem
  }
}

// Hook for managing skip links
export const useSkipLinks = (links = []) => {
  const [isVisible, setIsVisible] = React.useState(false)

  const SkipLinks = React.useCallback(() => (
    <div className={`fixed top-0 left-0 z-50 ${isVisible ? 'block' : 'sr-only'}`}>
      {links.map(({ href, label }) => (
        <a
          key={href}
          href={href}
          className="block bg-primary text-primary-foreground px-4 py-2 text-sm font-medium focus:not-sr-only focus:absolute focus:top-0 focus:left-0"
          onFocus={() => setIsVisible(true)}
          onBlur={() => setIsVisible(false)}
        >
          {label}
        </a>
      ))}
    </div>
  ), [links, isVisible])

  return { SkipLinks }
}

// Hook for managing live regions
export const useLiveRegion = () => {
  const [politeMessages, setPoliteMessages] = React.useState([])
  const [assertiveMessages, setAssertiveMessages] = React.useState([])

  const announcePolite = React.useCallback((message) => {
    const id = Date.now()
    setPoliteMessages(prev => [...prev, { id, message }])
    setTimeout(() => {
      setPoliteMessages(prev => prev.filter(m => m.id !== id))
    }, 1000)
  }, [])

  const announceAssertive = React.useCallback((message) => {
    const id = Date.now()
    setAssertiveMessages(prev => [...prev, { id, message }])
    setTimeout(() => {
      setAssertiveMessages(prev => prev.filter(m => m.id !== id))
    }, 1000)
  }, [])

  const LiveRegions = React.useCallback(() => (
    <>
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {politeMessages.map(({ id, message }) => (
          <div key={id}>{message}</div>
        ))}
      </div>
      <div aria-live="assertive" aria-atomic="true" className="sr-only">
        {assertiveMessages.map(({ id, message }) => (
          <div key={id}>{message}</div>
        ))}
      </div>
    </>
  ), [politeMessages, assertiveMessages])

  return {
    announcePolite,
    announceAssertive,
    LiveRegions
  }
}

