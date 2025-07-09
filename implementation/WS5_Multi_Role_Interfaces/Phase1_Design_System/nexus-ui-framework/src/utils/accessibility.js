// Accessibility Utilities
// WCAG 2.1 AA Compliance Helpers

// Color contrast utilities
export const getContrastRatio = (color1, color2) => {
  const getLuminance = (color) => {
    // Convert hex to RGB
    const hex = color.replace('#', '')
    const r = parseInt(hex.substr(0, 2), 16) / 255
    const g = parseInt(hex.substr(2, 2), 16) / 255
    const b = parseInt(hex.substr(4, 2), 16) / 255

    // Calculate relative luminance
    const sRGB = [r, g, b].map(c => {
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
    })

    return 0.2126 * sRGB[0] + 0.7152 * sRGB[1] + 0.0722 * sRGB[2]
  }

  const lum1 = getLuminance(color1)
  const lum2 = getLuminance(color2)
  const brightest = Math.max(lum1, lum2)
  const darkest = Math.min(lum1, lum2)

  return (brightest + 0.05) / (darkest + 0.05)
}

export const meetsContrastRequirement = (foreground, background, level = 'AA', size = 'normal') => {
  const ratio = getContrastRatio(foreground, background)
  
  if (level === 'AAA') {
    return size === 'large' ? ratio >= 4.5 : ratio >= 7
  }
  
  // AA level (default)
  return size === 'large' ? ratio >= 3 : ratio >= 4.5
}

// Focus management utilities
export const getFocusableElements = (container) => {
  if (!container) return []
  
  const focusableSelectors = [
    'button:not([disabled])',
    '[href]',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
    '[contenteditable="true"]'
  ].join(', ')

  return Array.from(container.querySelectorAll(focusableSelectors))
}

export const trapFocus = (container) => {
  const focusableElements = getFocusableElements(container)
  const firstElement = focusableElements[0]
  const lastElement = focusableElements[focusableElements.length - 1]

  const handleTabKey = (e) => {
    if (e.key !== 'Tab') return

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault()
        lastElement?.focus()
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault()
        firstElement?.focus()
      }
    }
  }

  container.addEventListener('keydown', handleTabKey)
  
  // Focus first element
  firstElement?.focus()

  return () => {
    container.removeEventListener('keydown', handleTabKey)
  }
}

// ARIA utilities
export const generateId = (prefix = 'nexus') => {
  return `${prefix}-${Math.random().toString(36).substr(2, 9)}`
}

export const createAriaLabel = (text, context) => {
  if (!text) return undefined
  return context ? `${text}, ${context}` : text
}

export const createAriaDescription = (description, additionalInfo) => {
  const parts = [description, additionalInfo].filter(Boolean)
  return parts.length > 0 ? parts.join('. ') : undefined
}

// Keyboard navigation utilities
export const KEYBOARD_KEYS = {
  ENTER: 'Enter',
  SPACE: ' ',
  ESCAPE: 'Escape',
  TAB: 'Tab',
  ARROW_UP: 'ArrowUp',
  ARROW_DOWN: 'ArrowDown',
  ARROW_LEFT: 'ArrowLeft',
  ARROW_RIGHT: 'ArrowRight',
  HOME: 'Home',
  END: 'End',
  PAGE_UP: 'PageUp',
  PAGE_DOWN: 'PageDown'
}

export const isNavigationKey = (key) => {
  return Object.values(KEYBOARD_KEYS).includes(key)
}

export const handleRovingTabIndex = (items, currentIndex, key, orientation = 'horizontal') => {
  const isHorizontal = orientation === 'horizontal'
  const isVertical = orientation === 'vertical'
  const isBoth = orientation === 'both'

  let newIndex = currentIndex

  switch (key) {
    case KEYBOARD_KEYS.ARROW_RIGHT:
      if (isHorizontal || isBoth) {
        newIndex = (currentIndex + 1) % items.length
      }
      break
    case KEYBOARD_KEYS.ARROW_LEFT:
      if (isHorizontal || isBoth) {
        newIndex = (currentIndex - 1 + items.length) % items.length
      }
      break
    case KEYBOARD_KEYS.ARROW_DOWN:
      if (isVertical || isBoth) {
        newIndex = (currentIndex + 1) % items.length
      }
      break
    case KEYBOARD_KEYS.ARROW_UP:
      if (isVertical || isBoth) {
        newIndex = (currentIndex - 1 + items.length) % items.length
      }
      break
    case KEYBOARD_KEYS.HOME:
      newIndex = 0
      break
    case KEYBOARD_KEYS.END:
      newIndex = items.length - 1
      break
  }

  return newIndex
}

// Screen reader utilities
export const announceToScreenReader = (message, priority = 'polite') => {
  const announcement = document.createElement('div')
  announcement.setAttribute('aria-live', priority)
  announcement.setAttribute('aria-atomic', 'true')
  announcement.className = 'sr-only'
  announcement.textContent = message

  document.body.appendChild(announcement)

  setTimeout(() => {
    document.body.removeChild(announcement)
  }, 1000)
}

// Form accessibility utilities
export const getFormFieldIds = (name) => {
  const baseId = generateId(name)
  return {
    field: baseId,
    label: `${baseId}-label`,
    description: `${baseId}-description`,
    error: `${baseId}-error`
  }
}

export const createFieldAriaAttributes = (fieldIds, hasError = false, hasDescription = false) => {
  const attributes = {
    id: fieldIds.field,
    'aria-labelledby': fieldIds.label
  }

  const describedBy = []
  if (hasDescription) describedBy.push(fieldIds.description)
  if (hasError) describedBy.push(fieldIds.error)

  if (describedBy.length > 0) {
    attributes['aria-describedby'] = describedBy.join(' ')
  }

  if (hasError) {
    attributes['aria-invalid'] = 'true'
  }

  return attributes
}

// Media query utilities for accessibility preferences
export const getAccessibilityPreferences = () => {
  return {
    prefersReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    prefersHighContrast: window.matchMedia('(prefers-contrast: high)').matches,
    prefersColorScheme: window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light',
    prefersReducedTransparency: window.matchMedia('(prefers-reduced-transparency: reduce)').matches
  }
}

// Text utilities for accessibility
export const truncateText = (text, maxLength, suffix = '...') => {
  if (!text || text.length <= maxLength) return text
  return text.substring(0, maxLength - suffix.length) + suffix
}

export const createAccessibleText = (text, maxLength) => {
  if (!maxLength || text.length <= maxLength) {
    return { displayText: text, fullText: text }
  }

  const displayText = truncateText(text, maxLength)
  return { displayText, fullText: text }
}

// Skip link utilities
export const createSkipLink = (targetId, label) => {
  return {
    href: `#${targetId}`,
    label,
    onClick: (e) => {
      e.preventDefault()
      const target = document.getElementById(targetId)
      if (target) {
        target.focus()
        target.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }
  }
}

// Landmark utilities
export const LANDMARK_ROLES = {
  BANNER: 'banner',
  NAVIGATION: 'navigation',
  MAIN: 'main',
  COMPLEMENTARY: 'complementary',
  CONTENTINFO: 'contentinfo',
  SEARCH: 'search',
  FORM: 'form',
  REGION: 'region'
}

export const createLandmarkProps = (role, label) => {
  const props = { role }
  if (label) {
    props['aria-label'] = label
  }
  return props
}

// Table accessibility utilities
export const createTableHeaders = (headers) => {
  return headers.map((header, index) => ({
    id: generateId(`header-${index}`),
    text: header
  }))
}

export const createTableCellProps = (headerIds, rowIndex, colIndex) => {
  return {
    headers: Array.isArray(headerIds) ? headerIds[colIndex] : headerIds,
    role: 'gridcell',
    'aria-rowindex': rowIndex + 1,
    'aria-colindex': colIndex + 1
  }
}

// Live region utilities
export const LIVE_REGION_TYPES = {
  POLITE: 'polite',
  ASSERTIVE: 'assertive',
  OFF: 'off'
}

export const createLiveRegion = (type = LIVE_REGION_TYPES.POLITE) => {
  const region = document.createElement('div')
  region.setAttribute('aria-live', type)
  region.setAttribute('aria-atomic', 'true')
  region.className = 'sr-only'
  document.body.appendChild(region)

  return {
    announce: (message) => {
      region.textContent = message
      setTimeout(() => {
        region.textContent = ''
      }, 1000)
    },
    destroy: () => {
      if (region.parentNode) {
        region.parentNode.removeChild(region)
      }
    }
  }
}

// Validation utilities
export const validateAccessibility = {
  hasAriaLabel: (element) => {
    return element.hasAttribute('aria-label') || element.hasAttribute('aria-labelledby')
  },
  
  hasKeyboardSupport: (element) => {
    const tabIndex = element.getAttribute('tabindex')
    return element.tagName.toLowerCase() === 'button' || 
           element.tagName.toLowerCase() === 'a' ||
           element.tagName.toLowerCase() === 'input' ||
           (tabIndex !== null && tabIndex !== '-1')
  },
  
  hasProperContrast: (element, background) => {
    const style = window.getComputedStyle(element)
    const color = style.color
    const bgColor = background || style.backgroundColor
    
    // This is a simplified check - in production, you'd use a proper color parsing library
    return true // Placeholder
  },
  
  hasProperHeadingStructure: (container) => {
    const headings = container.querySelectorAll('h1, h2, h3, h4, h5, h6')
    let previousLevel = 0
    
    for (const heading of headings) {
      const level = parseInt(heading.tagName.charAt(1))
      if (level > previousLevel + 1) {
        return false // Skipped heading level
      }
      previousLevel = level
    }
    
    return true
  }
}

// Export all utilities
export default {
  getContrastRatio,
  meetsContrastRequirement,
  getFocusableElements,
  trapFocus,
  generateId,
  createAriaLabel,
  createAriaDescription,
  KEYBOARD_KEYS,
  isNavigationKey,
  handleRovingTabIndex,
  announceToScreenReader,
  getFormFieldIds,
  createFieldAriaAttributes,
  getAccessibilityPreferences,
  truncateText,
  createAccessibleText,
  createSkipLink,
  LANDMARK_ROLES,
  createLandmarkProps,
  createTableHeaders,
  createTableCellProps,
  LIVE_REGION_TYPES,
  createLiveRegion,
  validateAccessibility
}

