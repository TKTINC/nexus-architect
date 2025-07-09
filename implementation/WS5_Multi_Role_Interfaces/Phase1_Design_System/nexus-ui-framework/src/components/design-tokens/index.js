// Design Tokens - Index
// Nexus Architect Design System

import { colors } from './colors.js'
import { typography } from './typography.js'
import { spacing } from './spacing.js'

// Breakpoints for responsive design
export const breakpoints = {
  xs: '0px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px'
}

// Border radius values
export const borderRadius = {
  none: '0',
  sm: '0.125rem',     // 2px
  base: '0.25rem',    // 4px
  md: '0.375rem',     // 6px
  lg: '0.5rem',       // 8px
  xl: '0.75rem',      // 12px
  '2xl': '1rem',      // 16px
  '3xl': '1.5rem',    // 24px
  full: '9999px'
}

// Shadow values
export const shadows = {
  xs: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
  sm: '0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)',
  base: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
  md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
  lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
  xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
  '2xl': '0 25px 50px -12px rgb(0 0 0 / 0.25)',
  inner: 'inset 0 2px 4px 0 rgb(0 0 0 / 0.05)',
  none: '0 0 #0000'
}

// Z-index scale
export const zIndex = {
  auto: 'auto',
  0: '0',
  10: '10',
  20: '20',
  30: '30',
  40: '40',
  50: '50',
  dropdown: '1000',
  sticky: '1020',
  fixed: '1030',
  modal: '1040',
  popover: '1050',
  tooltip: '1060',
  toast: '1070',
  overlay: '1080'
}

// Animation durations
export const duration = {
  75: '75ms',
  100: '100ms',
  150: '150ms',
  200: '200ms',
  300: '300ms',
  500: '500ms',
  700: '700ms',
  1000: '1000ms'
}

// Animation timing functions
export const timingFunction = {
  linear: 'linear',
  in: 'cubic-bezier(0.4, 0, 1, 1)',
  out: 'cubic-bezier(0, 0, 0.2, 1)',
  'in-out': 'cubic-bezier(0.4, 0, 0.2, 1)'
}

// Component-specific design tokens
export const components = {
  button: {
    borderRadius: borderRadius.md,
    fontWeight: typography.fontWeight.medium,
    fontSize: typography.fontSize.sm,
    padding: {
      xs: { x: spacing.scale[2], y: spacing.scale[1] },
      sm: { x: spacing.scale[3], y: spacing.scale[1.5] },
      md: { x: spacing.scale[4], y: spacing.scale[2] },
      lg: { x: spacing.scale[6], y: spacing.scale[2.5] },
      xl: { x: spacing.scale[8], y: spacing.scale[3] }
    },
    transition: `all ${duration[150]} ${timingFunction['in-out']}`
  },

  input: {
    borderRadius: borderRadius.md,
    fontSize: typography.fontSize.sm,
    padding: {
      xs: { x: spacing.scale[2], y: spacing.scale[1] },
      sm: { x: spacing.scale[3], y: spacing.scale[1.5] },
      md: { x: spacing.scale[3], y: spacing.scale[2] },
      lg: { x: spacing.scale[4], y: spacing.scale[2.5] },
      xl: { x: spacing.scale[4], y: spacing.scale[3] }
    },
    transition: `all ${duration[150]} ${timingFunction['in-out']}`
  },

  card: {
    borderRadius: borderRadius.lg,
    shadow: shadows.sm,
    padding: {
      xs: spacing.scale[3],
      sm: spacing.scale[4],
      md: spacing.scale[6],
      lg: spacing.scale[8],
      xl: spacing.scale[10]
    }
  },

  modal: {
    borderRadius: borderRadius.xl,
    shadow: shadows['2xl'],
    backdrop: 'rgba(0, 0, 0, 0.5)',
    zIndex: zIndex.modal,
    transition: `all ${duration[200]} ${timingFunction['in-out']}`
  },

  tooltip: {
    borderRadius: borderRadius.md,
    fontSize: typography.fontSize.xs,
    padding: { x: spacing.scale[2], y: spacing.scale[1] },
    zIndex: zIndex.tooltip,
    transition: `all ${duration[150]} ${timingFunction['in-out']}`
  },

  dropdown: {
    borderRadius: borderRadius.md,
    shadow: shadows.lg,
    zIndex: zIndex.dropdown,
    transition: `all ${duration[150]} ${timingFunction['in-out']}`
  }
}

// Theme configuration
export const themes = {
  light: {
    colors: colors.light,
    shadows: shadows,
    name: 'light'
  },
  dark: {
    colors: colors.dark,
    shadows: {
      ...shadows,
      // Adjust shadows for dark theme
      sm: '0 1px 3px 0 rgb(0 0 0 / 0.3), 0 1px 2px -1px rgb(0 0 0 / 0.3)',
      base: '0 4px 6px -1px rgb(0 0 0 / 0.3), 0 2px 4px -2px rgb(0 0 0 / 0.3)',
      lg: '0 10px 15px -3px rgb(0 0 0 / 0.3), 0 4px 6px -4px rgb(0 0 0 / 0.3)',
      xl: '0 20px 25px -5px rgb(0 0 0 / 0.3), 0 8px 10px -6px rgb(0 0 0 / 0.3)'
    },
    name: 'dark'
  },
  highContrast: {
    colors: colors.highContrast,
    shadows: {
      ...shadows,
      // High contrast shadows
      sm: '0 1px 3px 0 rgb(255 255 255 / 0.2), 0 1px 2px -1px rgb(255 255 255 / 0.2)',
      base: '0 4px 6px -1px rgb(255 255 255 / 0.2), 0 2px 4px -2px rgb(255 255 255 / 0.2)',
      lg: '0 10px 15px -3px rgb(255 255 255 / 0.2), 0 4px 6px -4px rgb(255 255 255 / 0.2)'
    },
    name: 'high-contrast'
  }
}

// Utility functions
export const getToken = (path, theme = 'light') => {
  const pathArray = path.split('.')
  let value = { colors, typography, spacing, breakpoints, borderRadius, shadows, zIndex, duration, timingFunction, components, themes }
  
  for (const key of pathArray) {
    value = value[key]
    if (!value) return null
  }
  
  return value
}

export const createUtilityClass = (property, value, responsive = false) => {
  const propertyMap = {
    color: 'text',
    backgroundColor: 'bg',
    borderColor: 'border',
    fontSize: 'text',
    fontWeight: 'font',
    lineHeight: 'leading',
    letterSpacing: 'tracking',
    margin: 'm',
    padding: 'p',
    borderRadius: 'rounded',
    shadow: 'shadow',
    zIndex: 'z'
  }
  
  const prefix = propertyMap[property] || property
  const className = `${prefix}-${value}`
  
  if (responsive && typeof responsive === 'string') {
    return `${responsive}:${className}`
  }
  
  return className
}

// Export all design tokens
export {
  colors,
  typography,
  spacing
}

// Default export with all tokens
export default {
  colors,
  typography,
  spacing,
  breakpoints,
  borderRadius,
  shadows,
  zIndex,
  duration,
  timingFunction,
  components,
  themes,
  getToken,
  createUtilityClass
}

