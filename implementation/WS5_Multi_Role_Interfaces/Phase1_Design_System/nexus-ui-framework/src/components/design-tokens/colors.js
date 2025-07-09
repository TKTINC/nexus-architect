// Design Tokens - Colors
// Nexus Architect Design System

export const colors = {
  // Primary Brand Colors
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9',
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
    950: '#082f49'
  },

  // Secondary Colors
  secondary: {
    50: '#f8fafc',
    100: '#f1f5f9',
    200: '#e2e8f0',
    300: '#cbd5e1',
    400: '#94a3b8',
    500: '#64748b',
    600: '#475569',
    700: '#334155',
    800: '#1e293b',
    900: '#0f172a',
    950: '#020617'
  },

  // Accent Colors
  accent: {
    50: '#fdf4ff',
    100: '#fae8ff',
    200: '#f5d0fe',
    300: '#f0abfc',
    400: '#e879f9',
    500: '#d946ef',
    600: '#c026d3',
    700: '#a21caf',
    800: '#86198f',
    900: '#701a75',
    950: '#4a044e'
  },

  // Success Colors
  success: {
    50: '#f0fdf4',
    100: '#dcfce7',
    200: '#bbf7d0',
    300: '#86efac',
    400: '#4ade80',
    500: '#22c55e',
    600: '#16a34a',
    700: '#15803d',
    800: '#166534',
    900: '#14532d',
    950: '#052e16'
  },

  // Warning Colors
  warning: {
    50: '#fffbeb',
    100: '#fef3c7',
    200: '#fde68a',
    300: '#fcd34d',
    400: '#fbbf24',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
    800: '#92400e',
    900: '#78350f',
    950: '#451a03'
  },

  // Error Colors
  error: {
    50: '#fef2f2',
    100: '#fee2e2',
    200: '#fecaca',
    300: '#fca5a5',
    400: '#f87171',
    500: '#ef4444',
    600: '#dc2626',
    700: '#b91c1c',
    800: '#991b1b',
    900: '#7f1d1d',
    950: '#450a0a'
  },

  // Neutral Colors
  neutral: {
    50: '#fafafa',
    100: '#f5f5f5',
    200: '#e5e5e5',
    300: '#d4d4d4',
    400: '#a3a3a3',
    500: '#737373',
    600: '#525252',
    700: '#404040',
    800: '#262626',
    900: '#171717',
    950: '#0a0a0a'
  },

  // Semantic Colors
  semantic: {
    info: '#0ea5e9',
    success: '#22c55e',
    warning: '#f59e0b',
    error: '#ef4444',
    critical: '#dc2626'
  },

  // Theme-specific Colors
  light: {
    background: '#ffffff',
    foreground: '#0f172a',
    card: '#ffffff',
    cardForeground: '#0f172a',
    popover: '#ffffff',
    popoverForeground: '#0f172a',
    muted: '#f1f5f9',
    mutedForeground: '#64748b',
    border: '#e2e8f0',
    input: '#e2e8f0',
    ring: '#94a3b8'
  },

  dark: {
    background: '#0f172a',
    foreground: '#f8fafc',
    card: '#1e293b',
    cardForeground: '#f8fafc',
    popover: '#1e293b',
    popoverForeground: '#f8fafc',
    muted: '#334155',
    mutedForeground: '#94a3b8',
    border: '#334155',
    input: '#475569',
    ring: '#64748b'
  },

  // High Contrast Theme
  highContrast: {
    background: '#000000',
    foreground: '#ffffff',
    card: '#1a1a1a',
    cardForeground: '#ffffff',
    popover: '#1a1a1a',
    popoverForeground: '#ffffff',
    muted: '#333333',
    mutedForeground: '#cccccc',
    border: '#666666',
    input: '#333333',
    ring: '#ffffff',
    primary: '#ffffff',
    primaryForeground: '#000000',
    secondary: '#333333',
    secondaryForeground: '#ffffff'
  }
}

// Color utility functions
export const getColorValue = (colorPath, theme = 'light') => {
  const pathArray = colorPath.split('.')
  let value = colors
  
  for (const key of pathArray) {
    value = value[key]
    if (!value) return null
  }
  
  return value
}

export const generateColorScale = (baseColor, steps = 11) => {
  // Generate a color scale from a base color
  // This is a simplified implementation - in production, you'd use a proper color library
  const scale = {}
  const stepSize = 100 / (steps - 1)
  
  for (let i = 0; i < steps; i++) {
    const step = i * stepSize
    scale[step === 0 ? 50 : step === 100 ? 950 : Math.round(step * 9 + 50)] = baseColor
  }
  
  return scale
}

export default colors

