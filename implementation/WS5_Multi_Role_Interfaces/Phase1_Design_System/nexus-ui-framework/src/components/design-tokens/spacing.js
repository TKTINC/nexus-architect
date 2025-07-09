// Design Tokens - Spacing
// Nexus Architect Design System

export const spacing = {
  // Base spacing scale (rem units)
  scale: {
    0: '0',
    px: '1px',
    0.5: '0.125rem',    // 2px
    1: '0.25rem',       // 4px
    1.5: '0.375rem',    // 6px
    2: '0.5rem',        // 8px
    2.5: '0.625rem',    // 10px
    3: '0.75rem',       // 12px
    3.5: '0.875rem',    // 14px
    4: '1rem',          // 16px
    5: '1.25rem',       // 20px
    6: '1.5rem',        // 24px
    7: '1.75rem',       // 28px
    8: '2rem',          // 32px
    9: '2.25rem',       // 36px
    10: '2.5rem',       // 40px
    11: '2.75rem',      // 44px
    12: '3rem',         // 48px
    14: '3.5rem',       // 56px
    16: '4rem',         // 64px
    20: '5rem',         // 80px
    24: '6rem',         // 96px
    28: '7rem',         // 112px
    32: '8rem',         // 128px
    36: '9rem',         // 144px
    40: '10rem',        // 160px
    44: '11rem',        // 176px
    48: '12rem',        // 192px
    52: '13rem',        // 208px
    56: '14rem',        // 224px
    60: '15rem',        // 240px
    64: '16rem',        // 256px
    72: '18rem',        // 288px
    80: '20rem',        // 320px
    96: '24rem'         // 384px
  },

  // Semantic spacing values
  semantic: {
    // Component spacing
    component: {
      xs: '0.25rem',      // 4px
      sm: '0.5rem',       // 8px
      md: '1rem',         // 16px
      lg: '1.5rem',       // 24px
      xl: '2rem',         // 32px
      '2xl': '3rem',      // 48px
      '3xl': '4rem'       // 64px
    },

    // Layout spacing
    layout: {
      xs: '1rem',         // 16px
      sm: '1.5rem',       // 24px
      md: '2rem',         // 32px
      lg: '3rem',         // 48px
      xl: '4rem',         // 64px
      '2xl': '6rem',      // 96px
      '3xl': '8rem'       // 128px
    },

    // Section spacing
    section: {
      xs: '2rem',         // 32px
      sm: '3rem',         // 48px
      md: '4rem',         // 64px
      lg: '6rem',         // 96px
      xl: '8rem',         // 128px
      '2xl': '12rem',     // 192px
      '3xl': '16rem'      // 256px
    },

    // Container spacing
    container: {
      xs: '1rem',         // 16px
      sm: '1.5rem',       // 24px
      md: '2rem',         // 32px
      lg: '2.5rem',       // 40px
      xl: '3rem',         // 48px
      '2xl': '4rem'       // 64px
    }
  },

  // Responsive spacing
  responsive: {
    // Mobile-first responsive spacing
    xs: {
      component: '0.5rem',
      layout: '1rem',
      section: '2rem',
      container: '1rem'
    },
    sm: {
      component: '0.75rem',
      layout: '1.5rem',
      section: '3rem',
      container: '1.5rem'
    },
    md: {
      component: '1rem',
      layout: '2rem',
      section: '4rem',
      container: '2rem'
    },
    lg: {
      component: '1.25rem',
      layout: '2.5rem',
      section: '5rem',
      container: '2.5rem'
    },
    xl: {
      component: '1.5rem',
      layout: '3rem',
      section: '6rem',
      container: '3rem'
    },
    '2xl': {
      component: '2rem',
      layout: '4rem',
      section: '8rem',
      container: '4rem'
    }
  },

  // Grid spacing
  grid: {
    gap: {
      xs: '0.5rem',       // 8px
      sm: '1rem',         // 16px
      md: '1.5rem',       // 24px
      lg: '2rem',         // 32px
      xl: '3rem',         // 48px
      '2xl': '4rem'       // 64px
    },
    gutter: {
      xs: '1rem',         // 16px
      sm: '1.5rem',       // 24px
      md: '2rem',         // 32px
      lg: '2.5rem',       // 40px
      xl: '3rem',         // 48px
      '2xl': '4rem'       // 64px
    }
  },

  // Form spacing
  form: {
    field: {
      xs: '0.5rem',       // 8px
      sm: '0.75rem',      // 12px
      md: '1rem',         // 16px
      lg: '1.25rem',      // 20px
      xl: '1.5rem'        // 24px
    },
    group: {
      xs: '1rem',         // 16px
      sm: '1.5rem',       // 24px
      md: '2rem',         // 32px
      lg: '2.5rem',       // 40px
      xl: '3rem'          // 48px
    }
  },

  // Card spacing
  card: {
    padding: {
      xs: '0.75rem',      // 12px
      sm: '1rem',         // 16px
      md: '1.5rem',       // 24px
      lg: '2rem',         // 32px
      xl: '2.5rem',       // 40px
      '2xl': '3rem'       // 48px
    },
    gap: {
      xs: '0.5rem',       // 8px
      sm: '0.75rem',      // 12px
      md: '1rem',         // 16px
      lg: '1.5rem',       // 24px
      xl: '2rem'          // 32px
    }
  }
}

// Spacing utility functions
export const getSpacing = (size) => {
  return spacing.scale[size] || spacing.semantic.component[size] || null
}

export const getSemanticSpacing = (category, size) => {
  return spacing.semantic[category]?.[size] || null
}

export const getResponsiveSpacing = (breakpoint, category) => {
  return spacing.responsive[breakpoint]?.[category] || null
}

export const createSpacingClass = (property, size, responsive = false) => {
  const prefixes = {
    margin: 'm',
    marginTop: 'mt',
    marginRight: 'mr',
    marginBottom: 'mb',
    marginLeft: 'ml',
    marginX: 'mx',
    marginY: 'my',
    padding: 'p',
    paddingTop: 'pt',
    paddingRight: 'pr',
    paddingBottom: 'pb',
    paddingLeft: 'pl',
    paddingX: 'px',
    paddingY: 'py',
    gap: 'gap',
    space: 'space'
  }
  
  const prefix = prefixes[property]
  if (!prefix) return ''
  
  if (responsive && typeof responsive === 'string') {
    return `${responsive}:${prefix}-${size}`
  }
  
  return `${prefix}-${size}`
}

// Spacing presets for common use cases
export const spacingPresets = {
  // Button spacing
  button: {
    xs: { paddingX: '2', paddingY: '1' },
    sm: { paddingX: '3', paddingY: '1.5' },
    md: { paddingX: '4', paddingY: '2' },
    lg: { paddingX: '6', paddingY: '2.5' },
    xl: { paddingX: '8', paddingY: '3' }
  },

  // Input spacing
  input: {
    xs: { paddingX: '2', paddingY: '1' },
    sm: { paddingX: '3', paddingY: '1.5' },
    md: { paddingX: '3', paddingY: '2' },
    lg: { paddingX: '4', paddingY: '2.5' },
    xl: { paddingX: '4', paddingY: '3' }
  },

  // Card spacing
  cardContent: {
    xs: { padding: '3' },
    sm: { padding: '4' },
    md: { padding: '6' },
    lg: { padding: '8' },
    xl: { padding: '10' }
  },

  // Modal spacing
  modal: {
    xs: { padding: '4', gap: '4' },
    sm: { padding: '6', gap: '4' },
    md: { padding: '6', gap: '6' },
    lg: { padding: '8', gap: '6' },
    xl: { padding: '10', gap: '8' }
  },

  // Navigation spacing
  nav: {
    item: { paddingX: '3', paddingY: '2' },
    section: { paddingY: '4' },
    container: { padding: '4' }
  },

  // List spacing
  list: {
    item: { paddingY: '2' },
    section: { marginY: '4' },
    container: { gap: '2' }
  }
}

export default spacing

