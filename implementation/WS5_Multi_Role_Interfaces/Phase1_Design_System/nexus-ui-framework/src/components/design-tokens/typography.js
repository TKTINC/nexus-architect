// Design Tokens - Typography
// Nexus Architect Design System

export const typography = {
  // Font Families
  fontFamily: {
    sans: [
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      'Segoe UI',
      'Roboto',
      'Oxygen',
      'Ubuntu',
      'Cantarell',
      'Fira Sans',
      'Droid Sans',
      'Helvetica Neue',
      'sans-serif'
    ],
    mono: [
      'JetBrains Mono',
      'Fira Code',
      'Monaco',
      'Consolas',
      'Liberation Mono',
      'Courier New',
      'monospace'
    ],
    display: [
      'Cal Sans',
      'Inter',
      '-apple-system',
      'BlinkMacSystemFont',
      'sans-serif'
    ]
  },

  // Font Sizes
  fontSize: {
    xs: ['0.75rem', { lineHeight: '1rem' }],      // 12px
    sm: ['0.875rem', { lineHeight: '1.25rem' }],  // 14px
    base: ['1rem', { lineHeight: '1.5rem' }],     // 16px
    lg: ['1.125rem', { lineHeight: '1.75rem' }],  // 18px
    xl: ['1.25rem', { lineHeight: '1.75rem' }],   // 20px
    '2xl': ['1.5rem', { lineHeight: '2rem' }],    // 24px
    '3xl': ['1.875rem', { lineHeight: '2.25rem' }], // 30px
    '4xl': ['2.25rem', { lineHeight: '2.5rem' }], // 36px
    '5xl': ['3rem', { lineHeight: '1' }],         // 48px
    '6xl': ['3.75rem', { lineHeight: '1' }],      // 60px
    '7xl': ['4.5rem', { lineHeight: '1' }],       // 72px
    '8xl': ['6rem', { lineHeight: '1' }],         // 96px
    '9xl': ['8rem', { lineHeight: '1' }]          // 128px
  },

  // Font Weights
  fontWeight: {
    thin: '100',
    extralight: '200',
    light: '300',
    normal: '400',
    medium: '500',
    semibold: '600',
    bold: '700',
    extrabold: '800',
    black: '900'
  },

  // Line Heights
  lineHeight: {
    none: '1',
    tight: '1.25',
    snug: '1.375',
    normal: '1.5',
    relaxed: '1.625',
    loose: '2'
  },

  // Letter Spacing
  letterSpacing: {
    tighter: '-0.05em',
    tight: '-0.025em',
    normal: '0em',
    wide: '0.025em',
    wider: '0.05em',
    widest: '0.1em'
  },

  // Text Styles - Semantic Typography Scale
  textStyles: {
    // Display Styles
    'display-2xl': {
      fontSize: '4.5rem',
      lineHeight: '1.1',
      fontWeight: '700',
      letterSpacing: '-0.02em',
      fontFamily: 'display'
    },
    'display-xl': {
      fontSize: '3.75rem',
      lineHeight: '1.1',
      fontWeight: '700',
      letterSpacing: '-0.02em',
      fontFamily: 'display'
    },
    'display-lg': {
      fontSize: '3rem',
      lineHeight: '1.2',
      fontWeight: '600',
      letterSpacing: '-0.02em',
      fontFamily: 'display'
    },
    'display-md': {
      fontSize: '2.25rem',
      lineHeight: '1.2',
      fontWeight: '600',
      letterSpacing: '-0.02em',
      fontFamily: 'display'
    },
    'display-sm': {
      fontSize: '1.875rem',
      lineHeight: '1.3',
      fontWeight: '600',
      letterSpacing: '-0.02em',
      fontFamily: 'display'
    },

    // Heading Styles
    'heading-xl': {
      fontSize: '1.5rem',
      lineHeight: '1.3',
      fontWeight: '600',
      letterSpacing: '-0.01em'
    },
    'heading-lg': {
      fontSize: '1.25rem',
      lineHeight: '1.4',
      fontWeight: '600',
      letterSpacing: '-0.01em'
    },
    'heading-md': {
      fontSize: '1.125rem',
      lineHeight: '1.4',
      fontWeight: '600',
      letterSpacing: '-0.01em'
    },
    'heading-sm': {
      fontSize: '1rem',
      lineHeight: '1.5',
      fontWeight: '600'
    },
    'heading-xs': {
      fontSize: '0.875rem',
      lineHeight: '1.5',
      fontWeight: '600'
    },

    // Body Text Styles
    'body-xl': {
      fontSize: '1.25rem',
      lineHeight: '1.6',
      fontWeight: '400'
    },
    'body-lg': {
      fontSize: '1.125rem',
      lineHeight: '1.6',
      fontWeight: '400'
    },
    'body-md': {
      fontSize: '1rem',
      lineHeight: '1.6',
      fontWeight: '400'
    },
    'body-sm': {
      fontSize: '0.875rem',
      lineHeight: '1.5',
      fontWeight: '400'
    },
    'body-xs': {
      fontSize: '0.75rem',
      lineHeight: '1.5',
      fontWeight: '400'
    },

    // Caption and Label Styles
    'caption-lg': {
      fontSize: '0.875rem',
      lineHeight: '1.4',
      fontWeight: '500'
    },
    'caption-md': {
      fontSize: '0.75rem',
      lineHeight: '1.4',
      fontWeight: '500'
    },
    'caption-sm': {
      fontSize: '0.6875rem',
      lineHeight: '1.4',
      fontWeight: '500'
    },

    // Code Styles
    'code-lg': {
      fontSize: '0.875rem',
      lineHeight: '1.5',
      fontWeight: '400',
      fontFamily: 'mono'
    },
    'code-md': {
      fontSize: '0.8125rem',
      lineHeight: '1.5',
      fontWeight: '400',
      fontFamily: 'mono'
    },
    'code-sm': {
      fontSize: '0.75rem',
      lineHeight: '1.5',
      fontWeight: '400',
      fontFamily: 'mono'
    }
  },

  // Responsive Typography
  responsive: {
    'display-2xl': {
      base: 'display-lg',
      md: 'display-xl',
      lg: 'display-2xl'
    },
    'display-xl': {
      base: 'display-md',
      md: 'display-lg',
      lg: 'display-xl'
    },
    'display-lg': {
      base: 'display-sm',
      md: 'display-md',
      lg: 'display-lg'
    }
  }
}

// Typography utility functions
export const getTextStyle = (styleName) => {
  return typography.textStyles[styleName] || null
}

export const getFontSize = (size) => {
  return typography.fontSize[size] || null
}

export const createTextClass = (style) => {
  const textStyle = getTextStyle(style)
  if (!textStyle) return ''
  
  let classes = []
  
  if (textStyle.fontSize) {
    const sizeKey = Object.keys(typography.fontSize).find(
      key => typography.fontSize[key][0] === textStyle.fontSize
    )
    if (sizeKey) classes.push(`text-${sizeKey}`)
  }
  
  if (textStyle.fontWeight) {
    const weightKey = Object.keys(typography.fontWeight).find(
      key => typography.fontWeight[key] === textStyle.fontWeight
    )
    if (weightKey) classes.push(`font-${weightKey}`)
  }
  
  if (textStyle.lineHeight) {
    const lineHeightKey = Object.keys(typography.lineHeight).find(
      key => typography.lineHeight[key] === textStyle.lineHeight
    )
    if (lineHeightKey) classes.push(`leading-${lineHeightKey}`)
  }
  
  if (textStyle.letterSpacing) {
    const spacingKey = Object.keys(typography.letterSpacing).find(
      key => typography.letterSpacing[key] === textStyle.letterSpacing
    )
    if (spacingKey) classes.push(`tracking-${spacingKey}`)
  }
  
  if (textStyle.fontFamily) {
    classes.push(`font-${textStyle.fontFamily}`)
  }
  
  return classes.join(' ')
}

export default typography

