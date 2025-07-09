import React, { createContext, useContext, useEffect, useState } from 'react';

const ThemeProviderContext = createContext({
  theme: 'system',
  setTheme: () => null,
  accessibility: {
    highContrast: false,
    reducedMotion: false,
    fontSize: 'medium'
  },
  setAccessibility: () => null,
  personalizedTheme: null,
  setPersonalizedTheme: () => null
});

export function ThemeProvider({
  children,
  defaultTheme = 'system',
  storageKey = 'nexus-ui-theme',
  ...props
}) {
  const [theme, setTheme] = useState(
    () => (typeof window !== 'undefined' && localStorage.getItem(storageKey)) || defaultTheme
  );

  const [accessibility, setAccessibility] = useState({
    highContrast: false,
    reducedMotion: false,
    fontSize: 'medium'
  });

  const [personalizedTheme, setPersonalizedTheme] = useState(null);

  useEffect(() => {
    // Load accessibility preferences
    const savedAccessibility = localStorage.getItem('nexus-accessibility');
    if (savedAccessibility) {
      setAccessibility(JSON.parse(savedAccessibility));
    } else {
      // Detect system preferences
      const systemPreferences = {
        highContrast: window.matchMedia('(prefers-contrast: high)').matches,
        reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
        fontSize: 'medium'
      };
      setAccessibility(systemPreferences);
    }

    // Load personalized theme
    const savedPersonalizedTheme = localStorage.getItem('nexus-personalized-theme');
    if (savedPersonalizedTheme) {
      setPersonalizedTheme(JSON.parse(savedPersonalizedTheme));
    }
  }, []);

  useEffect(() => {
    const root = window.document.documentElement;

    root.classList.remove('light', 'dark', 'system');

    if (theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';

      root.classList.add(systemTheme);
    } else {
      root.classList.add(theme);
    }

    // Apply accessibility preferences
    if (accessibility.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    if (accessibility.reducedMotion) {
      root.style.setProperty('--animation-duration', '0.01ms');
    } else {
      root.style.removeProperty('--animation-duration');
    }

    // Apply font size preference
    const fontSizeMap = {
      small: '14px',
      medium: '16px',
      large: '18px',
      xlarge: '20px'
    };
    root.style.setProperty('--base-font-size', fontSizeMap[accessibility.fontSize]);

    // Apply personalized theme if available
    if (personalizedTheme) {
      Object.entries(personalizedTheme.colors || {}).forEach(([key, value]) => {
        root.style.setProperty(`--${key}`, value);
      });
    }
  }, [theme, accessibility, personalizedTheme]);

  const updateTheme = (newTheme) => {
    localStorage.setItem(storageKey, newTheme);
    setTheme(newTheme);
  };

  const updateAccessibility = (newAccessibility) => {
    const updatedAccessibility = { ...accessibility, ...newAccessibility };
    localStorage.setItem('nexus-accessibility', JSON.stringify(updatedAccessibility));
    setAccessibility(updatedAccessibility);
  };

  const updatePersonalizedTheme = (newPersonalizedTheme) => {
    localStorage.setItem('nexus-personalized-theme', JSON.stringify(newPersonalizedTheme));
    setPersonalizedTheme(newPersonalizedTheme);
  };

  // AI-powered theme generation based on user behavior
  const generatePersonalizedTheme = async (userBehaviorData) => {
    try {
      // This would use TensorFlow.js or similar to analyze user preferences
      // For now, we'll simulate with a simple algorithm
      const timeOfDay = new Date().getHours();
      const isNightTime = timeOfDay < 6 || timeOfDay > 18;
      
      const baseTheme = isNightTime ? 'dark' : 'light';
      
      // Analyze user's color preferences from behavior data
      const preferredColors = analyzeColorPreferences(userBehaviorData);
      
      const personalizedColors = {
        primary: preferredColors.primary || (baseTheme === 'dark' ? '217.2 91.2% 59.8%' : '221.2 83.2% 53.3%'),
        accent: preferredColors.accent || (baseTheme === 'dark' ? '217.2 32.6% 17.5%' : '210 40% 96%'),
        background: baseTheme === 'dark' ? '222.2 84% 4.9%' : '0 0% 100%',
        foreground: baseTheme === 'dark' ? '210 40% 98%' : '222.2 84% 4.9%'
      };

      const generatedTheme = {
        name: 'personalized',
        baseTheme,
        colors: personalizedColors,
        generatedAt: new Date().toISOString(),
        confidence: 0.85
      };

      updatePersonalizedTheme(generatedTheme);
      return generatedTheme;
    } catch (error) {
      console.error('Failed to generate personalized theme:', error);
      return null;
    }
  };

  const analyzeColorPreferences = (behaviorData) => {
    // Simulate color preference analysis
    // In production, this would use ML models to analyze user interactions
    const defaultColors = {
      primary: null,
      accent: null
    };

    if (!behaviorData || !behaviorData.interactions) {
      return defaultColors;
    }

    // Analyze most interacted elements and their colors
    const colorInteractions = behaviorData.interactions
      .filter(interaction => interaction.elementColor)
      .reduce((acc, interaction) => {
        acc[interaction.elementColor] = (acc[interaction.elementColor] || 0) + 1;
        return acc;
      }, {});

    const mostUsedColor = Object.keys(colorInteractions).reduce((a, b) => 
      colorInteractions[a] > colorInteractions[b] ? a : b
    );

    return {
      primary: mostUsedColor || defaultColors.primary,
      accent: defaultColors.accent
    };
  };

  const value = {
    theme,
    setTheme: updateTheme,
    accessibility,
    setAccessibility: updateAccessibility,
    personalizedTheme,
    setPersonalizedTheme: updatePersonalizedTheme,
    generatePersonalizedTheme
  };

  return (
    <ThemeProviderContext.Provider {...props} value={value}>
      {children}
    </ThemeProviderContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeProviderContext);

  if (context === undefined)
    throw new Error('useTheme must be used within a ThemeProvider');

  return context;
};

