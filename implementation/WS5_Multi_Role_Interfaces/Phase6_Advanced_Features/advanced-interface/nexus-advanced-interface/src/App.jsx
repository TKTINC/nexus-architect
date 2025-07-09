import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from './components/theme-provider';
import { AdvancedLayout } from './components/layout/AdvancedLayout';
import { PersonalizationDashboard } from './components/personalization/PersonalizationDashboard';
import './App.css';

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [userPreferences, setUserPreferences] = useState({
    theme: 'system',
    language: 'en',
    accessibility: {
      highContrast: false,
      reducedMotion: false,
      fontSize: 'medium'
    },
    personalization: {
      aiAssistance: true,
      voiceEnabled: true,
      adaptiveUI: true
    }
  });

  useEffect(() => {
    // Initialize advanced features
    const initializeApp = async () => {
      try {
        // Load user preferences from localStorage
        const savedPreferences = localStorage.getItem('nexus-user-preferences');
        if (savedPreferences) {
          setUserPreferences(JSON.parse(savedPreferences));
        }

        // Initialize AI models for personalization
        if (userPreferences.personalization.aiAssistance) {
          await initializeAIModels();
        }

        // Initialize voice recognition if enabled
        if (userPreferences.personalization.voiceEnabled && 'webkitSpeechRecognition' in window) {
          await initializeVoiceRecognition();
        }

        // Initialize performance monitoring
        await initializePerformanceMonitoring();

        setIsLoading(false);
      } catch (error) {
        console.error('Failed to initialize advanced features:', error);
        setIsLoading(false);
      }
    };

    initializeApp();
  }, []);

  const initializeAIModels = async () => {
    // Initialize TensorFlow.js models for personalization
    try {
      console.log('Initializing AI models for personalization...');
      // This would load actual TensorFlow.js models in production
      return Promise.resolve();
    } catch (error) {
      console.error('Failed to initialize AI models:', error);
    }
  };

  const initializeVoiceRecognition = async () => {
    // Initialize Web Speech API
    try {
      if ('webkitSpeechRecognition' in window) {
        console.log('Voice recognition available and enabled');
        return Promise.resolve();
      }
    } catch (error) {
      console.error('Failed to initialize voice recognition:', error);
    }
  };

  const initializePerformanceMonitoring = async () => {
    // Initialize performance monitoring
    try {
      // Monitor Core Web Vitals
      if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            console.log(`Performance metric: ${entry.name} = ${entry.value}`);
          }
        });
        observer.observe({ entryTypes: ['measure', 'navigation', 'paint'] });
      }
      return Promise.resolve();
    } catch (error) {
      console.error('Failed to initialize performance monitoring:', error);
    }
  };

  const updateUserPreferences = (newPreferences) => {
    const updatedPreferences = { ...userPreferences, ...newPreferences };
    setUserPreferences(updatedPreferences);
    localStorage.setItem('nexus-user-preferences', JSON.stringify(updatedPreferences));
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Initializing Advanced Features...</p>
        </div>
      </div>
    );
  }

  return (
    <ThemeProvider defaultTheme={userPreferences.theme} storageKey="nexus-ui-theme">
      <Router>
        <div className="min-h-screen bg-background font-sans antialiased">
          <AdvancedLayout userPreferences={userPreferences} onPreferencesChange={updateUserPreferences}>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route 
                path="/dashboard" 
                element={
                  <PersonalizationDashboard 
                    userPreferences={userPreferences}
                    onPreferencesChange={updateUserPreferences}
                  />
                } 
              />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </AdvancedLayout>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;

