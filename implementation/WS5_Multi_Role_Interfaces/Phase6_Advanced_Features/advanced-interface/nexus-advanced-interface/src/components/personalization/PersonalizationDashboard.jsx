import React, { useState, useEffect } from 'react';

export function PersonalizationDashboard({ userPreferences, onPreferencesChange }) {
  const [aiInsights, setAiInsights] = useState([]);
  const [userBehavior, setUserBehavior] = useState({
    clickPatterns: [],
    timeSpent: {},
    preferences: {}
  });
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    // Simulate AI-powered insights generation
    generateAIInsights();
    analyzeUserBehavior();
    generateRecommendations();
  }, [userPreferences]);

  const generateAIInsights = () => {
    // Simulate AI analysis
    const insights = [
      {
        id: 1,
        type: 'productivity',
        title: 'Peak Productivity Hours',
        description: 'You are most productive between 9 AM - 11 AM',
        confidence: 0.92,
        action: 'Schedule important tasks during this time'
      },
      {
        id: 2,
        type: 'preference',
        title: 'Interface Adaptation',
        description: 'Dark mode usage increased by 40% this week',
        confidence: 0.87,
        action: 'Consider setting dark mode as default'
      },
      {
        id: 3,
        type: 'workflow',
        title: 'Workflow Optimization',
        description: 'Voice commands could save 15 minutes daily',
        confidence: 0.78,
        action: 'Enable voice shortcuts for frequent actions'
      }
    ];
    setAiInsights(insights);
  };

  const analyzeUserBehavior = () => {
    // Simulate user behavior analysis
    const behavior = {
      clickPatterns: [
        { area: 'Dashboard', frequency: 45 },
        { area: 'Reports', frequency: 32 },
        { area: 'Settings', frequency: 18 },
        { area: 'Help', frequency: 5 }
      ],
      timeSpent: {
        dashboard: '2h 15m',
        reports: '1h 45m',
        settings: '25m',
        help: '10m'
      },
      preferences: {
        theme: userPreferences.theme,
        language: userPreferences.language,
        voiceEnabled: userPreferences.personalization.voiceEnabled
      }
    };
    setUserBehavior(behavior);
  };

  const generateRecommendations = () => {
    // AI-powered recommendations
    const recs = [
      {
        id: 1,
        title: 'Enable Smart Notifications',
        description: 'Get AI-curated updates based on your work patterns',
        impact: 'High',
        effort: 'Low'
      },
      {
        id: 2,
        title: 'Customize Quick Actions',
        description: 'Add frequently used functions to your toolbar',
        impact: 'Medium',
        effort: 'Low'
      },
      {
        id: 3,
        title: 'Voice Command Training',
        description: 'Learn advanced voice commands for faster navigation',
        impact: 'High',
        effort: 'Medium'
      }
    ];
    setRecommendations(recs);
  };

  const handlePreferenceUpdate = (key, value) => {
    const updatedPreferences = {
      ...userPreferences,
      [key]: value
    };
    onPreferencesChange(updatedPreferences);
  };

  const handlePersonalizationUpdate = (key, value) => {
    const updatedPreferences = {
      ...userPreferences,
      personalization: {
        ...userPreferences.personalization,
        [key]: value
      }
    };
    onPreferencesChange(updatedPreferences);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">AI Personalization Dashboard</h1>
          <p className="text-muted-foreground">
            Intelligent insights and adaptive interface powered by machine learning
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-2 rounded-lg bg-green-100 px-3 py-1 text-green-800 dark:bg-green-900 dark:text-green-200">
            <div className="h-2 w-2 rounded-full bg-green-500"></div>
            <span className="text-sm font-medium">AI Active</span>
          </div>
        </div>
      </div>

      {/* AI Insights */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {aiInsights.map((insight) => (
          <div key={insight.id} className="rounded-lg border bg-card p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">{insight.title}</h3>
              <div className="rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                {Math.round(insight.confidence * 100)}% confident
              </div>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">{insight.description}</p>
            <div className="mt-4">
              <button className="text-sm font-medium text-primary hover:underline">
                {insight.action}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* User Behavior Analytics */}
      <div className="grid gap-6 md:grid-cols-2">
        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Usage Patterns</h3>
          <div className="space-y-3">
            {userBehavior.clickPatterns.map((pattern, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm">{pattern.area}</span>
                <div className="flex items-center space-x-2">
                  <div className="h-2 w-20 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500 rounded-full"
                      style={{ width: `${(pattern.frequency / 50) * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-muted-foreground">{pattern.frequency}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6 shadow-sm">
          <h3 className="text-lg font-semibold mb-4">Time Spent</h3>
          <div className="space-y-3">
            {Object.entries(userBehavior.timeSpent).map(([area, time]) => (
              <div key={area} className="flex items-center justify-between">
                <span className="text-sm capitalize">{area}</span>
                <span className="text-sm font-medium">{time}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Personalization Settings */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4">Personalization Settings</h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-2">
            <label className="text-sm font-medium">Theme Preference</label>
            <select 
              value={userPreferences.theme}
              onChange={(e) => handlePreferenceUpdate('theme', e.target.value)}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Language</label>
            <select 
              value={userPreferences.language}
              onChange={(e) => handlePreferenceUpdate('language', e.target.value)}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            >
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
              <option value="de">German</option>
              <option value="zh">Chinese</option>
              <option value="ja">Japanese</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">AI Assistance</label>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={userPreferences.personalization.aiAssistance}
                onChange={(e) => handlePersonalizationUpdate('aiAssistance', e.target.checked)}
                className="rounded border-gray-300"
              />
              <span className="text-sm">Enable AI recommendations</span>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Voice Interface</label>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={userPreferences.personalization.voiceEnabled}
                onChange={(e) => handlePersonalizationUpdate('voiceEnabled', e.target.checked)}
                className="rounded border-gray-300"
              />
              <span className="text-sm">Enable voice commands</span>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">Adaptive UI</label>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={userPreferences.personalization.adaptiveUI}
                onChange={(e) => handlePersonalizationUpdate('adaptiveUI', e.target.checked)}
                className="rounded border-gray-300"
              />
              <span className="text-sm">Auto-adjust interface</span>
            </div>
          </div>
        </div>
      </div>

      {/* AI Recommendations */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4">AI Recommendations</h3>
        <div className="space-y-4">
          {recommendations.map((rec) => (
            <div key={rec.id} className="flex items-start justify-between rounded-lg border p-4">
              <div className="flex-1">
                <h4 className="font-medium">{rec.title}</h4>
                <p className="text-sm text-muted-foreground mt-1">{rec.description}</p>
                <div className="flex items-center space-x-4 mt-2">
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    rec.impact === 'High' ? 'bg-red-100 text-red-800' :
                    rec.impact === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {rec.impact} Impact
                  </span>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    rec.effort === 'Low' ? 'bg-green-100 text-green-800' :
                    rec.effort === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {rec.effort} Effort
                  </span>
                </div>
              </div>
              <div className="flex space-x-2">
                <button className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90">
                  Apply
                </button>
                <button className="rounded-md border px-3 py-1 text-xs font-medium hover:bg-accent">
                  Dismiss
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

