import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../theme-provider';
import {
  Brain,
  Cube,
  Mic,
  Activity,
  Settings,
  User,
  Moon,
  Sun,
  Monitor,
  Menu,
  X,
  Bell,
  Search,
  Palette,
  Accessibility,
  Zap
} from 'lucide-react';

export function AdvancedLayout({ children, userPreferences, onPreferencesChange }) {
  const { theme, setTheme, accessibility, setAccessibility, generatePersonalizedTheme } = useTheme();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');

  const navigationItems = [
    {
      name: 'AI Dashboard',
      href: '/dashboard',
      icon: Brain,
      description: 'Personalized AI-powered dashboard'
    },
    {
      name: '3D Visualization',
      href: '/3d-visualization',
      icon: Cube,
      description: 'Interactive 3D data visualization'
    },
    {
      name: 'Voice Interface',
      href: '/voice-interface',
      icon: Mic,
      description: 'Natural language voice commands'
    },
    {
      name: 'Performance',
      href: '/performance',
      icon: Activity,
      description: 'Real-time performance monitoring'
    }
  ];

  useEffect(() => {
    // Simulate real-time notifications
    const notificationInterval = setInterval(() => {
      const newNotification = {
        id: Date.now(),
        title: 'AI Insight',
        message: 'New optimization opportunity detected',
        type: 'info',
        timestamp: new Date()
      };
      setNotifications(prev => [newNotification, ...prev.slice(0, 4)]);
    }, 30000);

    return () => clearInterval(notificationInterval);
  }, []);

  const handleThemeChange = (newTheme) => {
    setTheme(newTheme);
    onPreferencesChange({ theme: newTheme });
  };

  const handleAccessibilityChange = (key, value) => {
    const newAccessibility = { ...accessibility, [key]: value };
    setAccessibility(newAccessibility);
    onPreferencesChange({ accessibility: newAccessibility });
  };

  const generateAITheme = async () => {
    try {
      // Simulate user behavior data
      const behaviorData = {
        interactions: [
          { elementColor: '221.2 83.2% 53.3%', count: 15 },
          { elementColor: '217.2 91.2% 59.8%', count: 8 }
        ],
        timeSpent: { dashboard: 3600, visualization: 1200 },
        preferences: userPreferences
      };

      const personalizedTheme = await generatePersonalizedTheme(behaviorData);
      if (personalizedTheme) {
        console.log('Generated personalized theme:', personalizedTheme);
      }
    } catch (error) {
      console.error('Failed to generate AI theme:', error);
    }
  };

  const ThemeToggle = () => (
    <div className="flex items-center space-x-2">
      <button
        onClick={() => handleThemeChange('light')}
        className={`p-2 rounded-md transition-colors ${
          theme === 'light' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
        }`}
        title="Light theme"
      >
        <Sun className="h-4 w-4" />
      </button>
      <button
        onClick={() => handleThemeChange('dark')}
        className={`p-2 rounded-md transition-colors ${
          theme === 'dark' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
        }`}
        title="Dark theme"
      >
        <Moon className="h-4 w-4" />
      </button>
      <button
        onClick={() => handleThemeChange('system')}
        className={`p-2 rounded-md transition-colors ${
          theme === 'system' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
        }`}
        title="System theme"
      >
        <Monitor className="h-4 w-4" />
      </button>
      <button
        onClick={generateAITheme}
        className="p-2 rounded-md hover:bg-muted transition-colors"
        title="Generate AI theme"
      >
        <Palette className="h-4 w-4" />
      </button>
    </div>
  );

  const AccessibilityControls = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">High Contrast</label>
        <button
          onClick={() => handleAccessibilityChange('highContrast', !accessibility.highContrast)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            accessibility.highContrast ? 'bg-primary' : 'bg-muted'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              accessibility.highContrast ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
      
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">Reduced Motion</label>
        <button
          onClick={() => handleAccessibilityChange('reducedMotion', !accessibility.reducedMotion)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            accessibility.reducedMotion ? 'bg-primary' : 'bg-muted'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              accessibility.reducedMotion ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      <div className="space-y-2">
        <label className="text-sm font-medium">Font Size</label>
        <select
          value={accessibility.fontSize}
          onChange={(e) => handleAccessibilityChange('fontSize', e.target.value)}
          className="w-full p-2 border rounded-md bg-background"
        >
          <option value="small">Small</option>
          <option value="medium">Medium</option>
          <option value="large">Large</option>
          <option value="xlarge">Extra Large</option>
        </select>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 bg-card border-r transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="flex items-center justify-between h-16 px-6 border-b">
          <h1 className="text-xl font-bold gradient-text">Nexus Advanced</h1>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-2 rounded-md hover:bg-muted"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors group ${
                  isActive
                    ? 'bg-primary text-primary-foreground shadow-glow'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                }`}
                onClick={() => setSidebarOpen(false)}
              >
                <Icon className="mr-3 h-5 w-5" />
                <div>
                  <div>{item.name}</div>
                  <div className="text-xs opacity-70">{item.description}</div>
                </div>
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t">
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-medium mb-2 flex items-center">
                <Palette className="mr-2 h-4 w-4" />
                Theme
              </h3>
              <ThemeToggle />
            </div>
            
            <div>
              <h3 className="text-sm font-medium mb-2 flex items-center">
                <Accessibility className="mr-2 h-4 w-4" />
                Accessibility
              </h3>
              <AccessibilityControls />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-card border-b px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 rounded-md hover:bg-muted"
              >
                <Menu className="h-5 w-5" />
              </button>
              
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search or ask AI..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 pr-4 py-2 w-64 border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                />
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* AI Status Indicator */}
              <div className="flex items-center space-x-2 px-3 py-1 rounded-full bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200">
                <Zap className="h-4 w-4" />
                <span className="text-sm font-medium">AI Active</span>
              </div>

              {/* Notifications */}
              <div className="relative">
                <button className="p-2 rounded-md hover:bg-muted relative">
                  <Bell className="h-5 w-5" />
                  {notifications.length > 0 && (
                    <span className="absolute -top-1 -right-1 h-4 w-4 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                      {notifications.length}
                    </span>
                  )}
                </button>
              </div>

              {/* User Menu */}
              <button className="flex items-center space-x-2 p-2 rounded-md hover:bg-muted">
                <User className="h-5 w-5" />
                <span className="text-sm font-medium">Admin</span>
              </button>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-6">
          <div className="animate-fade-in">
            {children}
          </div>
        </main>
      </div>

      {/* Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
}

