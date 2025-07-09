import React, { useState, useEffect } from 'react';
import {
  Brain,
  TrendingUp,
  Users,
  Clock,
  Target,
  Lightbulb,
  Activity,
  BarChart3,
  PieChart,
  Zap,
  Star,
  ArrowUp,
  ArrowDown,
  Eye,
  Heart,
  MessageSquare,
  Settings,
  Sparkles,
  Cpu,
  Database,
  Globe
} from 'lucide-react';

export function PersonalizationDashboard({ userPreferences, onPreferencesChange }) {
  const [aiInsights, setAiInsights] = useState([]);
  const [userBehavior, setUserBehavior] = useState({});
  const [recommendations, setRecommendations] = useState([]);
  const [adaptiveMetrics, setAdaptiveMetrics] = useState({});
  const [personalizedContent, setPersonalizedContent] = useState([]);

  useEffect(() => {
    // Initialize AI-powered personalization data
    initializePersonalizationData();
    
    // Set up real-time behavior tracking
    const behaviorTracker = setInterval(() => {
      updateUserBehavior();
      generateAIInsights();
      updateRecommendations();
    }, 5000);

    return () => clearInterval(behaviorTracker);
  }, [userPreferences]);

  const initializePersonalizationData = () => {
    // Simulate AI-generated insights
    setAiInsights([
      {
        id: 1,
        type: 'productivity',
        title: 'Peak Performance Window Detected',
        description: 'Your productivity is 35% higher between 9-11 AM. Consider scheduling important tasks during this time.',
        confidence: 0.92,
        impact: 'high',
        actionable: true
      },
      {
        id: 2,
        type: 'workflow',
        title: 'Workflow Optimization Opportunity',
        description: 'Automating your data export process could save 2.5 hours per week.',
        confidence: 0.87,
        impact: 'medium',
        actionable: true
      },
      {
        id: 3,
        type: 'collaboration',
        title: 'Team Collaboration Pattern',
        description: 'Your team responds 40% faster to visual communications than text-only messages.',
        confidence: 0.94,
        impact: 'medium',
        actionable: false
      }
    ]);

    // Simulate user behavior analytics
    setUserBehavior({
      sessionDuration: 145, // minutes
      interactionsPerHour: 42,
      preferredFeatures: ['dashboard', 'analytics', 'reports'],
      timeDistribution: {
        dashboard: 35,
        analytics: 28,
        reports: 20,
        settings: 10,
        other: 7
      },
      engagementScore: 8.7,
      satisfactionScore: 9.2
    });

    // Simulate AI recommendations
    setRecommendations([
      {
        id: 1,
        category: 'interface',
        title: 'Adaptive Layout Suggestion',
        description: 'Switch to compact view for better information density',
        priority: 'high',
        estimatedBenefit: '15% faster navigation',
        action: 'apply_compact_layout'
      },
      {
        id: 2,
        category: 'workflow',
        title: 'Smart Shortcuts',
        description: 'Enable keyboard shortcuts for your most-used actions',
        priority: 'medium',
        estimatedBenefit: '25% time savings',
        action: 'enable_shortcuts'
      },
      {
        id: 3,
        category: 'content',
        title: 'Personalized Dashboard',
        description: 'Customize widget order based on your usage patterns',
        priority: 'low',
        estimatedBenefit: '10% efficiency gain',
        action: 'reorder_widgets'
      }
    ]);

    // Simulate adaptive metrics
    setAdaptiveMetrics({
      adaptationAccuracy: 92,
      userSatisfaction: 94,
      timeToAdapt: 2.3, // seconds
      improvementRate: 15, // percentage
      learningProgress: 78
    });

    // Simulate personalized content
    setPersonalizedContent([
      {
        id: 1,
        type: 'insight',
        title: 'Your Weekly Performance Summary',
        content: 'You completed 23% more tasks this week compared to last week. Your focus time increased by 1.2 hours.',
        relevanceScore: 0.95
      },
      {
        id: 2,
        type: 'tip',
        title: 'Productivity Tip',
        content: 'Based on your patterns, taking a 5-minute break every 45 minutes could boost your efficiency by 12%.',
        relevanceScore: 0.88
      },
      {
        id: 3,
        type: 'update',
        title: 'Feature Recommendation',
        content: 'The new voice commands feature aligns with your preference for hands-free operation.',
        relevanceScore: 0.91
      }
    ]);
  };

  const updateUserBehavior = () => {
    // Simulate real-time behavior updates
    setUserBehavior(prev => ({
      ...prev,
      sessionDuration: prev.sessionDuration + Math.random() * 2,
      interactionsPerHour: prev.interactionsPerHour + Math.floor(Math.random() * 3 - 1),
      engagementScore: Math.min(10, prev.engagementScore + (Math.random() - 0.5) * 0.1)
    }));
  };

  const generateAIInsights = () => {
    // Simulate AI generating new insights
    const newInsights = [
      'Your attention span is optimal during morning hours',
      'Visual elements increase your task completion rate by 23%',
      'You prefer detailed analytics over summary views',
      'Collaborative features boost your productivity by 18%'
    ];

    if (Math.random() > 0.7) {
      const randomInsight = newInsights[Math.floor(Math.random() * newInsights.length)];
      setAiInsights(prev => [
        {
          id: Date.now(),
          type: 'behavior',
          title: 'New Behavioral Pattern Detected',
          description: randomInsight,
          confidence: 0.75 + Math.random() * 0.2,
          impact: Math.random() > 0.5 ? 'medium' : 'low',
          actionable: Math.random() > 0.5
        },
        ...prev.slice(0, 4)
      ]);
    }
  };

  const updateRecommendations = () => {
    // Simulate dynamic recommendation updates
    setRecommendations(prev => 
      prev.map(rec => ({
        ...rec,
        priority: Math.random() > 0.8 ? 
          (rec.priority === 'high' ? 'medium' : rec.priority === 'medium' ? 'low' : 'high') : 
          rec.priority
      }))
    );
  };

  const applyRecommendation = (recommendation) => {
    console.log('Applying recommendation:', recommendation);
    
    switch (recommendation.action) {
      case 'apply_compact_layout':
        onPreferencesChange({ layout: 'compact' });
        break;
      case 'enable_shortcuts':
        onPreferencesChange({ shortcuts: true });
        break;
      case 'reorder_widgets':
        onPreferencesChange({ customWidgetOrder: true });
        break;
      default:
        console.log('Unknown recommendation action');
    }

    // Remove applied recommendation
    setRecommendations(prev => prev.filter(r => r.id !== recommendation.id));
  };

  const MetricCard = ({ title, value, change, icon: Icon, color = 'blue' }) => (
    <div className="bg-card rounded-lg p-6 border shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold">{value}</p>
          {change && (
            <div className={`flex items-center mt-1 text-sm ${
              change > 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {change > 0 ? <ArrowUp className="h-4 w-4 mr-1" /> : <ArrowDown className="h-4 w-4 mr-1" />}
              {Math.abs(change)}%
            </div>
          )}
        </div>
        <div className={`p-3 rounded-full bg-${color}-100 dark:bg-${color}-900`}>
          <Icon className={`h-6 w-6 text-${color}-600 dark:text-${color}-400`} />
        </div>
      </div>
    </div>
  );

  const InsightCard = ({ insight }) => (
    <div className="bg-card rounded-lg p-6 border shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Brain className="h-5 w-5 text-primary" />
          <span className="font-medium">{insight.title}</span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">
            {Math.round(insight.confidence * 100)}% confidence
          </span>
          <span className={`text-xs px-2 py-1 rounded-full ${
            insight.impact === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
            insight.impact === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
            'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
          }`}>
            {insight.impact} impact
          </span>
        </div>
      </div>
      <p className="text-muted-foreground text-sm mb-3">{insight.description}</p>
      {insight.actionable && (
        <button className="text-primary hover:text-primary/80 text-sm font-medium">
          Take Action →
        </button>
      )}
    </div>
  );

  const RecommendationCard = ({ recommendation }) => (
    <div className="bg-card rounded-lg p-6 border shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-medium">{recommendation.title}</h3>
          <p className="text-sm text-muted-foreground mt-1">{recommendation.description}</p>
        </div>
        <span className={`text-xs px-2 py-1 rounded-full ${
          recommendation.priority === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
          recommendation.priority === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
          'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
        }`}>
          {recommendation.priority}
        </span>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-sm text-green-600 dark:text-green-400">
          {recommendation.estimatedBenefit}
        </span>
        <button
          onClick={() => applyRecommendation(recommendation)}
          className="px-3 py-1 bg-primary text-primary-foreground rounded-md text-sm hover:bg-primary/90 transition-colors"
        >
          Apply
        </button>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">AI-Powered Personalization</h1>
          <p className="text-muted-foreground mt-1">
            Intelligent insights and adaptive interface optimization
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-2 px-3 py-2 bg-green-100 dark:bg-green-900 rounded-lg">
            <Sparkles className="h-4 w-4 text-green-600 dark:text-green-400" />
            <span className="text-sm font-medium text-green-800 dark:text-green-200">
              AI Learning Active
            </span>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Adaptation Accuracy"
          value={`${adaptiveMetrics.adaptationAccuracy}%`}
          change={5}
          icon={Target}
          color="blue"
        />
        <MetricCard
          title="User Satisfaction"
          value={`${adaptiveMetrics.userSatisfaction}%`}
          change={3}
          icon={Heart}
          color="green"
        />
        <MetricCard
          title="Learning Progress"
          value={`${adaptiveMetrics.learningProgress}%`}
          change={8}
          icon={TrendingUp}
          color="purple"
        />
        <MetricCard
          title="Engagement Score"
          value={userBehavior.engagementScore?.toFixed(1)}
          change={2}
          icon={Activity}
          color="orange"
        />
      </div>

      {/* AI Insights */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <Brain className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">AI-Generated Insights</h2>
          <span className="text-sm text-muted-foreground">
            ({aiInsights.length} active insights)
          </span>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {aiInsights.map(insight => (
            <InsightCard key={insight.id} insight={insight} />
          ))}
        </div>
      </div>

      {/* Recommendations */}
      <div>
        <div className="flex items-center space-x-2 mb-4">
          <Lightbulb className="h-5 w-5 text-primary" />
          <h2 className="text-xl font-semibold">Smart Recommendations</h2>
          <span className="text-sm text-muted-foreground">
            ({recommendations.length} pending)
          </span>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {recommendations.map(recommendation => (
            <RecommendationCard key={recommendation.id} recommendation={recommendation} />
          ))}
        </div>
      </div>

      {/* Behavior Analytics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card rounded-lg p-6 border">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2 text-primary" />
            Usage Patterns
          </h3>
          <div className="space-y-4">
            {Object.entries(userBehavior.timeDistribution || {}).map(([feature, percentage]) => (
              <div key={feature} className="flex items-center justify-between">
                <span className="text-sm capitalize">{feature}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-24 bg-muted rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all duration-300"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                  <span className="text-sm text-muted-foreground w-8">{percentage}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-card rounded-lg p-6 border">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <MessageSquare className="h-5 w-5 mr-2 text-primary" />
            Personalized Content
          </h3>
          <div className="space-y-4">
            {personalizedContent.map(content => (
              <div key={content.id} className="border-l-4 border-primary pl-4">
                <h4 className="font-medium text-sm">{content.title}</h4>
                <p className="text-xs text-muted-foreground mt-1">{content.content}</p>
                <div className="flex items-center mt-2">
                  <Star className="h-3 w-3 text-yellow-500 mr-1" />
                  <span className="text-xs text-muted-foreground">
                    {Math.round(content.relevanceScore * 100)}% relevant
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* System Performance */}
      <div className="bg-card rounded-lg p-6 border">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Cpu className="h-5 w-5 mr-2 text-primary" />
          AI System Performance
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary">{adaptiveMetrics.timeToAdapt}s</div>
            <div className="text-sm text-muted-foreground">Adaptation Time</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{adaptiveMetrics.improvementRate}%</div>
            <div className="text-sm text-muted-foreground">Performance Improvement</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{userBehavior.interactionsPerHour}</div>
            <div className="text-sm text-muted-foreground">Interactions/Hour</div>
          </div>
        </div>
      </div>
    </div>
  );
}

