// Mock data for Developer Dashboard
// Nexus Architect - WS5 Phase 3

export const developerMetrics = {
  productivity: {
    linesOfCode: 2847,
    linesOfCodeChange: 12.5,
    commitsToday: 8,
    commitsChange: 33.3,
    pullRequests: 3,
    pullRequestsChange: -25.0,
    codeReviews: 5,
    codeReviewsChange: 66.7,
    testsWritten: 24,
    testsChange: 20.0,
    bugsFixed: 7,
    bugsChange: -12.5
  },
  codeQuality: {
    overallScore: 87,
    scoreChange: 3.2,
    coverage: 94.2,
    coverageChange: 1.8,
    complexity: 2.3,
    complexityChange: -0.2,
    duplication: 1.8,
    duplicationChange: -0.5,
    maintainability: 'A',
    reliability: 'A',
    security: 'B+'
  },
  performance: {
    buildTime: '2m 34s',
    buildTimeChange: -8.2,
    testTime: '45s',
    testTimeChange: -12.1,
    deployTime: '3m 12s',
    deployTimeChange: -15.3,
    errorRate: 0.12,
    errorRateChange: -45.5
  }
}

export const recentActivity = [
  {
    id: 1,
    type: 'commit',
    message: 'feat: Add user authentication middleware',
    repository: 'nexus-backend',
    branch: 'feature/auth-middleware',
    timestamp: '2 minutes ago',
    status: 'success',
    author: 'Alex Chen'
  },
  {
    id: 2,
    type: 'pull_request',
    message: 'fix: Resolve memory leak in data processing',
    repository: 'nexus-core',
    branch: 'bugfix/memory-leak',
    timestamp: '15 minutes ago',
    status: 'pending',
    author: 'Alex Chen',
    reviewers: ['Sarah Kim', 'Mike Johnson']
  },
  {
    id: 3,
    type: 'deployment',
    message: 'Deploy v2.1.3 to staging environment',
    repository: 'nexus-frontend',
    branch: 'main',
    timestamp: '1 hour ago',
    status: 'success',
    author: 'CI/CD Pipeline'
  },
  {
    id: 4,
    type: 'code_review',
    message: 'Review: Implement caching layer for API responses',
    repository: 'nexus-api',
    branch: 'feature/response-caching',
    timestamp: '2 hours ago',
    status: 'approved',
    author: 'Jordan Lee',
    reviewer: 'Alex Chen'
  },
  {
    id: 5,
    type: 'test',
    message: 'Unit tests for payment processing module',
    repository: 'nexus-payments',
    branch: 'feature/payment-tests',
    timestamp: '3 hours ago',
    status: 'failed',
    author: 'Alex Chen',
    details: '2 tests failing, 18 passing'
  }
]

export const codeQualityTrends = [
  { date: '2024-01-01', coverage: 89.2, complexity: 2.8, duplication: 2.1, score: 82 },
  { date: '2024-01-02', coverage: 90.1, complexity: 2.7, duplication: 2.0, score: 83 },
  { date: '2024-01-03', coverage: 91.3, complexity: 2.6, duplication: 1.9, score: 84 },
  { date: '2024-01-04', coverage: 92.0, complexity: 2.5, duplication: 1.8, score: 85 },
  { date: '2024-01-05', coverage: 92.8, complexity: 2.4, duplication: 1.8, score: 86 },
  { date: '2024-01-06', coverage: 93.5, complexity: 2.3, duplication: 1.8, score: 87 },
  { date: '2024-01-07', coverage: 94.2, complexity: 2.3, duplication: 1.8, score: 87 }
]

export const productivityTrends = [
  { date: '2024-01-01', commits: 12, linesOfCode: 2156, pullRequests: 2, codeReviews: 4 },
  { date: '2024-01-02', commits: 8, linesOfCode: 1987, pullRequests: 3, codeReviews: 3 },
  { date: '2024-01-03', commits: 15, linesOfCode: 2543, pullRequests: 1, codeReviews: 5 },
  { date: '2024-01-04', commits: 10, linesOfCode: 2234, pullRequests: 4, codeReviews: 2 },
  { date: '2024-01-05', commits: 6, linesOfCode: 1876, pullRequests: 2, codeReviews: 6 },
  { date: '2024-01-06', commits: 11, linesOfCode: 2456, pullRequests: 3, codeReviews: 4 },
  { date: '2024-01-07', commits: 8, linesOfCode: 2847, pullRequests: 3, codeReviews: 5 }
]

export const repositories = [
  {
    name: 'nexus-frontend',
    language: 'TypeScript',
    coverage: 96.2,
    lastCommit: '2 hours ago',
    status: 'healthy',
    issues: 3,
    pullRequests: 2,
    contributors: 8
  },
  {
    name: 'nexus-backend',
    language: 'Python',
    coverage: 94.8,
    lastCommit: '2 minutes ago',
    status: 'healthy',
    issues: 1,
    pullRequests: 4,
    contributors: 6
  },
  {
    name: 'nexus-core',
    language: 'JavaScript',
    coverage: 91.5,
    lastCommit: '15 minutes ago',
    status: 'warning',
    issues: 7,
    pullRequests: 1,
    contributors: 12
  },
  {
    name: 'nexus-api',
    language: 'Node.js',
    coverage: 89.3,
    lastCommit: '1 hour ago',
    status: 'healthy',
    issues: 2,
    pullRequests: 3,
    contributors: 5
  },
  {
    name: 'nexus-mobile',
    language: 'React Native',
    coverage: 87.1,
    lastCommit: '4 hours ago',
    status: 'critical',
    issues: 12,
    pullRequests: 0,
    contributors: 4
  }
]

export const technicalDebt = [
  {
    id: 1,
    title: 'Refactor legacy authentication system',
    severity: 'high',
    effort: '3 days',
    impact: 'Security & Performance',
    repository: 'nexus-backend',
    file: 'auth/legacy_auth.py',
    lines: 245,
    created: '2024-01-01'
  },
  {
    id: 2,
    title: 'Remove deprecated API endpoints',
    severity: 'medium',
    effort: '1 day',
    impact: 'Maintainability',
    repository: 'nexus-api',
    file: 'routes/deprecated.js',
    lines: 89,
    created: '2024-01-03'
  },
  {
    id: 3,
    title: 'Update outdated dependencies',
    severity: 'medium',
    effort: '2 days',
    impact: 'Security',
    repository: 'nexus-frontend',
    file: 'package.json',
    lines: 0,
    created: '2024-01-05'
  },
  {
    id: 4,
    title: 'Optimize database queries in user service',
    severity: 'low',
    effort: '4 hours',
    impact: 'Performance',
    repository: 'nexus-backend',
    file: 'services/user_service.py',
    lines: 67,
    created: '2024-01-06'
  }
]

export const learningRecommendations = [
  {
    id: 1,
    title: 'Advanced React Patterns',
    type: 'course',
    provider: 'Nexus Learning',
    duration: '4 hours',
    difficulty: 'Advanced',
    relevance: 95,
    description: 'Learn advanced React patterns including render props, higher-order components, and hooks.',
    skills: ['React', 'JavaScript', 'Frontend'],
    progress: 0
  },
  {
    id: 2,
    title: 'Python Performance Optimization',
    type: 'workshop',
    provider: 'Internal Training',
    duration: '2 hours',
    difficulty: 'Intermediate',
    relevance: 88,
    description: 'Techniques for optimizing Python code performance and memory usage.',
    skills: ['Python', 'Performance', 'Backend'],
    progress: 25
  },
  {
    id: 3,
    title: 'Microservices Architecture Best Practices',
    type: 'documentation',
    provider: 'Architecture Team',
    duration: '1 hour',
    difficulty: 'Advanced',
    relevance: 92,
    description: 'Guidelines and best practices for designing and implementing microservices.',
    skills: ['Architecture', 'Microservices', 'Design'],
    progress: 60
  },
  {
    id: 4,
    title: 'Docker and Kubernetes Fundamentals',
    type: 'tutorial',
    provider: 'DevOps Team',
    duration: '3 hours',
    difficulty: 'Beginner',
    relevance: 78,
    description: 'Introduction to containerization and orchestration with Docker and Kubernetes.',
    skills: ['Docker', 'Kubernetes', 'DevOps'],
    progress: 0
  }
]

export const ideIntegrations = [
  {
    name: 'VS Code Extension',
    status: 'installed',
    version: '2.1.3',
    lastUpdated: '2024-01-07',
    features: [
      'Real-time code analysis',
      'Contextual documentation',
      'Code review assistance',
      'AI-powered suggestions'
    ],
    usage: {
      daily: 8.5,
      weekly: 42.3,
      suggestions: 156,
      accepted: 89
    }
  },
  {
    name: 'IntelliJ IDEA Plugin',
    status: 'available',
    version: '1.8.2',
    lastUpdated: '2024-01-05',
    features: [
      'Code quality analysis',
      'Refactoring suggestions',
      'Project structure optimization',
      'Performance insights'
    ],
    usage: {
      daily: 0,
      weekly: 0,
      suggestions: 0,
      accepted: 0
    }
  },
  {
    name: 'Web IDE',
    status: 'beta',
    version: '0.9.1',
    lastUpdated: '2024-01-06',
    features: [
      'Cloud-based editing',
      'Real-time collaboration',
      'Version control integration',
      'Live preview'
    ],
    usage: {
      daily: 2.1,
      weekly: 8.7,
      suggestions: 23,
      accepted: 18
    }
  }
]

export const workflowOptimizations = [
  {
    id: 1,
    title: 'Automate code formatting on commit',
    category: 'Code Quality',
    impact: 'High',
    effort: 'Low',
    timeSaved: '15 min/day',
    description: 'Set up pre-commit hooks to automatically format code using Prettier and ESLint.',
    status: 'recommended',
    implementation: 'Add pre-commit configuration to repository'
  },
  {
    id: 2,
    title: 'Implement parallel test execution',
    category: 'Testing',
    impact: 'High',
    effort: 'Medium',
    timeSaved: '5 min/build',
    description: 'Configure test runner to execute tests in parallel, reducing overall test time.',
    status: 'in_progress',
    implementation: 'Update Jest configuration for parallel execution'
  },
  {
    id: 3,
    title: 'Set up automated dependency updates',
    category: 'Maintenance',
    impact: 'Medium',
    effort: 'Low',
    timeSaved: '2 hours/week',
    description: 'Use Dependabot to automatically create PRs for dependency updates.',
    status: 'completed',
    implementation: 'Dependabot configuration added to all repositories'
  },
  {
    id: 4,
    title: 'Optimize Docker build caching',
    category: 'DevOps',
    impact: 'Medium',
    effort: 'Medium',
    timeSaved: '3 min/build',
    description: 'Improve Docker build performance by optimizing layer caching strategy.',
    status: 'recommended',
    implementation: 'Restructure Dockerfile for better caching'
  }
]

export const aiSuggestions = [
  {
    id: 1,
    type: 'code_improvement',
    title: 'Optimize database query in getUserProfile',
    file: 'services/user_service.py',
    line: 45,
    severity: 'medium',
    description: 'This query can be optimized by adding an index on user_id and using select_related.',
    suggestion: 'Add database index and use select_related for better performance',
    confidence: 92,
    estimatedImpact: '40% faster query execution'
  },
  {
    id: 2,
    type: 'security',
    title: 'Potential SQL injection vulnerability',
    file: 'api/search.js',
    line: 23,
    severity: 'high',
    description: 'User input is directly concatenated into SQL query without sanitization.',
    suggestion: 'Use parameterized queries or ORM methods to prevent SQL injection',
    confidence: 98,
    estimatedImpact: 'Eliminates security vulnerability'
  },
  {
    id: 3,
    type: 'refactoring',
    title: 'Extract common validation logic',
    file: 'components/forms/UserForm.jsx',
    line: 78,
    severity: 'low',
    description: 'Validation logic is duplicated across multiple form components.',
    suggestion: 'Create a custom hook for form validation to reduce code duplication',
    confidence: 85,
    estimatedImpact: '30% reduction in form validation code'
  },
  {
    id: 4,
    type: 'performance',
    title: 'Unnecessary re-renders in component',
    file: 'components/Dashboard.jsx',
    line: 156,
    severity: 'medium',
    description: 'Component re-renders on every state change due to missing dependencies.',
    suggestion: 'Use useMemo and useCallback to optimize component performance',
    confidence: 88,
    estimatedImpact: '25% reduction in render time'
  }
]

