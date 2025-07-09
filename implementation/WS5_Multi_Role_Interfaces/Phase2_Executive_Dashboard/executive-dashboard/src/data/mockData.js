// Mock data for Executive Dashboard
export const executiveKPIs = {
  developmentVelocity: {
    current: 87,
    previous: 82,
    trend: 'up',
    target: 90,
    unit: '%'
  },
  technicalDebt: {
    current: 23,
    previous: 28,
    trend: 'down',
    target: 20,
    unit: '%'
  },
  securityPosture: {
    current: 94,
    previous: 91,
    trend: 'up',
    target: 95,
    unit: '%'
  },
  teamProductivity: {
    current: 91,
    previous: 89,
    trend: 'up',
    target: 92,
    unit: '%'
  }
}

export const businessMetrics = {
  roi: {
    current: 245,
    previous: 220,
    trend: 'up',
    target: 250,
    unit: '%'
  },
  costSavings: {
    current: 2.4,
    previous: 2.1,
    trend: 'up',
    target: 2.5,
    unit: 'M'
  },
  timeToMarket: {
    current: 14,
    previous: 18,
    trend: 'down',
    target: 12,
    unit: 'days'
  },
  customerSatisfaction: {
    current: 4.7,
    previous: 4.5,
    trend: 'up',
    target: 4.8,
    unit: '/5'
  }
}

export const systemHealth = {
  uptime: {
    current: 99.97,
    previous: 99.94,
    trend: 'up',
    target: 99.95,
    unit: '%'
  },
  responseTime: {
    current: 145,
    previous: 167,
    trend: 'down',
    target: 150,
    unit: 'ms'
  },
  errorRate: {
    current: 0.03,
    previous: 0.05,
    trend: 'down',
    target: 0.02,
    unit: '%'
  },
  throughput: {
    current: 12500,
    previous: 11800,
    trend: 'up',
    target: 13000,
    unit: 'req/min'
  }
}

export const developmentTrends = [
  { month: 'Jan', velocity: 78, quality: 85, deployment: 92 },
  { month: 'Feb', velocity: 82, quality: 87, deployment: 94 },
  { month: 'Mar', velocity: 79, quality: 89, deployment: 91 },
  { month: 'Apr', velocity: 85, quality: 91, deployment: 96 },
  { month: 'May', velocity: 88, quality: 93, deployment: 98 },
  { month: 'Jun', velocity: 87, quality: 94, deployment: 97 }
]

export const costAnalysis = [
  { category: 'Development', current: 1.2, previous: 1.4, budget: 1.3 },
  { category: 'Infrastructure', current: 0.8, previous: 0.9, budget: 0.85 },
  { category: 'Security', current: 0.3, previous: 0.35, budget: 0.32 },
  { category: 'Operations', current: 0.5, previous: 0.6, budget: 0.55 },
  { category: 'Training', current: 0.2, previous: 0.25, budget: 0.22 }
]

export const riskAssessment = [
  {
    category: 'Security',
    level: 'Medium',
    impact: 'High',
    probability: 'Low',
    mitigation: 'Enhanced monitoring deployed',
    status: 'In Progress'
  },
  {
    category: 'Performance',
    level: 'Low',
    impact: 'Medium',
    probability: 'Low',
    mitigation: 'Auto-scaling configured',
    status: 'Completed'
  },
  {
    category: 'Compliance',
    level: 'High',
    impact: 'High',
    probability: 'Medium',
    mitigation: 'Audit scheduled for Q2',
    status: 'Planned'
  },
  {
    category: 'Technical Debt',
    level: 'Medium',
    impact: 'Medium',
    probability: 'High',
    mitigation: 'Refactoring sprint planned',
    status: 'In Progress'
  }
]

export const teamPerformance = [
  { team: 'Frontend', productivity: 92, satisfaction: 4.6, velocity: 89 },
  { team: 'Backend', productivity: 88, satisfaction: 4.4, velocity: 91 },
  { team: 'DevOps', productivity: 95, satisfaction: 4.8, velocity: 87 },
  { team: 'QA', productivity: 90, satisfaction: 4.5, velocity: 93 },
  { team: 'Security', productivity: 87, satisfaction: 4.3, velocity: 85 }
]

export const incidentData = [
  { date: '2024-01-01', critical: 0, high: 1, medium: 3, low: 5 },
  { date: '2024-01-02', critical: 1, high: 0, medium: 2, low: 4 },
  { date: '2024-01-03', critical: 0, high: 2, medium: 1, low: 6 },
  { date: '2024-01-04', critical: 0, high: 1, medium: 4, low: 3 },
  { date: '2024-01-05', critical: 0, high: 0, medium: 2, low: 7 },
  { date: '2024-01-06', critical: 0, high: 1, medium: 3, low: 4 },
  { date: '2024-01-07', critical: 0, high: 0, medium: 1, low: 5 }
]

export const capacityPlanning = {
  cpu: { current: 67, projected: 78, capacity: 100, alert: 85 },
  memory: { current: 72, projected: 84, capacity: 100, alert: 90 },
  storage: { current: 45, projected: 58, capacity: 100, alert: 80 },
  network: { current: 34, projected: 42, capacity: 100, alert: 75 }
}

export const executiveSummary = {
  title: "Q1 2024 Technology Performance Summary",
  generatedAt: new Date().toISOString(),
  keyInsights: [
    "Development velocity increased 6% quarter-over-quarter, exceeding targets",
    "Technical debt reduced by 18% through focused refactoring initiatives",
    "Security posture improved with 94% compliance score, up from 91%",
    "Cost optimization efforts resulted in $300K quarterly savings"
  ],
  recommendations: [
    "Invest in automated testing infrastructure to maintain velocity gains",
    "Continue technical debt reduction with dedicated 20% sprint allocation",
    "Implement advanced threat detection to reach 95% security target",
    "Scale successful cost optimization practices to additional teams"
  ],
  actionItems: [
    {
      item: "Deploy automated testing pipeline",
      owner: "Engineering Team",
      dueDate: "2024-02-15",
      priority: "High"
    },
    {
      item: "Complete security audit",
      owner: "Security Team", 
      dueDate: "2024-02-28",
      priority: "Critical"
    },
    {
      item: "Finalize Q2 budget allocation",
      owner: "Finance Team",
      dueDate: "2024-02-10",
      priority: "Medium"
    }
  ]
}

