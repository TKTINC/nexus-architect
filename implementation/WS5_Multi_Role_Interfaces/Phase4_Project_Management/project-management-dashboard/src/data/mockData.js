// Mock data for Project Management Dashboard
// Nexus Architect - WS5 Phase 4

export const projects = [
  {
    id: 1,
    name: 'Nexus Architect Core Platform',
    description: 'Development of the core Nexus Architect platform with AI intelligence and autonomous capabilities',
    status: 'active',
    priority: 'high',
    progress: 78,
    startDate: '2024-01-01',
    endDate: '2024-03-31',
    budget: 2800000,
    spent: 1950000,
    manager: 'Sarah Chen',
    team: ['John Doe', 'Alice Johnson', 'Bob Wilson', 'Emma Davis', 'Michael Brown'],
    milestones: [
      { id: 1, name: 'Foundation Setup', date: '2024-01-15', status: 'completed' },
      { id: 2, name: 'AI Intelligence Module', date: '2024-02-15', status: 'completed' },
      { id: 3, name: 'Data Ingestion System', date: '2024-02-28', status: 'in-progress' },
      { id: 4, name: 'Autonomous Capabilities', date: '2024-03-15', status: 'pending' },
      { id: 5, name: 'User Interfaces', date: '2024-03-31', status: 'pending' }
    ],
    risks: [
      { id: 1, description: 'Integration complexity with legacy systems', severity: 'medium', probability: 'high' },
      { id: 2, description: 'Resource availability for Q1', severity: 'low', probability: 'medium' }
    ]
  },
  {
    id: 2,
    name: 'Mobile Application Development',
    description: 'Native mobile applications for iOS and Android platforms',
    status: 'planning',
    priority: 'medium',
    progress: 25,
    startDate: '2024-02-01',
    endDate: '2024-05-31',
    budget: 850000,
    spent: 125000,
    manager: 'David Kim',
    team: ['Lisa Wang', 'Tom Anderson', 'Maria Garcia'],
    milestones: [
      { id: 1, name: 'Requirements Analysis', date: '2024-02-15', status: 'completed' },
      { id: 2, name: 'UI/UX Design', date: '2024-03-01', status: 'in-progress' },
      { id: 3, name: 'iOS Development', date: '2024-04-15', status: 'pending' },
      { id: 4, name: 'Android Development', date: '2024-04-30', status: 'pending' },
      { id: 5, name: 'Testing & Launch', date: '2024-05-31', status: 'pending' }
    ],
    risks: [
      { id: 1, description: 'App store approval delays', severity: 'medium', probability: 'medium' },
      { id: 2, description: 'Cross-platform compatibility issues', severity: 'high', probability: 'low' }
    ]
  },
  {
    id: 3,
    name: 'Security Enhancement Initiative',
    description: 'Comprehensive security audit and enhancement across all systems',
    status: 'completed',
    priority: 'high',
    progress: 100,
    startDate: '2023-11-01',
    endDate: '2024-01-15',
    budget: 450000,
    spent: 425000,
    manager: 'Alex Rodriguez',
    team: ['Jennifer Lee', 'Carlos Martinez', 'Rachel Green'],
    milestones: [
      { id: 1, name: 'Security Audit', date: '2023-11-30', status: 'completed' },
      { id: 2, name: 'Vulnerability Assessment', date: '2023-12-15', status: 'completed' },
      { id: 3, name: 'Security Implementation', date: '2024-01-05', status: 'completed' },
      { id: 4, name: 'Penetration Testing', date: '2024-01-15', status: 'completed' }
    ],
    risks: []
  }
]

export const tasks = [
  {
    id: 1,
    title: 'Implement user authentication system',
    description: 'Design and implement secure user authentication with OAuth2 and JWT tokens',
    projectId: 1,
    assignee: 'John Doe',
    status: 'in-progress',
    priority: 'high',
    dueDate: '2024-01-20',
    estimatedHours: 40,
    actualHours: 28,
    tags: ['backend', 'security', 'authentication'],
    dependencies: [],
    subtasks: [
      { id: 1, title: 'OAuth2 integration', completed: true },
      { id: 2, title: 'JWT token management', completed: true },
      { id: 3, title: 'User session handling', completed: false },
      { id: 4, title: 'Security testing', completed: false }
    ]
  },
  {
    id: 2,
    title: 'Design project dashboard UI',
    description: 'Create responsive dashboard interface for project management',
    projectId: 1,
    assignee: 'Alice Johnson',
    status: 'completed',
    priority: 'medium',
    dueDate: '2024-01-15',
    estimatedHours: 32,
    actualHours: 35,
    tags: ['frontend', 'ui', 'dashboard'],
    dependencies: [],
    subtasks: [
      { id: 1, title: 'Wireframe design', completed: true },
      { id: 2, title: 'Component development', completed: true },
      { id: 3, title: 'Responsive testing', completed: true },
      { id: 4, title: 'User feedback integration', completed: true }
    ]
  },
  {
    id: 3,
    title: 'Database optimization',
    description: 'Optimize database queries and implement caching strategies',
    projectId: 1,
    assignee: 'Bob Wilson',
    status: 'pending',
    priority: 'medium',
    dueDate: '2024-01-25',
    estimatedHours: 24,
    actualHours: 0,
    tags: ['database', 'performance', 'optimization'],
    dependencies: [1],
    subtasks: [
      { id: 1, title: 'Query analysis', completed: false },
      { id: 2, title: 'Index optimization', completed: false },
      { id: 3, title: 'Caching implementation', completed: false },
      { id: 4, title: 'Performance testing', completed: false }
    ]
  },
  {
    id: 4,
    title: 'Mobile app wireframes',
    description: 'Create wireframes for mobile application user interface',
    projectId: 2,
    assignee: 'Lisa Wang',
    status: 'in-progress',
    priority: 'high',
    dueDate: '2024-02-10',
    estimatedHours: 20,
    actualHours: 12,
    tags: ['mobile', 'design', 'wireframes'],
    dependencies: [],
    subtasks: [
      { id: 1, title: 'User flow mapping', completed: true },
      { id: 2, title: 'Screen wireframes', completed: false },
      { id: 3, title: 'Navigation design', completed: false },
      { id: 4, title: 'Prototype creation', completed: false }
    ]
  }
]

export const teamMembers = [
  {
    id: 1,
    name: 'Sarah Chen',
    role: 'Project Manager',
    email: 'sarah.chen@nexusarchitect.com',
    avatar: '/avatars/sarah-chen.jpg',
    skills: ['Project Management', 'Agile', 'Risk Management', 'Team Leadership'],
    availability: 100,
    currentProjects: [1],
    workload: 85,
    performance: 92
  },
  {
    id: 2,
    name: 'John Doe',
    role: 'Senior Backend Developer',
    email: 'john.doe@nexusarchitect.com',
    avatar: '/avatars/john-doe.jpg',
    skills: ['Python', 'Node.js', 'PostgreSQL', 'Docker', 'AWS'],
    availability: 90,
    currentProjects: [1],
    workload: 78,
    performance: 88
  },
  {
    id: 3,
    name: 'Alice Johnson',
    role: 'Frontend Developer',
    email: 'alice.johnson@nexusarchitect.com',
    avatar: '/avatars/alice-johnson.jpg',
    skills: ['React', 'TypeScript', 'CSS', 'UI/UX Design'],
    availability: 100,
    currentProjects: [1],
    workload: 82,
    performance: 90
  },
  {
    id: 4,
    name: 'Bob Wilson',
    role: 'Database Engineer',
    email: 'bob.wilson@nexusarchitect.com',
    avatar: '/avatars/bob-wilson.jpg',
    skills: ['PostgreSQL', 'MongoDB', 'Redis', 'Performance Optimization'],
    availability: 80,
    currentProjects: [1],
    workload: 65,
    performance: 85
  },
  {
    id: 5,
    name: 'David Kim',
    role: 'Mobile Development Lead',
    email: 'david.kim@nexusarchitect.com',
    avatar: '/avatars/david-kim.jpg',
    skills: ['React Native', 'iOS', 'Android', 'Flutter'],
    availability: 100,
    currentProjects: [2],
    workload: 70,
    performance: 87
  },
  {
    id: 6,
    name: 'Lisa Wang',
    role: 'UI/UX Designer',
    email: 'lisa.wang@nexusarchitect.com',
    avatar: '/avatars/lisa-wang.jpg',
    skills: ['Figma', 'Adobe Creative Suite', 'User Research', 'Prototyping'],
    availability: 90,
    currentProjects: [2],
    workload: 75,
    performance: 93
  }
]

export const timelineData = [
  { date: '2024-01-01', project: 'Nexus Core', milestone: 'Project Kickoff', status: 'completed' },
  { date: '2024-01-15', project: 'Nexus Core', milestone: 'Foundation Setup', status: 'completed' },
  { date: '2024-02-01', project: 'Mobile App', milestone: 'Project Start', status: 'completed' },
  { date: '2024-02-15', project: 'Nexus Core', milestone: 'AI Intelligence Module', status: 'completed' },
  { date: '2024-02-15', project: 'Mobile App', milestone: 'Requirements Analysis', status: 'completed' },
  { date: '2024-02-28', project: 'Nexus Core', milestone: 'Data Ingestion System', status: 'in-progress' },
  { date: '2024-03-01', project: 'Mobile App', milestone: 'UI/UX Design', status: 'in-progress' },
  { date: '2024-03-15', project: 'Nexus Core', milestone: 'Autonomous Capabilities', status: 'pending' },
  { date: '2024-03-31', project: 'Nexus Core', milestone: 'User Interfaces', status: 'pending' },
  { date: '2024-04-15', project: 'Mobile App', milestone: 'iOS Development', status: 'pending' },
  { date: '2024-04-30', project: 'Mobile App', milestone: 'Android Development', status: 'pending' },
  { date: '2024-05-31', project: 'Mobile App', milestone: 'Testing & Launch', status: 'pending' }
]

export const budgetData = [
  { month: 'Jan', planned: 800000, actual: 750000, forecast: 780000 },
  { month: 'Feb', planned: 900000, actual: 920000, forecast: 950000 },
  { month: 'Mar', planned: 850000, actual: 0, forecast: 880000 },
  { month: 'Apr', planned: 750000, actual: 0, forecast: 720000 },
  { month: 'May', planned: 600000, actual: 0, forecast: 580000 },
  { month: 'Jun', planned: 500000, actual: 0, forecast: 520000 }
]

export const performanceMetrics = [
  { metric: 'On-Time Delivery', value: 87, target: 90, trend: 'up' },
  { metric: 'Budget Adherence', value: 92, target: 95, trend: 'up' },
  { metric: 'Quality Score', value: 94, target: 90, trend: 'up' },
  { metric: 'Team Satisfaction', value: 88, target: 85, trend: 'stable' },
  { metric: 'Risk Mitigation', value: 85, target: 80, trend: 'up' },
  { metric: 'Stakeholder Satisfaction', value: 91, target: 90, trend: 'up' }
]

export const resourceUtilization = [
  { name: 'Sarah Chen', utilization: 85, capacity: 100, efficiency: 92 },
  { name: 'John Doe', utilization: 78, capacity: 90, efficiency: 88 },
  { name: 'Alice Johnson', utilization: 82, capacity: 100, efficiency: 90 },
  { name: 'Bob Wilson', utilization: 65, capacity: 80, efficiency: 85 },
  { name: 'David Kim', utilization: 70, capacity: 100, efficiency: 87 },
  { name: 'Lisa Wang', utilization: 75, capacity: 90, efficiency: 93 }
]

export const riskAssessment = [
  {
    id: 1,
    title: 'Integration Complexity',
    description: 'Complex integration with legacy systems may cause delays',
    probability: 'High',
    impact: 'Medium',
    severity: 'Medium',
    mitigation: 'Dedicated integration team and extended testing phase',
    owner: 'John Doe',
    status: 'Active'
  },
  {
    id: 2,
    title: 'Resource Availability',
    description: 'Key team members may not be available during critical phases',
    probability: 'Medium',
    impact: 'High',
    severity: 'Medium',
    mitigation: 'Cross-training and backup resource identification',
    owner: 'Sarah Chen',
    status: 'Mitigated'
  },
  {
    id: 3,
    title: 'Technology Changes',
    description: 'Rapid changes in AI technology may require architecture updates',
    probability: 'Low',
    impact: 'High',
    severity: 'Low',
    mitigation: 'Modular architecture design and regular technology reviews',
    owner: 'Alice Johnson',
    status: 'Monitoring'
  }
]

export const communicationData = [
  {
    id: 1,
    type: 'meeting',
    title: 'Weekly Project Standup',
    participants: ['Sarah Chen', 'John Doe', 'Alice Johnson', 'Bob Wilson'],
    date: '2024-01-08',
    time: '09:00',
    duration: 30,
    status: 'completed'
  },
  {
    id: 2,
    type: 'message',
    title: 'Database optimization discussion',
    participants: ['Bob Wilson', 'John Doe'],
    date: '2024-01-08',
    time: '14:30',
    status: 'active'
  },
  {
    id: 3,
    type: 'document',
    title: 'Project Requirements Document v2.1',
    author: 'Sarah Chen',
    date: '2024-01-07',
    collaborators: ['John Doe', 'Alice Johnson', 'David Kim'],
    status: 'review'
  },
  {
    id: 4,
    type: 'meeting',
    title: 'Mobile App Design Review',
    participants: ['David Kim', 'Lisa Wang', 'Sarah Chen'],
    date: '2024-01-09',
    time: '15:00',
    duration: 60,
    status: 'scheduled'
  }
]

export const workflowSteps = [
  {
    id: 1,
    name: 'Task Creation',
    description: 'Create and assign new tasks to team members',
    status: 'active',
    assignee: 'Project Manager',
    approver: null,
    duration: 15
  },
  {
    id: 2,
    name: 'Development',
    description: 'Implementation and coding phase',
    status: 'active',
    assignee: 'Developer',
    approver: null,
    duration: 480
  },
  {
    id: 3,
    name: 'Code Review',
    description: 'Peer review of code changes',
    status: 'pending',
    assignee: 'Senior Developer',
    approver: 'Tech Lead',
    duration: 60
  },
  {
    id: 4,
    name: 'Testing',
    description: 'Quality assurance and testing',
    status: 'pending',
    assignee: 'QA Engineer',
    approver: null,
    duration: 120
  },
  {
    id: 5,
    name: 'Deployment',
    description: 'Deploy to production environment',
    status: 'pending',
    assignee: 'DevOps Engineer',
    approver: 'Project Manager',
    duration: 30
  }
]

