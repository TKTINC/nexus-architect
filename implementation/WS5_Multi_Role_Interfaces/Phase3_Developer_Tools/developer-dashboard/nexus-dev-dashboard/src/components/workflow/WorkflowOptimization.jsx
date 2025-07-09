import { useState } from 'react'
import { 
  Workflow, 
  Clock, 
  TrendingUp, 
  CheckCircle, 
  AlertCircle,
  Play,
  Pause,
  Settings,
  Zap,
  Target,
  BarChart3,
  Timer,
  Cpu,
  Database,
  GitBranch,
  TestTube,
  Rocket,
  Bot
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Progress } from '../ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { Switch } from '../ui/switch'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

import { workflowOptimizations, developerMetrics } from '../../data/mockData'

const WorkflowOptimization = () => {
  const [automationEnabled, setAutomationEnabled] = useState({
    codeFormatting: true,
    testExecution: false,
    dependencyUpdates: true,
    buildOptimization: false,
    deploymentPipeline: true
  })

  const workflowMetrics = [
    {
      name: 'Time Saved',
      value: '2.5 hours',
      change: 15.3,
      icon: Clock,
      color: 'text-green-500'
    },
    {
      name: 'Automation Rate',
      value: '78%',
      change: 12.1,
      icon: Bot,
      color: 'text-blue-500'
    },
    {
      name: 'Process Efficiency',
      value: '92%',
      change: 8.7,
      icon: Target,
      color: 'text-purple-500'
    },
    {
      name: 'Error Reduction',
      value: '45%',
      change: 23.4,
      icon: CheckCircle,
      color: 'text-orange-500'
    }
  ]

  const automationTasks = [
    {
      name: 'Code Formatting',
      description: 'Automatically format code on commit using Prettier and ESLint',
      enabled: automationEnabled.codeFormatting,
      timeSaved: '15 min/day',
      impact: 'High',
      category: 'Code Quality',
      icon: Settings
    },
    {
      name: 'Test Execution',
      description: 'Run tests in parallel and cache results for faster feedback',
      enabled: automationEnabled.testExecution,
      timeSaved: '5 min/build',
      impact: 'High',
      category: 'Testing',
      icon: TestTube
    },
    {
      name: 'Dependency Updates',
      description: 'Automatically create PRs for dependency updates with security checks',
      enabled: automationEnabled.dependencyUpdates,
      timeSaved: '2 hours/week',
      impact: 'Medium',
      category: 'Maintenance',
      icon: Database
    },
    {
      name: 'Build Optimization',
      description: 'Optimize Docker builds with intelligent layer caching',
      enabled: automationEnabled.buildOptimization,
      timeSaved: '3 min/build',
      impact: 'Medium',
      category: 'DevOps',
      icon: Cpu
    },
    {
      name: 'Deployment Pipeline',
      description: 'Automated deployment with rollback capabilities and health checks',
      enabled: automationEnabled.deploymentPipeline,
      timeSaved: '10 min/deploy',
      impact: 'High',
      category: 'Deployment',
      icon: Rocket
    }
  ]

  const processMetrics = [
    { date: '2024-01-01', efficiency: 78, automation: 65, timeSaved: 1.2 },
    { date: '2024-01-02', efficiency: 82, automation: 68, timeSaved: 1.5 },
    { date: '2024-01-03', efficiency: 85, automation: 72, timeSaved: 1.8 },
    { date: '2024-01-04', efficiency: 88, automation: 75, timeSaved: 2.1 },
    { date: '2024-01-05', efficiency: 90, automation: 76, timeSaved: 2.3 },
    { date: '2024-01-06', efficiency: 91, automation: 77, timeSaved: 2.4 },
    { date: '2024-01-07', efficiency: 92, automation: 78, timeSaved: 2.5 }
  ]

  const bottlenecks = [
    {
      id: 1,
      name: 'Code Review Process',
      impact: 'High',
      frequency: 'Daily',
      avgTime: '45 minutes',
      suggestion: 'Implement automated code review checks and reviewer assignment',
      category: 'Review Process'
    },
    {
      id: 2,
      name: 'Manual Testing',
      impact: 'Medium',
      frequency: 'Per Release',
      avgTime: '2 hours',
      suggestion: 'Expand automated test coverage and implement visual regression testing',
      category: 'Testing'
    },
    {
      id: 3,
      name: 'Environment Setup',
      impact: 'Medium',
      frequency: 'Weekly',
      avgTime: '30 minutes',
      suggestion: 'Create standardized development containers and setup scripts',
      category: 'Development'
    },
    {
      id: 4,
      name: 'Deployment Coordination',
      impact: 'Low',
      frequency: 'Per Release',
      avgTime: '15 minutes',
      suggestion: 'Implement automated deployment scheduling and notifications',
      category: 'Deployment'
    }
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500'
      case 'in_progress':
        return 'bg-blue-500'
      case 'recommended':
        return 'bg-yellow-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getImpactColor = (impact) => {
    switch (impact) {
      case 'High':
        return 'text-red-500'
      case 'Medium':
        return 'text-yellow-500'
      case 'Low':
        return 'text-green-500'
      default:
        return 'text-gray-500'
    }
  }

  const toggleAutomation = (task) => {
    setAutomationEnabled(prev => ({
      ...prev,
      [task]: !prev[task]
    }))
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Workflow Optimization</h1>
          <p className="text-muted-foreground">
            Automate processes, identify bottlenecks, and optimize your development workflow
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <BarChart3 className="h-4 w-4 mr-2" />
            View Analytics
          </Button>
          <Button size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Configure
          </Button>
        </div>
      </div>

      {/* Workflow Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {workflowMetrics.map((metric) => {
          const Icon = metric.icon
          return (
            <Card key={metric.name}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{metric.name}</CardTitle>
                <Icon className={`h-4 w-4 ${metric.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metric.value}</div>
                <div className="flex items-center text-xs text-muted-foreground">
                  <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
                  +{metric.change}% from last week
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="automation" className="space-y-6">
        <TabsList>
          <TabsTrigger value="automation">Automation</TabsTrigger>
          <TabsTrigger value="optimizations">Optimizations</TabsTrigger>
          <TabsTrigger value="bottlenecks">Bottlenecks</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="automation" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Automation Controls</CardTitle>
              <CardDescription>Enable or disable automated workflow processes</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {automationTasks.map((task) => {
                  const Icon = task.icon
                  return (
                    <div key={task.name} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <Icon className="h-5 w-5 text-muted-foreground" />
                        <div className="flex-1">
                          <div className="flex items-center space-x-2">
                            <h4 className="font-medium">{task.name}</h4>
                            <Badge variant="outline" className={getImpactColor(task.impact)}>
                              {task.impact} Impact
                            </Badge>
                            <Badge variant="secondary">{task.category}</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">{task.description}</p>
                          <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                            <span>Saves: {task.timeSaved}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Switch
                          checked={task.enabled}
                          onCheckedChange={() => toggleAutomation(task.name.toLowerCase().replace(/\s+/g, ''))}
                        />
                        <Button size="sm" variant="outline">
                          Configure
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="optimizations" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Optimization Recommendations</CardTitle>
              <CardDescription>Suggested improvements to enhance your workflow efficiency</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {workflowOptimizations.map((optimization) => (
                  <div key={optimization.id} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <div className={`w-3 h-3 rounded-full ${getStatusColor(optimization.status)}`}></div>
                          <Badge variant="outline" className={getImpactColor(optimization.impact)}>
                            {optimization.impact} Impact
                          </Badge>
                          <Badge variant="secondary">{optimization.effort} Effort</Badge>
                          <Badge variant="outline">{optimization.category}</Badge>
                        </div>
                        <h4 className="font-medium text-foreground">{optimization.title}</h4>
                        <p className="text-sm text-muted-foreground mt-1">{optimization.description}</p>
                        <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                          <span>Time Saved: {optimization.timeSaved}</span>
                          <span>Status: {optimization.status.replace('_', ' ')}</span>
                        </div>
                        {optimization.implementation && (
                          <div className="mt-2 p-2 bg-muted rounded text-xs">
                            <strong>Implementation:</strong> {optimization.implementation}
                          </div>
                        )}
                      </div>
                      <div className="flex space-x-2">
                        {optimization.status === 'recommended' && (
                          <>
                            <Button size="sm" variant="outline">
                              Learn More
                            </Button>
                            <Button size="sm">
                              Implement
                            </Button>
                          </>
                        )}
                        {optimization.status === 'in_progress' && (
                          <Button size="sm" variant="outline">
                            View Progress
                          </Button>
                        )}
                        {optimization.status === 'completed' && (
                          <Button size="sm" variant="outline" disabled>
                            <CheckCircle className="h-4 w-4 mr-1" />
                            Completed
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="bottlenecks" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Process Bottlenecks</CardTitle>
              <CardDescription>Identified bottlenecks in your development process</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {bottlenecks.map((bottleneck) => (
                  <div key={bottleneck.id} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <Badge variant="outline" className={getImpactColor(bottleneck.impact)}>
                            {bottleneck.impact} Impact
                          </Badge>
                          <Badge variant="secondary">{bottleneck.category}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {bottleneck.frequency} • {bottleneck.avgTime}
                          </span>
                        </div>
                        <h4 className="font-medium text-foreground">{bottleneck.name}</h4>
                        <p className="text-sm text-muted-foreground mt-1">
                          <strong>Suggestion:</strong> {bottleneck.suggestion}
                        </p>
                      </div>
                      <div className="flex space-x-2">
                        <Button size="sm" variant="outline">
                          Analyze
                        </Button>
                        <Button size="sm">
                          Optimize
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Process Efficiency Trends */}
            <Card>
              <CardHeader>
                <CardTitle>Process Efficiency</CardTitle>
                <CardDescription>Workflow efficiency and automation trends</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={processMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <YAxis />
                    <Tooltip labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <Line type="monotone" dataKey="efficiency" stroke="#8b5cf6" strokeWidth={2} name="Efficiency %" />
                    <Line type="monotone" dataKey="automation" stroke="#22c55e" strokeWidth={2} name="Automation %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Time Savings */}
            <Card>
              <CardHeader>
                <CardTitle>Time Savings</CardTitle>
                <CardDescription>Daily time saved through automation</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={processMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <YAxis />
                    <Tooltip labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <Bar dataKey="timeSaved" fill="#f59e0b" name="Hours Saved" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Performance Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Summary</CardTitle>
              <CardDescription>Overall workflow performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-500">17.5</div>
                  <div className="text-sm text-muted-foreground">Hours Saved This Week</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Through automation and optimization
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-500">78%</div>
                  <div className="text-sm text-muted-foreground">Automation Coverage</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Of repetitive tasks automated
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-500">92%</div>
                  <div className="text-sm text-muted-foreground">Process Efficiency</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Overall workflow efficiency score
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default WorkflowOptimization

