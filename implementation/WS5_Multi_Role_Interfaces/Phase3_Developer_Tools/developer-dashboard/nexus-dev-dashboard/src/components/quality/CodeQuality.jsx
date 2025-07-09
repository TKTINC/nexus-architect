import { useState } from 'react'
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  Shield,
  Clock,
  Code,
  FileText,
  Bug,
  Zap,
  Target,
  Activity
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Progress } from '../ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'

import { 
  developerMetrics, 
  codeQualityTrends, 
  repositories,
  technicalDebt
} from '../../data/mockData'

const CodeQuality = () => {
  const [selectedRepository, setSelectedRepository] = useState('all')
  const [timeRange, setTimeRange] = useState('7d')

  const qualityMetrics = [
    {
      name: 'Test Coverage',
      value: developerMetrics.codeQuality.coverage,
      change: developerMetrics.codeQuality.coverageChange,
      target: 95,
      icon: Target,
      color: 'text-green-500'
    },
    {
      name: 'Code Complexity',
      value: developerMetrics.codeQuality.complexity,
      change: developerMetrics.codeQuality.complexityChange,
      target: 2.0,
      icon: Activity,
      color: 'text-blue-500',
      inverse: true
    },
    {
      name: 'Code Duplication',
      value: developerMetrics.codeQuality.duplication,
      change: developerMetrics.codeQuality.duplicationChange,
      target: 1.0,
      icon: FileText,
      color: 'text-yellow-500',
      inverse: true
    },
    {
      name: 'Overall Score',
      value: developerMetrics.codeQuality.overallScore,
      change: developerMetrics.codeQuality.scoreChange,
      target: 90,
      icon: BarChart3,
      color: 'text-purple-500'
    }
  ]

  const securityMetrics = [
    { name: 'Critical', value: 0, color: '#ef4444' },
    { name: 'High', value: 2, color: '#f97316' },
    { name: 'Medium', value: 5, color: '#eab308' },
    { name: 'Low', value: 12, color: '#22c55e' },
    { name: 'Info', value: 8, color: '#6b7280' }
  ]

  const performanceMetrics = [
    { name: 'Build Time', value: '2m 34s', trend: -8.2, icon: Clock },
    { name: 'Test Execution', value: '45s', trend: -12.1, icon: CheckCircle },
    { name: 'Code Analysis', value: '1m 12s', trend: -5.3, icon: Code },
    { name: 'Deployment', value: '3m 12s', trend: -15.3, icon: Zap }
  ]

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high':
        return 'bg-red-500'
      case 'medium':
        return 'bg-yellow-500'
      case 'low':
        return 'bg-green-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getGradeColor = (grade) => {
    if (grade.startsWith('A')) return 'text-green-500'
    if (grade.startsWith('B')) return 'text-yellow-500'
    if (grade.startsWith('C')) return 'text-orange-500'
    return 'text-red-500'
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Code Quality</h1>
          <p className="text-muted-foreground">
            Monitor code quality metrics, technical debt, and security vulnerabilities
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Select value={selectedRepository} onValueChange={setSelectedRepository}>
            <SelectTrigger className="w-48">
              <SelectValue placeholder="Select repository" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Repositories</SelectItem>
              {repositories.map((repo) => (
                <SelectItem key={repo.name} value={repo.name}>
                  {repo.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1d">1 Day</SelectItem>
              <SelectItem value="7d">7 Days</SelectItem>
              <SelectItem value="30d">30 Days</SelectItem>
              <SelectItem value="90d">90 Days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Quality Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {qualityMetrics.map((metric) => {
          const Icon = metric.icon
          const isOnTarget = metric.inverse 
            ? metric.value <= metric.target 
            : metric.value >= metric.target
          
          return (
            <Card key={metric.name}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{metric.name}</CardTitle>
                <Icon className={`h-4 w-4 ${metric.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {typeof metric.value === 'number' && metric.value < 10 
                    ? metric.value.toFixed(1) 
                    : metric.value}
                  {metric.name.includes('Coverage') || metric.name.includes('Score') ? '%' : ''}
                </div>
                <div className="flex items-center justify-between mt-2">
                  <div className="flex items-center text-xs text-muted-foreground">
                    {metric.change > 0 ? (
                      <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
                    ) : (
                      <TrendingDown className="h-3 w-3 text-red-500 mr-1" />
                    )}
                    {Math.abs(metric.change)}% from last week
                  </div>
                  {isOnTarget ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  )}
                </div>
                <div className="mt-2">
                  <Progress 
                    value={metric.inverse 
                      ? Math.max(0, 100 - (metric.value / metric.target * 100))
                      : (metric.value / metric.target * 100)
                    } 
                    className="h-2" 
                  />
                  <div className="text-xs text-muted-foreground mt-1">
                    Target: {metric.target}{metric.name.includes('Coverage') || metric.name.includes('Score') ? '%' : ''}
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="technical-debt">Technical Debt</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Quality Trends */}
            <Card>
              <CardHeader>
                <CardTitle>Quality Trends</CardTitle>
                <CardDescription>Code quality metrics over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={codeQualityTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <YAxis />
                    <Tooltip labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <Line type="monotone" dataKey="coverage" stroke="#22c55e" strokeWidth={2} name="Coverage %" />
                    <Line type="monotone" dataKey="score" stroke="#8b5cf6" strokeWidth={2} name="Quality Score" />
                    <Line type="monotone" dataKey="complexity" stroke="#f59e0b" strokeWidth={2} name="Complexity" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Repository Quality Comparison */}
            <Card>
              <CardHeader>
                <CardTitle>Repository Quality</CardTitle>
                <CardDescription>Quality comparison across repositories</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={repositories}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="coverage" fill="#22c55e" name="Coverage %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Quality Grades */}
          <Card>
            <CardHeader>
              <CardTitle>Quality Grades</CardTitle>
              <CardDescription>Overall quality assessment by category</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className={`text-4xl font-bold ${getGradeColor(developerMetrics.codeQuality.maintainability)}`}>
                    {developerMetrics.codeQuality.maintainability}
                  </div>
                  <div className="text-sm text-muted-foreground">Maintainability</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Code structure and readability
                  </div>
                </div>
                <div className="text-center">
                  <div className={`text-4xl font-bold ${getGradeColor(developerMetrics.codeQuality.reliability)}`}>
                    {developerMetrics.codeQuality.reliability}
                  </div>
                  <div className="text-sm text-muted-foreground">Reliability</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Bug-free and stable code
                  </div>
                </div>
                <div className="text-center">
                  <div className={`text-4xl font-bold ${getGradeColor(developerMetrics.codeQuality.security)}`}>
                    {developerMetrics.codeQuality.security}
                  </div>
                  <div className="text-sm text-muted-foreground">Security</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Security vulnerabilities and risks
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="technical-debt" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Technical Debt</CardTitle>
              <CardDescription>Issues that need attention to maintain code quality</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {technicalDebt.map((debt) => (
                  <div key={debt.id} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <div className={`w-3 h-3 rounded-full ${getSeverityColor(debt.severity)}`}></div>
                          <Badge variant="outline" className="capitalize">
                            {debt.severity} Priority
                          </Badge>
                          <Badge variant="secondary">{debt.effort}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {debt.repository}
                          </span>
                        </div>
                        <h4 className="font-medium text-foreground">{debt.title}</h4>
                        <p className="text-sm text-muted-foreground mt-1">
                          Impact: {debt.impact}
                        </p>
                        <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                          <span>{debt.file}</span>
                          {debt.lines > 0 && <span>{debt.lines} lines</span>}
                          <span>Created: {debt.created}</span>
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        <Button size="sm" variant="outline">
                          View Code
                        </Button>
                        <Button size="sm">
                          Create Task
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Security Overview */}
            <Card>
              <CardHeader>
                <CardTitle>Security Vulnerabilities</CardTitle>
                <CardDescription>Distribution of security issues by severity</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={securityMetrics}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {securityMetrics.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Security Actions */}
            <Card>
              <CardHeader>
                <CardTitle>Security Actions</CardTitle>
                <CardDescription>Recommended security improvements</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center space-x-3 p-3 border rounded-lg">
                    <Shield className="h-5 w-5 text-red-500" />
                    <div className="flex-1">
                      <div className="font-medium">Update vulnerable dependencies</div>
                      <div className="text-sm text-muted-foreground">2 high-severity vulnerabilities found</div>
                    </div>
                    <Button size="sm">Fix</Button>
                  </div>
                  <div className="flex items-center space-x-3 p-3 border rounded-lg">
                    <Shield className="h-5 w-5 text-yellow-500" />
                    <div className="flex-1">
                      <div className="font-medium">Enable security scanning</div>
                      <div className="text-sm text-muted-foreground">Add automated security checks to CI/CD</div>
                    </div>
                    <Button size="sm" variant="outline">Configure</Button>
                  </div>
                  <div className="flex items-center space-x-3 p-3 border rounded-lg">
                    <Shield className="h-5 w-5 text-green-500" />
                    <div className="flex-1">
                      <div className="font-medium">Review access permissions</div>
                      <div className="text-sm text-muted-foreground">Audit repository access and permissions</div>
                    </div>
                    <Button size="sm" variant="outline">Review</Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {performanceMetrics.map((metric) => {
              const Icon = metric.icon
              return (
                <Card key={metric.name}>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">{metric.name}</CardTitle>
                    <Icon className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{metric.value}</div>
                    <div className="flex items-center text-xs text-muted-foreground">
                      <TrendingDown className="h-3 w-3 text-green-500 mr-1" />
                      {Math.abs(metric.trend)}% faster than last week
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Performance Optimization Recommendations</CardTitle>
              <CardDescription>Suggestions to improve build and deployment performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center space-x-3 p-3 border rounded-lg">
                  <Zap className="h-5 w-5 text-blue-500" />
                  <div className="flex-1">
                    <div className="font-medium">Enable build caching</div>
                    <div className="text-sm text-muted-foreground">
                      Reduce build time by 40% with intelligent caching
                    </div>
                  </div>
                  <Button size="sm">Enable</Button>
                </div>
                <div className="flex items-center space-x-3 p-3 border rounded-lg">
                  <Zap className="h-5 w-5 text-green-500" />
                  <div className="flex-1">
                    <div className="font-medium">Optimize test execution</div>
                    <div className="text-sm text-muted-foreground">
                      Run tests in parallel to reduce execution time
                    </div>
                  </div>
                  <Button size="sm" variant="outline">Configure</Button>
                </div>
                <div className="flex items-center space-x-3 p-3 border rounded-lg">
                  <Zap className="h-5 w-5 text-purple-500" />
                  <div className="flex-1">
                    <div className="font-medium">Upgrade CI/CD infrastructure</div>
                    <div className="text-sm text-muted-foreground">
                      Use faster runners for improved performance
                    </div>
                  </div>
                  <Button size="sm" variant="outline">Review</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default CodeQuality

