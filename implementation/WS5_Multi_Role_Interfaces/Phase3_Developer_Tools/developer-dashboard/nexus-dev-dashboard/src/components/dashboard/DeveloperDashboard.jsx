import { useState, useEffect } from 'react'
import { 
  Code2, 
  GitCommit, 
  GitPullRequest, 
  MessageSquare, 
  TestTube, 
  Bug,
  TrendingUp,
  TrendingDown,
  Clock,
  CheckCircle,
  AlertTriangle,
  XCircle,
  Activity,
  BarChart3,
  Zap
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Progress } from '../ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

import { 
  developerMetrics, 
  recentActivity, 
  codeQualityTrends, 
  productivityTrends,
  repositories,
  aiSuggestions
} from '../../data/mockData'

const DeveloperDashboard = () => {
  const [timeRange, setTimeRange] = useState('7d')
  const [selectedMetric, setSelectedMetric] = useState('productivity')

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />
      case 'approved':
        return <CheckCircle className="h-4 w-4 text-blue-500" />
      default:
        return <Activity className="h-4 w-4 text-gray-500" />
    }
  }

  const getActivityTypeIcon = (type) => {
    switch (type) {
      case 'commit':
        return <GitCommit className="h-4 w-4" />
      case 'pull_request':
        return <GitPullRequest className="h-4 w-4" />
      case 'deployment':
        return <Zap className="h-4 w-4" />
      case 'code_review':
        return <MessageSquare className="h-4 w-4" />
      case 'test':
        return <TestTube className="h-4 w-4" />
      default:
        return <Activity className="h-4 w-4" />
    }
  }

  const getRepositoryStatus = (status) => {
    switch (status) {
      case 'healthy':
        return <Badge variant="default" className="bg-green-500">Healthy</Badge>
      case 'warning':
        return <Badge variant="default" className="bg-yellow-500">Warning</Badge>
      case 'critical':
        return <Badge variant="destructive">Critical</Badge>
      default:
        return <Badge variant="secondary">Unknown</Badge>
    }
  }

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high':
        return 'text-red-500'
      case 'medium':
        return 'text-yellow-500'
      case 'low':
        return 'text-green-500'
      default:
        return 'text-gray-500'
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Developer Dashboard</h1>
          <p className="text-muted-foreground">
            Track your productivity, code quality, and development insights
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <BarChart3 className="h-4 w-4 mr-2" />
            Export Report
          </Button>
          <Button size="sm">
            <Activity className="h-4 w-4 mr-2" />
            View Analytics
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Lines of Code</CardTitle>
            <Code2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{developerMetrics.productivity.linesOfCode.toLocaleString()}</div>
            <div className="flex items-center text-xs text-muted-foreground">
              {developerMetrics.productivity.linesOfCodeChange > 0 ? (
                <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
              ) : (
                <TrendingDown className="h-3 w-3 text-red-500 mr-1" />
              )}
              {Math.abs(developerMetrics.productivity.linesOfCodeChange)}% from last week
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Code Quality</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{developerMetrics.codeQuality.overallScore}/100</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
              +{developerMetrics.codeQuality.scoreChange} from last week
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pull Requests</CardTitle>
            <GitPullRequest className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{developerMetrics.productivity.pullRequests}</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingDown className="h-3 w-3 text-red-500 mr-1" />
              {Math.abs(developerMetrics.productivity.pullRequestsChange)}% from last week
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Test Coverage</CardTitle>
            <TestTube className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{developerMetrics.codeQuality.coverage}%</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
              +{developerMetrics.codeQuality.coverageChange}% from last week
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="activity">Recent Activity</TabsTrigger>
          <TabsTrigger value="repositories">Repositories</TabsTrigger>
          <TabsTrigger value="ai-insights">AI Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Productivity Trends */}
            <Card>
              <CardHeader>
                <CardTitle>Productivity Trends</CardTitle>
                <CardDescription>Your development activity over the past week</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={productivityTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <YAxis />
                    <Tooltip labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <Line type="monotone" dataKey="commits" stroke="#8884d8" strokeWidth={2} />
                    <Line type="monotone" dataKey="pullRequests" stroke="#82ca9d" strokeWidth={2} />
                    <Line type="monotone" dataKey="codeReviews" stroke="#ffc658" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Code Quality Trends */}
            <Card>
              <CardHeader>
                <CardTitle>Code Quality Trends</CardTitle>
                <CardDescription>Quality metrics improvement over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={codeQualityTrends}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <YAxis />
                    <Tooltip labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <Line type="monotone" dataKey="coverage" stroke="#8884d8" strokeWidth={2} />
                    <Line type="monotone" dataKey="score" stroke="#82ca9d" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Performance Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>Build times, test execution, and deployment performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Build Time</span>
                    <span className="text-sm text-muted-foreground">{developerMetrics.performance.buildTime}</span>
                  </div>
                  <Progress value={75} className="h-2" />
                  <div className="text-xs text-green-500">
                    {Math.abs(developerMetrics.performance.buildTimeChange)}% faster
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Test Time</span>
                    <span className="text-sm text-muted-foreground">{developerMetrics.performance.testTime}</span>
                  </div>
                  <Progress value={85} className="h-2" />
                  <div className="text-xs text-green-500">
                    {Math.abs(developerMetrics.performance.testTimeChange)}% faster
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Deploy Time</span>
                    <span className="text-sm text-muted-foreground">{developerMetrics.performance.deployTime}</span>
                  </div>
                  <Progress value={90} className="h-2" />
                  <div className="text-xs text-green-500">
                    {Math.abs(developerMetrics.performance.deployTimeChange)}% faster
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Error Rate</span>
                    <span className="text-sm text-muted-foreground">{developerMetrics.performance.errorRate}%</span>
                  </div>
                  <Progress value={95} className="h-2" />
                  <div className="text-xs text-green-500">
                    {Math.abs(developerMetrics.performance.errorRateChange)}% reduction
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="activity" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>Your latest development activities and their status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentActivity.map((activity) => (
                  <div key={activity.id} className="flex items-start space-x-4 p-4 border rounded-lg">
                    <div className="flex items-center space-x-2">
                      {getActivityTypeIcon(activity.type)}
                      {getStatusIcon(activity.status)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {activity.message}
                      </p>
                      <div className="flex items-center space-x-4 mt-1">
                        <span className="text-xs text-muted-foreground">{activity.repository}</span>
                        <span className="text-xs text-muted-foreground">{activity.branch}</span>
                        <span className="text-xs text-muted-foreground">{activity.timestamp}</span>
                      </div>
                      {activity.reviewers && (
                        <div className="flex items-center space-x-1 mt-2">
                          <span className="text-xs text-muted-foreground">Reviewers:</span>
                          {activity.reviewers.map((reviewer, index) => (
                            <Badge key={index} variant="secondary" className="text-xs">
                              {reviewer}
                            </Badge>
                          ))}
                        </div>
                      )}
                      {activity.details && (
                        <p className="text-xs text-muted-foreground mt-1">{activity.details}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="repositories" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {repositories.map((repo, index) => (
              <Card key={index}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{repo.name}</CardTitle>
                    {getRepositoryStatus(repo.status)}
                  </div>
                  <CardDescription>{repo.language}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Test Coverage</span>
                      <span className="text-sm font-medium">{repo.coverage}%</span>
                    </div>
                    <Progress value={repo.coverage} className="h-2" />
                    
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <div className="text-lg font-bold">{repo.issues}</div>
                        <div className="text-xs text-muted-foreground">Issues</div>
                      </div>
                      <div>
                        <div className="text-lg font-bold">{repo.pullRequests}</div>
                        <div className="text-xs text-muted-foreground">PRs</div>
                      </div>
                      <div>
                        <div className="text-lg font-bold">{repo.contributors}</div>
                        <div className="text-xs text-muted-foreground">Contributors</div>
                      </div>
                    </div>
                    
                    <div className="text-xs text-muted-foreground">
                      Last commit: {repo.lastCommit}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="ai-insights" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>AI-Powered Insights</CardTitle>
              <CardDescription>Intelligent suggestions to improve your code and workflow</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {aiSuggestions.map((suggestion) => (
                  <div key={suggestion.id} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <Badge variant="outline" className={getSeverityColor(suggestion.severity)}>
                            {suggestion.severity}
                          </Badge>
                          <Badge variant="secondary">{suggestion.type.replace('_', ' ')}</Badge>
                          <span className="text-xs text-muted-foreground">
                            {suggestion.confidence}% confidence
                          </span>
                        </div>
                        <h4 className="font-medium text-foreground">{suggestion.title}</h4>
                        <p className="text-sm text-muted-foreground mt-1">{suggestion.description}</p>
                        <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                          <span>{suggestion.file}:{suggestion.line}</span>
                          <span>Impact: {suggestion.estimatedImpact}</span>
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        <Button size="sm" variant="outline">
                          View Code
                        </Button>
                        <Button size="sm">
                          Apply Fix
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default DeveloperDashboard

