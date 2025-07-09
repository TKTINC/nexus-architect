import { useState, useEffect } from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  Clock, 
  CheckCircle, 
  AlertTriangle,
  Users,
  Calendar,
  BarChart3,
  Activity,
  Zap,
  Target,
  ArrowRight,
  RefreshCw,
  Wifi,
  WifiOff
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Progress } from '../ui/progress'
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar'
import { Separator } from '../ui/separator'
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts'

const HomeScreen = ({ currentUser, isOnline, deviceInfo }) => {
  const [refreshing, setRefreshing] = useState(false)
  const [lastRefresh, setLastRefresh] = useState(new Date())

  const handleRefresh = async () => {
    setRefreshing(true)
    // Simulate refresh delay
    await new Promise(resolve => setTimeout(resolve, 1500))
    setLastRefresh(new Date())
    setRefreshing(false)
  }

  // Mock data for dashboard
  const dashboardData = {
    kpis: [
      {
        title: 'Active Projects',
        value: '12',
        change: '+2',
        trend: 'up',
        icon: Target,
        color: 'text-blue-500'
      },
      {
        title: 'Tasks Due Today',
        value: '8',
        change: '-3',
        trend: 'down',
        icon: Clock,
        color: 'text-orange-500'
      },
      {
        title: 'Team Members',
        value: '24',
        change: '+1',
        trend: 'up',
        icon: Users,
        color: 'text-green-500'
      },
      {
        title: 'Completion Rate',
        value: '87%',
        change: '+5%',
        trend: 'up',
        icon: CheckCircle,
        color: 'text-purple-500'
      }
    ],
    recentActivity: [
      {
        id: 1,
        type: 'task_completed',
        title: 'UI Design Review completed',
        user: 'Alice Johnson',
        time: '2 minutes ago',
        avatar: '/api/placeholder/32/32'
      },
      {
        id: 2,
        type: 'project_updated',
        title: 'Project Alpha milestone reached',
        user: 'Bob Wilson',
        time: '15 minutes ago',
        avatar: '/api/placeholder/32/32'
      },
      {
        id: 3,
        type: 'comment_added',
        title: 'New comment on Backend API task',
        user: 'Carol Davis',
        time: '1 hour ago',
        avatar: '/api/placeholder/32/32'
      },
      {
        id: 4,
        type: 'meeting_scheduled',
        title: 'Sprint Planning meeting scheduled',
        user: 'David Chen',
        time: '2 hours ago',
        avatar: '/api/placeholder/32/32'
      }
    ],
    upcomingTasks: [
      {
        id: 1,
        title: 'Review pull request #234',
        project: 'Project Alpha',
        priority: 'high',
        dueTime: '2:00 PM',
        assignee: 'You'
      },
      {
        id: 2,
        title: 'Update project documentation',
        project: 'Project Beta',
        priority: 'medium',
        dueTime: '4:30 PM',
        assignee: 'Alice Johnson'
      },
      {
        id: 3,
        title: 'Client presentation prep',
        project: 'Project Gamma',
        priority: 'high',
        dueTime: 'Tomorrow 9:00 AM',
        assignee: 'Team'
      }
    ],
    chartData: [
      { name: 'Mon', tasks: 12, completed: 10 },
      { name: 'Tue', tasks: 15, completed: 13 },
      { name: 'Wed', tasks: 18, completed: 16 },
      { name: 'Thu', tasks: 14, completed: 12 },
      { name: 'Fri', tasks: 16, completed: 15 },
      { name: 'Sat', tasks: 8, completed: 8 },
      { name: 'Sun', tasks: 5, completed: 5 }
    ]
  }

  const getPriorityColor = (priority) => {
    switch (priority) {
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

  const getActivityIcon = (type) => {
    switch (type) {
      case 'task_completed':
        return '✅'
      case 'project_updated':
        return '📁'
      case 'comment_added':
        return '💬'
      case 'meeting_scheduled':
        return '📅'
      default:
        return '📋'
    }
  }

  return (
    <div className="p-4 space-y-6">
      {/* Welcome Header */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">
              Good morning, {currentUser?.name?.split(' ')[0]}! 👋
            </h1>
            <p className="text-muted-foreground">
              Here's what's happening with your projects today
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
            className="p-2"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        {/* Connection Status */}
        <div className="flex items-center space-x-2 text-sm">
          {isOnline ? (
            <>
              <Wifi className="h-4 w-4 text-green-500" />
              <span className="text-green-500">Online</span>
            </>
          ) : (
            <>
              <WifiOff className="h-4 w-4 text-red-500" />
              <span className="text-red-500">Offline</span>
            </>
          )}
          <span className="text-muted-foreground">
            • Last updated {lastRefresh.toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 gap-4">
        {dashboardData.kpis.map((kpi, index) => {
          const Icon = kpi.icon
          return (
            <Card key={index} className="relative overflow-hidden">
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <Icon className={`h-5 w-5 ${kpi.color}`} />
                  <div className="flex items-center space-x-1 text-xs">
                    {kpi.trend === 'up' ? (
                      <TrendingUp className="h-3 w-3 text-green-500" />
                    ) : (
                      <TrendingDown className="h-3 w-3 text-red-500" />
                    )}
                    <span className={kpi.trend === 'up' ? 'text-green-500' : 'text-red-500'}>
                      {kpi.change}
                    </span>
                  </div>
                </div>
                <div className="space-y-1">
                  <p className="text-2xl font-bold">{kpi.value}</p>
                  <p className="text-xs text-muted-foreground">{kpi.title}</p>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Weekly Progress Chart */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <span>Weekly Progress</span>
          </CardTitle>
          <CardDescription>Tasks completed this week</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={dashboardData.chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="completed" 
                stroke="#3b82f6" 
                fill="#3b82f6" 
                fillOpacity={0.3}
                name="Completed"
              />
              <Area 
                type="monotone" 
                dataKey="tasks" 
                stroke="#e5e7eb" 
                fill="#e5e7eb" 
                fillOpacity={0.3}
                name="Total Tasks"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Upcoming Tasks */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center space-x-2">
              <Clock className="h-5 w-5" />
              <span>Upcoming Tasks</span>
            </CardTitle>
            <Button variant="ghost" size="sm">
              <span className="text-sm">View All</span>
              <ArrowRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {dashboardData.upcomingTasks.map((task) => (
            <div key={task.id} className="flex items-center space-x-3 p-3 border rounded-lg">
              <div className={`w-3 h-3 rounded-full ${getPriorityColor(task.priority)}`} />
              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm truncate">{task.title}</p>
                <p className="text-xs text-muted-foreground">{task.project} • {task.assignee}</p>
              </div>
              <div className="text-right">
                <p className="text-xs font-medium">{task.dueTime}</p>
                <Badge variant="secondary" className="text-xs">
                  {task.priority}
                </Badge>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Recent Activity</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {dashboardData.recentActivity.map((activity, index) => (
            <div key={activity.id}>
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center text-sm">
                    {getActivityIcon(activity.type)}
                  </div>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">{activity.title}</p>
                  <p className="text-xs text-muted-foreground">
                    by {activity.user} • {activity.time}
                  </p>
                </div>
              </div>
              {index < dashboardData.recentActivity.length - 1 && (
                <Separator className="mt-3" />
              )}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center space-x-2">
            <Zap className="h-5 w-5" />
            <span>Quick Actions</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-3">
            <Button variant="outline" className="h-16 flex flex-col space-y-1">
              <span className="text-lg">✅</span>
              <span className="text-xs">New Task</span>
            </Button>
            <Button variant="outline" className="h-16 flex flex-col space-y-1">
              <span className="text-lg">📁</span>
              <span className="text-xs">New Project</span>
            </Button>
            <Button variant="outline" className="h-16 flex flex-col space-y-1">
              <span className="text-lg">🎥</span>
              <span className="text-xs">Start Meeting</span>
            </Button>
            <Button variant="outline" className="h-16 flex flex-col space-y-1">
              <span className="text-lg">📷</span>
              <span className="text-xs">Scan Document</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default HomeScreen

