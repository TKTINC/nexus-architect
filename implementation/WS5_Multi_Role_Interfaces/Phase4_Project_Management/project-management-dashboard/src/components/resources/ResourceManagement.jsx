import { useState } from 'react'
import { 
  Users, 
  DollarSign, 
  Clock, 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  PieChart,
  Calendar,
  Target,
  AlertTriangle,
  CheckCircle,
  Activity,
  Zap,
  Settings,
  Filter,
  Download,
  RefreshCw,
  Plus,
  Edit,
  Eye,
  MoreHorizontal,
  User,
  Briefcase,
  Award,
  Star,
  ArrowUp,
  ArrowDown,
  Minus
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Progress } from '../ui/progress'
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '../ui/select'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuLabel, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from '../ui/dropdown-menu'
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  PieChart as RechartsPieChart, 
  Cell, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts'
import { teamMembers, resourceData, performanceData, capacityData } from '../../data/mockData'

const ResourceManagement = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState('month')
  const [selectedDepartment, setSelectedDepartment] = useState('all')
  const [viewMode, setViewMode] = useState('overview')

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up':
        return <ArrowUp className="h-4 w-4 text-green-500" />
      case 'down':
        return <ArrowDown className="h-4 w-4 text-red-500" />
      case 'stable':
        return <Minus className="h-4 w-4 text-yellow-500" />
      default:
        return <Minus className="h-4 w-4 text-gray-500" />
    }
  }

  const getUtilizationColor = (utilization) => {
    if (utilization >= 90) return 'text-red-500'
    if (utilization >= 75) return 'text-yellow-500'
    return 'text-green-500'
  }

  const getSkillLevelColor = (level) => {
    switch (level.toLowerCase()) {
      case 'expert':
        return 'bg-purple-500'
      case 'senior':
        return 'bg-blue-500'
      case 'intermediate':
        return 'bg-green-500'
      case 'junior':
        return 'bg-yellow-500'
      default:
        return 'bg-gray-500'
    }
  }

  const resourceMetrics = {
    totalTeamMembers: teamMembers.length,
    activeProjects: 8,
    averageUtilization: 87,
    totalBudget: 2800000,
    spentBudget: 1950000,
    availableHours: 1680,
    allocatedHours: 1462
  }

  const utilizationData = [
    { name: 'Jan', utilization: 82, capacity: 100, efficiency: 78 },
    { name: 'Feb', utilization: 85, capacity: 100, efficiency: 81 },
    { name: 'Mar', utilization: 89, capacity: 100, efficiency: 85 },
    { name: 'Apr', utilization: 87, capacity: 100, efficiency: 83 },
    { name: 'May', utilization: 91, capacity: 100, efficiency: 87 },
    { name: 'Jun', utilization: 88, capacity: 100, efficiency: 84 }
  ]

  const departmentData = [
    { name: 'Development', members: 12, utilization: 92, budget: 1200000, color: '#3b82f6' },
    { name: 'Design', members: 6, utilization: 78, budget: 480000, color: '#10b981' },
    { name: 'QA', members: 4, utilization: 85, budget: 320000, color: '#f59e0b' },
    { name: 'DevOps', members: 3, utilization: 95, budget: 360000, color: '#ef4444' },
    { name: 'Management', members: 3, utilization: 70, budget: 440000, color: '#8b5cf6' }
  ]

  const skillsData = [
    { skill: 'React', demand: 95, supply: 78, gap: -17 },
    { skill: 'Node.js', demand: 88, supply: 85, gap: -3 },
    { skill: 'Python', demand: 82, supply: 90, gap: 8 },
    { skill: 'DevOps', demand: 90, supply: 65, gap: -25 },
    { skill: 'UI/UX', demand: 75, supply: 80, gap: 5 },
    { skill: 'Data Science', demand: 70, supply: 45, gap: -25 }
  ]

  const upcomingAllocations = [
    {
      id: 1,
      project: 'Project Alpha',
      member: 'John Doe',
      role: 'Lead Developer',
      startDate: '2024-01-15',
      endDate: '2024-03-15',
      allocation: 80,
      status: 'confirmed'
    },
    {
      id: 2,
      project: 'Project Beta',
      member: 'Alice Johnson',
      role: 'UI Designer',
      startDate: '2024-01-20',
      endDate: '2024-02-28',
      allocation: 100,
      status: 'pending'
    },
    {
      id: 3,
      project: 'Project Gamma',
      member: 'Bob Wilson',
      role: 'DevOps Engineer',
      startDate: '2024-02-01',
      endDate: '2024-04-30',
      allocation: 60,
      status: 'confirmed'
    }
  ]

  const TeamMemberCard = ({ member }) => (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-3">
            <Avatar className="h-10 w-10">
              <AvatarImage src={member.avatar} alt={member.name} />
              <AvatarFallback>{member.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="font-medium text-sm">{member.name}</h3>
              <p className="text-xs text-muted-foreground">{member.role}</p>
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="p-1">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem>
                <Eye className="h-4 w-4 mr-2" />
                View Profile
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Edit className="h-4 w-4 mr-2" />
                Edit Allocation
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Calendar className="h-4 w-4 mr-2" />
                View Schedule
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        
        <div className="space-y-3">
          <div>
            <div className="flex items-center justify-between text-xs mb-1">
              <span>Utilization</span>
              <span className={getUtilizationColor(member.utilization)}>{member.utilization}%</span>
            </div>
            <Progress value={member.utilization} className="h-2" />
          </div>
          
          <div className="flex items-center justify-between text-xs">
            <span>Current Projects</span>
            <span>{member.currentProjects || 2}</span>
          </div>
          
          <div className="flex items-center justify-between text-xs">
            <span>Hourly Rate</span>
            <span>${member.hourlyRate || 85}/hr</span>
          </div>
          
          <div className="flex flex-wrap gap-1">
            {(member.skills || ['React', 'Node.js']).slice(0, 3).map((skill, index) => (
              <Badge key={index} variant="secondary" className="text-xs">
                {skill}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Resource Management</h1>
          <p className="text-muted-foreground">Optimize team allocation and track performance</p>
        </div>
        <div className="flex items-center space-x-2">
          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="week">This Week</SelectItem>
              <SelectItem value="month">This Month</SelectItem>
              <SelectItem value="quarter">This Quarter</SelectItem>
              <SelectItem value="year">This Year</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button size="sm">
            <Plus className="h-4 w-4 mr-2" />
            Add Resource
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Team Members</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{resourceMetrics.totalTeamMembers}</div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-500">+2</span> from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Utilization Rate</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{resourceMetrics.averageUtilization}%</div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-500">+3%</span> from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Budget Utilization</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Math.round((resourceMetrics.spentBudget / resourceMetrics.totalBudget) * 100)}%
            </div>
            <p className="text-xs text-muted-foreground">
              ${(resourceMetrics.totalBudget - resourceMetrics.spentBudget) / 1000}K remaining
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Available Hours</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {resourceMetrics.availableHours - resourceMetrics.allocatedHours}h
            </div>
            <p className="text-xs text-muted-foreground">
              {Math.round(((resourceMetrics.availableHours - resourceMetrics.allocatedHours) / resourceMetrics.availableHours) * 100)}% capacity available
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="team">Team</TabsTrigger>
          <TabsTrigger value="capacity">Capacity</TabsTrigger>
          <TabsTrigger value="skills">Skills</TabsTrigger>
          <TabsTrigger value="allocation">Allocation</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Utilization Trends */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5" />
                  <span>Utilization Trends</span>
                </CardTitle>
                <CardDescription>Team utilization over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={utilizationData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="utilization" stroke="#3b82f6" name="Utilization %" />
                    <Line type="monotone" dataKey="efficiency" stroke="#10b981" name="Efficiency %" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Department Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <PieChart className="h-5 w-5" />
                  <span>Department Distribution</span>
                </CardTitle>
                <CardDescription>Team members by department</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <RechartsPieChart>
                    <Pie
                      data={departmentData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      paddingAngle={5}
                      dataKey="members"
                    >
                      {departmentData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </RechartsPieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Department Performance */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <BarChart3 className="h-5 w-5" />
                <span>Department Performance</span>
              </CardTitle>
              <CardDescription>Utilization and budget by department</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {departmentData.map((dept) => (
                  <div key={dept.name} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full`} style={{ backgroundColor: dept.color }}></div>
                      <div>
                        <h4 className="font-medium text-sm">{dept.name}</h4>
                        <p className="text-xs text-muted-foreground">{dept.members} members</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="text-sm font-medium">{dept.utilization}%</p>
                        <p className="text-xs text-muted-foreground">Utilization</p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">${(dept.budget / 1000).toFixed(0)}K</p>
                        <p className="text-xs text-muted-foreground">Budget</p>
                      </div>
                      <Progress value={dept.utilization} className="w-20" />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Team Tab */}
        <TabsContent value="team">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <Select value={selectedDepartment} onValueChange={setSelectedDepartment}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="All Departments" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Departments</SelectItem>
                    <SelectItem value="development">Development</SelectItem>
                    <SelectItem value="design">Design</SelectItem>
                    <SelectItem value="qa">QA</SelectItem>
                    <SelectItem value="devops">DevOps</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button variant="outline" size="sm">
                <Filter className="h-4 w-4 mr-2" />
                Filter
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {teamMembers.map((member) => (
                <TeamMemberCard key={member.id} member={member} />
              ))}
            </div>
          </div>
        </TabsContent>

        {/* Capacity Tab */}
        <TabsContent value="capacity">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target className="h-5 w-5" />
                <span>Capacity Planning</span>
              </CardTitle>
              <CardDescription>Current and projected capacity utilization</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={utilizationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="capacity" stackId="1" stroke="#e5e7eb" fill="#e5e7eb" name="Total Capacity" />
                  <Area type="monotone" dataKey="utilization" stackId="2" stroke="#3b82f6" fill="#3b82f6" name="Current Utilization" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Skills Tab */}
        <TabsContent value="skills">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Award className="h-5 w-5" />
                <span>Skills Gap Analysis</span>
              </CardTitle>
              <CardDescription>Demand vs supply for key skills</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {skillsData.map((skill) => (
                  <div key={skill.skill} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-medium text-sm">{skill.skill}</h4>
                      <div className="flex items-center space-x-2">
                        <Badge variant={skill.gap < 0 ? 'destructive' : 'default'} className="text-xs">
                          {skill.gap > 0 ? '+' : ''}{skill.gap}%
                        </Badge>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span>Demand</span>
                          <span>{skill.demand}%</span>
                        </div>
                        <Progress value={skill.demand} className="h-2" />
                      </div>
                      <div>
                        <div className="flex items-center justify-between text-xs mb-1">
                          <span>Supply</span>
                          <span>{skill.supply}%</span>
                        </div>
                        <Progress value={skill.supply} className="h-2" />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Allocation Tab */}
        <TabsContent value="allocation">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Calendar className="h-5 w-5" />
                <span>Upcoming Allocations</span>
              </CardTitle>
              <CardDescription>Planned resource allocations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {upcomingAllocations.map((allocation) => (
                  <div key={allocation.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Avatar className="h-8 w-8">
                        <AvatarFallback className="text-xs">{allocation.member.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                      </Avatar>
                      <div>
                        <h4 className="font-medium text-sm">{allocation.member}</h4>
                        <p className="text-xs text-muted-foreground">{allocation.role} • {allocation.project}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right text-xs">
                        <p>{allocation.startDate} - {allocation.endDate}</p>
                        <p className="text-muted-foreground">{allocation.allocation}% allocation</p>
                      </div>
                      <Badge variant={allocation.status === 'confirmed' ? 'default' : 'secondary'}>
                        {allocation.status}
                      </Badge>
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

export default ResourceManagement

