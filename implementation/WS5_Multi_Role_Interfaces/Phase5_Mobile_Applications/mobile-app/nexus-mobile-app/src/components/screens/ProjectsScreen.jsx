import { useState, useEffect } from 'react'
import { 
  Search, 
  Filter, 
  Plus, 
  MoreVertical,
  Calendar,
  Users,
  Target,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Clock,
  ArrowRight
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { Progress } from '../ui/progress'
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar'
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { 
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '../ui/sheet'

const ProjectsScreen = ({ isOnline, syncStatus }) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedFilter, setSelectedFilter] = useState('all')
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)

  // Mock project data
  useEffect(() => {
    const mockProjects = [
      {
        id: 1,
        name: 'Project Alpha',
        description: 'Next-generation mobile application development',
        status: 'active',
        priority: 'high',
        progress: 75,
        dueDate: '2024-02-15',
        team: [
          { id: 1, name: 'Alice Johnson', avatar: '/api/placeholder/32/32' },
          { id: 2, name: 'Bob Wilson', avatar: '/api/placeholder/32/32' },
          { id: 3, name: 'Carol Davis', avatar: '/api/placeholder/32/32' }
        ],
        tasksTotal: 24,
        tasksCompleted: 18,
        budget: 150000,
        spent: 112500,
        lastActivity: '2 hours ago'
      },
      {
        id: 2,
        name: 'Project Beta',
        description: 'Enterprise dashboard redesign and optimization',
        status: 'planning',
        priority: 'medium',
        progress: 25,
        dueDate: '2024-03-20',
        team: [
          { id: 4, name: 'David Chen', avatar: '/api/placeholder/32/32' },
          { id: 5, name: 'Eva Martinez', avatar: '/api/placeholder/32/32' }
        ],
        tasksTotal: 16,
        tasksCompleted: 4,
        budget: 80000,
        spent: 20000,
        lastActivity: '1 day ago'
      },
      {
        id: 3,
        name: 'Project Gamma',
        description: 'AI-powered analytics and reporting system',
        status: 'review',
        priority: 'high',
        progress: 90,
        dueDate: '2024-01-30',
        team: [
          { id: 6, name: 'Frank Thompson', avatar: '/api/placeholder/32/32' },
          { id: 7, name: 'Grace Lee', avatar: '/api/placeholder/32/32' },
          { id: 8, name: 'Henry Kim', avatar: '/api/placeholder/32/32' },
          { id: 9, name: 'Iris Wang', avatar: '/api/placeholder/32/32' }
        ],
        tasksTotal: 32,
        tasksCompleted: 29,
        budget: 200000,
        spent: 180000,
        lastActivity: '30 minutes ago'
      },
      {
        id: 4,
        name: 'Project Delta',
        description: 'Cloud infrastructure migration and optimization',
        status: 'on-hold',
        priority: 'low',
        progress: 45,
        dueDate: '2024-04-15',
        team: [
          { id: 10, name: 'Jack Brown', avatar: '/api/placeholder/32/32' }
        ],
        tasksTotal: 20,
        tasksCompleted: 9,
        budget: 120000,
        spent: 54000,
        lastActivity: '1 week ago'
      }
    ]

    setTimeout(() => {
      setProjects(mockProjects)
      setLoading(false)
    }, 1000)
  }, [])

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'bg-green-500'
      case 'planning':
        return 'bg-blue-500'
      case 'review':
        return 'bg-yellow-500'
      case 'on-hold':
        return 'bg-gray-500'
      case 'completed':
        return 'bg-purple-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return 'text-red-500 bg-red-50'
      case 'medium':
        return 'text-yellow-500 bg-yellow-50'
      case 'low':
        return 'text-green-500 bg-green-50'
      default:
        return 'text-gray-500 bg-gray-50'
    }
  }

  const getProgressColor = (progress) => {
    if (progress >= 80) return 'bg-green-500'
    if (progress >= 60) return 'bg-blue-500'
    if (progress >= 40) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  const filteredProjects = projects.filter(project => {
    const matchesSearch = project.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         project.description.toLowerCase().includes(searchQuery.toLowerCase())
    
    if (selectedFilter === 'all') return matchesSearch
    return matchesSearch && project.status === selectedFilter
  })

  const filters = [
    { id: 'all', label: 'All Projects', count: projects.length },
    { id: 'active', label: 'Active', count: projects.filter(p => p.status === 'active').length },
    { id: 'planning', label: 'Planning', count: projects.filter(p => p.status === 'planning').length },
    { id: 'review', label: 'Review', count: projects.filter(p => p.status === 'review').length },
    { id: 'on-hold', label: 'On Hold', count: projects.filter(p => p.status === 'on-hold').length }
  ]

  if (loading) {
    return (
      <div className="p-4 space-y-4">
        <div className="animate-pulse space-y-4">
          {[1, 2, 3].map(i => (
            <Card key={i}>
              <CardContent className="p-4">
                <div className="space-y-3">
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                  <div className="h-2 bg-gray-200 rounded w-full"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Projects</h1>
          <p className="text-muted-foreground">
            {projects.length} projects • {projects.filter(p => p.status === 'active').length} active
          </p>
        </div>
        <Button size="sm" className="rounded-full">
          <Plus className="h-4 w-4 mr-1" />
          New
        </Button>
      </div>

      {/* Search and Filter */}
      <div className="space-y-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search projects..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>

        {/* Filter Tabs */}
        <div className="flex space-x-2 overflow-x-auto pb-2">
          {filters.map((filter) => (
            <Button
              key={filter.id}
              variant={selectedFilter === filter.id ? "default" : "outline"}
              size="sm"
              className="whitespace-nowrap"
              onClick={() => setSelectedFilter(filter.id)}
            >
              {filter.label}
              <Badge variant="secondary" className="ml-2 text-xs">
                {filter.count}
              </Badge>
            </Button>
          ))}
        </div>
      </div>

      {/* Sync Status */}
      {!isOnline && (
        <Card className="bg-yellow-50 border-yellow-200">
          <CardContent className="p-3">
            <div className="flex items-center space-x-2 text-sm">
              <AlertTriangle className="h-4 w-4 text-yellow-600" />
              <span>Viewing cached projects. Some data may be outdated.</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Projects List */}
      <div className="space-y-4">
        {filteredProjects.map((project) => (
          <Card key={project.id} className="overflow-hidden">
            <CardContent className="p-4">
              <div className="space-y-4">
                {/* Project Header */}
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(project.status)}`} />
                      <h3 className="font-semibold text-lg truncate">{project.name}</h3>
                      <Badge className={getPriorityColor(project.priority)}>
                        {project.priority}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {project.description}
                    </p>
                  </div>
                  
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="sm" className="p-1">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem>View Details</DropdownMenuItem>
                      <DropdownMenuItem>Edit Project</DropdownMenuItem>
                      <DropdownMenuItem>Share</DropdownMenuItem>
                      <DropdownMenuItem className="text-red-600">Archive</DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                {/* Progress */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span>Progress</span>
                    <span className="font-medium">{project.progress}%</span>
                  </div>
                  <Progress value={project.progress} className="h-2" />
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-muted-foreground">Tasks</span>
                    </div>
                    <p className="font-medium">
                      {project.tasksCompleted}/{project.tasksTotal} completed
                    </p>
                  </div>
                  
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <Calendar className="h-4 w-4 text-blue-500" />
                      <span className="text-muted-foreground">Due Date</span>
                    </div>
                    <p className="font-medium">
                      {new Date(project.dueDate).toLocaleDateString()}
                    </p>
                  </div>
                </div>

                {/* Team and Budget */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="flex -space-x-2">
                      {project.team.slice(0, 3).map((member) => (
                        <Avatar key={member.id} className="h-6 w-6 border-2 border-background">
                          <AvatarImage src={member.avatar} alt={member.name} />
                          <AvatarFallback className="text-xs">
                            {member.name.split(' ').map(n => n[0]).join('')}
                          </AvatarFallback>
                        </Avatar>
                      ))}
                      {project.team.length > 3 && (
                        <div className="h-6 w-6 rounded-full bg-muted border-2 border-background flex items-center justify-center">
                          <span className="text-xs font-medium">+{project.team.length - 3}</span>
                        </div>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {project.team.length} members
                    </span>
                  </div>
                  
                  <div className="text-right">
                    <p className="text-sm font-medium">
                      ${(project.spent / 1000).toFixed(0)}k / ${(project.budget / 1000).toFixed(0)}k
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {Math.round((project.spent / project.budget) * 100)}% spent
                    </p>
                  </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between pt-2 border-t">
                  <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    <span>Updated {project.lastActivity}</span>
                  </div>
                  
                  <Button variant="ghost" size="sm" className="h-8 px-3">
                    <span className="text-sm">View Details</span>
                    <ArrowRight className="h-3 w-3 ml-1" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Empty State */}
      {filteredProjects.length === 0 && (
        <Card>
          <CardContent className="p-8 text-center">
            <Target className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="font-semibold mb-2">No projects found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery 
                ? `No projects match "${searchQuery}"`
                : `No projects in ${selectedFilter} status`
              }
            </p>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Create New Project
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default ProjectsScreen

