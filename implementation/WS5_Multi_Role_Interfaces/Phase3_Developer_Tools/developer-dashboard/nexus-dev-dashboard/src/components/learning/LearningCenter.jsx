import { useState } from 'react'
import { 
  GraduationCap, 
  BookOpen, 
  Video, 
  FileText, 
  Award,
  TrendingUp,
  Clock,
  Star,
  Play,
  CheckCircle,
  Target,
  Users,
  Calendar,
  Filter,
  Search
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Progress } from '../ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { Input } from '../ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select'

import { learningRecommendations } from '../../data/mockData'

const LearningCenter = () => {
  const [searchTerm, setSearchTerm] = useState('')
  const [filterType, setFilterType] = useState('all')
  const [filterDifficulty, setFilterDifficulty] = useState('all')

  const skillProgress = [
    { skill: 'React', level: 85, target: 90, category: 'Frontend' },
    { skill: 'Python', level: 78, target: 85, category: 'Backend' },
    { skill: 'TypeScript', level: 72, target: 80, category: 'Frontend' },
    { skill: 'Docker', level: 45, target: 70, category: 'DevOps' },
    { skill: 'AWS', level: 38, target: 60, category: 'Cloud' },
    { skill: 'GraphQL', level: 25, target: 50, category: 'API' }
  ]

  const achievements = [
    {
      id: 1,
      title: 'Code Quality Champion',
      description: 'Maintained 95%+ test coverage for 30 days',
      icon: Award,
      earned: true,
      date: '2024-01-05'
    },
    {
      id: 2,
      title: 'Performance Optimizer',
      description: 'Improved application performance by 40%',
      icon: TrendingUp,
      earned: true,
      date: '2024-01-03'
    },
    {
      id: 3,
      title: 'Security Expert',
      description: 'Complete security best practices course',
      icon: Target,
      earned: false,
      progress: 75
    },
    {
      id: 4,
      title: 'Team Collaborator',
      description: 'Complete 50 code reviews',
      icon: Users,
      earned: false,
      progress: 32
    }
  ]

  const learningPaths = [
    {
      id: 1,
      title: 'Full-Stack Development Mastery',
      description: 'Complete path from frontend to backend development',
      duration: '12 weeks',
      courses: 8,
      progress: 45,
      difficulty: 'Advanced',
      skills: ['React', 'Node.js', 'Database Design', 'API Development']
    },
    {
      id: 2,
      title: 'DevOps Engineering',
      description: 'Learn containerization, CI/CD, and cloud deployment',
      duration: '8 weeks',
      courses: 6,
      progress: 12,
      difficulty: 'Intermediate',
      skills: ['Docker', 'Kubernetes', 'AWS', 'CI/CD']
    },
    {
      id: 3,
      title: 'Security Best Practices',
      description: 'Comprehensive security training for developers',
      duration: '6 weeks',
      courses: 5,
      progress: 0,
      difficulty: 'Intermediate',
      skills: ['Security', 'Authentication', 'Encryption', 'Compliance']
    }
  ]

  const upcomingEvents = [
    {
      id: 1,
      title: 'React Performance Workshop',
      type: 'Workshop',
      date: '2024-01-10',
      time: '2:00 PM',
      duration: '2 hours',
      instructor: 'Sarah Chen',
      attendees: 24
    },
    {
      id: 2,
      title: 'Microservices Architecture Talk',
      type: 'Webinar',
      date: '2024-01-12',
      time: '11:00 AM',
      duration: '1 hour',
      instructor: 'Mike Johnson',
      attendees: 156
    },
    {
      id: 3,
      title: 'Code Review Best Practices',
      type: 'Training',
      date: '2024-01-15',
      time: '10:00 AM',
      duration: '3 hours',
      instructor: 'Alex Rodriguez',
      attendees: 18
    }
  ]

  const getTypeIcon = (type) => {
    switch (type) {
      case 'course':
        return <BookOpen className="h-4 w-4" />
      case 'workshop':
        return <Users className="h-4 w-4" />
      case 'tutorial':
        return <Play className="h-4 w-4" />
      case 'documentation':
        return <FileText className="h-4 w-4" />
      default:
        return <BookOpen className="h-4 w-4" />
    }
  }

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Beginner':
        return 'text-green-500'
      case 'Intermediate':
        return 'text-yellow-500'
      case 'Advanced':
        return 'text-red-500'
      default:
        return 'text-gray-500'
    }
  }

  const getSkillLevelColor = (level) => {
    if (level >= 80) return 'text-green-500'
    if (level >= 60) return 'text-yellow-500'
    if (level >= 40) return 'text-orange-500'
    return 'text-red-500'
  }

  const filteredRecommendations = learningRecommendations.filter(item => {
    const matchesSearch = item.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         item.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesType = filterType === 'all' || item.type === filterType
    const matchesDifficulty = filterDifficulty === 'all' || item.difficulty === filterDifficulty
    
    return matchesSearch && matchesType && matchesDifficulty
  })

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Learning Center</h1>
          <p className="text-muted-foreground">
            Personalized learning recommendations and skill development tracking
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Calendar className="h-4 w-4 mr-2" />
            Schedule Learning
          </Button>
          <Button size="sm">
            <Target className="h-4 w-4 mr-2" />
            Set Goals
          </Button>
        </div>
      </div>

      {/* Learning Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Hours This Week</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8.5</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
              +2.3 hours from last week
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Courses Completed</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">12</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
              +3 this month
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Skill Level</CardTitle>
            <Star className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Advanced</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
              Improved from Intermediate
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Achievements</CardTitle>
            <Award className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">8</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
              +2 this month
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="recommendations" className="space-y-6">
        <TabsList>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          <TabsTrigger value="skills">Skills</TabsTrigger>
          <TabsTrigger value="paths">Learning Paths</TabsTrigger>
          <TabsTrigger value="events">Events</TabsTrigger>
          <TabsTrigger value="achievements">Achievements</TabsTrigger>
        </TabsList>

        <TabsContent value="recommendations" className="space-y-6">
          {/* Search and Filters */}
          <div className="flex items-center space-x-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search learning content..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="course">Courses</SelectItem>
                <SelectItem value="workshop">Workshops</SelectItem>
                <SelectItem value="tutorial">Tutorials</SelectItem>
                <SelectItem value="documentation">Documentation</SelectItem>
              </SelectContent>
            </Select>
            <Select value={filterDifficulty} onValueChange={setFilterDifficulty}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="Difficulty" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Levels</SelectItem>
                <SelectItem value="Beginner">Beginner</SelectItem>
                <SelectItem value="Intermediate">Intermediate</SelectItem>
                <SelectItem value="Advanced">Advanced</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Recommendations Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {filteredRecommendations.map((item) => (
              <Card key={item.id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-2">
                      {getTypeIcon(item.type)}
                      <CardTitle className="text-lg">{item.title}</CardTitle>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Star className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm">{item.relevance}%</span>
                    </div>
                  </div>
                  <CardDescription>{item.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Provider</span>
                      <span>{item.provider}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Duration</span>
                      <span>{item.duration}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Difficulty</span>
                      <Badge variant="outline" className={getDifficultyColor(item.difficulty)}>
                        {item.difficulty}
                      </Badge>
                    </div>
                    
                    {item.progress > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Progress</span>
                          <span>{item.progress}%</span>
                        </div>
                        <Progress value={item.progress} className="h-2" />
                      </div>
                    )}
                    
                    <div className="flex flex-wrap gap-1">
                      {item.skills.map((skill, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {skill}
                        </Badge>
                      ))}
                    </div>
                    
                    <div className="flex space-x-2">
                      {item.progress > 0 ? (
                        <Button size="sm" className="flex-1">
                          <Play className="h-4 w-4 mr-2" />
                          Continue
                        </Button>
                      ) : (
                        <Button size="sm" className="flex-1">
                          <Play className="h-4 w-4 mr-2" />
                          Start Learning
                        </Button>
                      )}
                      <Button size="sm" variant="outline">
                        Preview
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="skills" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Skill Progress</CardTitle>
              <CardDescription>Track your skill development and set learning goals</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {skillProgress.map((skill, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium">{skill.skill}</span>
                        <Badge variant="secondary">{skill.category}</Badge>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`text-sm font-medium ${getSkillLevelColor(skill.level)}`}>
                          {skill.level}%
                        </span>
                        <span className="text-xs text-muted-foreground">
                          Target: {skill.target}%
                        </span>
                      </div>
                    </div>
                    <div className="relative">
                      <Progress value={skill.level} className="h-3" />
                      <div 
                        className="absolute top-0 h-3 w-1 bg-border"
                        style={{ left: `${skill.target}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="paths" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {learningPaths.map((path) => (
              <Card key={path.id}>
                <CardHeader>
                  <CardTitle className="text-lg">{path.title}</CardTitle>
                  <CardDescription>{path.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Duration</span>
                      <span>{path.duration}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Courses</span>
                      <span>{path.courses} courses</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Difficulty</span>
                      <Badge variant="outline" className={getDifficultyColor(path.difficulty)}>
                        {path.difficulty}
                      </Badge>
                    </div>
                    
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Progress</span>
                        <span>{path.progress}%</span>
                      </div>
                      <Progress value={path.progress} className="h-2" />
                    </div>
                    
                    <div className="flex flex-wrap gap-1">
                      {path.skills.map((skill, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {skill}
                        </Badge>
                      ))}
                    </div>
                    
                    <Button size="sm" className="w-full">
                      {path.progress > 0 ? 'Continue Path' : 'Start Path'}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="events" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Upcoming Events</CardTitle>
              <CardDescription>Workshops, webinars, and training sessions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {upcomingEvents.map((event) => (
                  <div key={event.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="text-center">
                        <div className="text-lg font-bold">{new Date(event.date).getDate()}</div>
                        <div className="text-xs text-muted-foreground">
                          {new Date(event.date).toLocaleDateString('en', { month: 'short' })}
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <h4 className="font-medium">{event.title}</h4>
                          <Badge variant="outline">{event.type}</Badge>
                        </div>
                        <div className="flex items-center space-x-4 mt-1 text-sm text-muted-foreground">
                          <span>{event.time}</span>
                          <span>{event.duration}</span>
                          <span>by {event.instructor}</span>
                          <span>{event.attendees} attendees</span>
                        </div>
                      </div>
                    </div>
                    <Button size="sm">
                      Register
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="achievements" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {achievements.map((achievement) => {
              const Icon = achievement.icon
              return (
                <Card key={achievement.id} className={achievement.earned ? 'border-green-200' : ''}>
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <div className={`p-2 rounded-full ${achievement.earned ? 'bg-green-100' : 'bg-gray-100'}`}>
                        <Icon className={`h-6 w-6 ${achievement.earned ? 'text-green-600' : 'text-gray-400'}`} />
                      </div>
                      <div className="flex-1">
                        <CardTitle className="text-lg">{achievement.title}</CardTitle>
                        <CardDescription>{achievement.description}</CardDescription>
                      </div>
                      {achievement.earned && (
                        <CheckCircle className="h-6 w-6 text-green-500" />
                      )}
                    </div>
                  </CardHeader>
                  {!achievement.earned && achievement.progress && (
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">Progress</span>
                          <span>{achievement.progress}%</span>
                        </div>
                        <Progress value={achievement.progress} className="h-2" />
                      </div>
                    </CardContent>
                  )}
                  {achievement.earned && (
                    <CardContent>
                      <div className="text-sm text-muted-foreground">
                        Earned on {new Date(achievement.date).toLocaleDateString()}
                      </div>
                    </CardContent>
                  )}
                </Card>
              )
            })}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default LearningCenter

