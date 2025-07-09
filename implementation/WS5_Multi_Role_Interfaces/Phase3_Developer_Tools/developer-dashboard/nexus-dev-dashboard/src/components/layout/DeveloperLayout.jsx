import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  Code2, 
  BarChart3, 
  Workflow, 
  GraduationCap, 
  Puzzle, 
  Settings,
  Bell,
  Search,
  Moon,
  Sun,
  Menu,
  X,
  Home,
  Activity,
  GitBranch,
  Bug,
  Zap
} from 'lucide-react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar'
import { Badge } from '../ui/badge'

const DeveloperLayout = ({ children, user, theme, onThemeToggle }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const location = useLocation()

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home, current: location.pathname === '/dashboard' || location.pathname === '/' },
    { name: 'Code Quality', href: '/code-quality', icon: BarChart3, current: location.pathname === '/code-quality' },
    { name: 'Workflow', href: '/workflow', icon: Workflow, current: location.pathname === '/workflow' },
    { name: 'Learning Center', href: '/learning', icon: GraduationCap, current: location.pathname === '/learning' },
    { name: 'IDE Integration', href: '/ide-integration', icon: Puzzle, current: location.pathname === '/ide-integration' },
  ]

  const quickActions = [
    { name: 'Run Tests', icon: Activity, color: 'bg-green-500' },
    { name: 'Deploy', icon: Zap, color: 'bg-blue-500' },
    { name: 'Create Branch', icon: GitBranch, color: 'bg-purple-500' },
    { name: 'Report Bug', icon: Bug, color: 'bg-red-500' },
  ]

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-16'} transition-all duration-300 bg-sidebar border-r border-sidebar-border flex flex-col`}>
        {/* Sidebar Header */}
        <div className="p-4 border-b border-sidebar-border">
          <div className="flex items-center justify-between">
            {sidebarOpen && (
              <div className="flex items-center space-x-2">
                <Code2 className="h-8 w-8 text-sidebar-primary" />
                <span className="text-lg font-bold text-sidebar-foreground">Nexus Dev</span>
              </div>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-sidebar-foreground hover:bg-sidebar-accent"
            >
              {sidebarOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {/* User Profile */}
        {sidebarOpen && (
          <div className="p-4 border-b border-sidebar-border">
            <div className="flex items-center space-x-3">
              <Avatar className="h-10 w-10">
                <AvatarImage src={user.avatar} alt={user.name} />
                <AvatarFallback>{user.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
              </Avatar>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-sidebar-foreground truncate">{user.name}</p>
                <p className="text-xs text-sidebar-foreground/70 truncate">{user.role}</p>
                <div className="flex items-center space-x-1 mt-1">
                  <Badge variant="secondary" className="text-xs">{user.experience}</Badge>
                  <Badge variant="outline" className="text-xs">{user.team}</Badge>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navigation.map((item) => {
            const Icon = item.icon
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  item.current
                    ? 'bg-sidebar-accent text-sidebar-accent-foreground'
                    : 'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'
                }`}
              >
                <Icon className="h-5 w-5 mr-3" />
                {sidebarOpen && item.name}
              </Link>
            )
          })}
        </nav>

        {/* Quick Actions */}
        {sidebarOpen && (
          <div className="p-4 border-t border-sidebar-border">
            <h3 className="text-xs font-semibold text-sidebar-foreground/70 uppercase tracking-wider mb-3">
              Quick Actions
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {quickActions.map((action) => {
                const Icon = action.icon
                return (
                  <Button
                    key={action.name}
                    variant="outline"
                    size="sm"
                    className="h-auto p-2 flex flex-col items-center space-y-1 text-xs"
                  >
                    <div className={`p-1 rounded ${action.color}`}>
                      <Icon className="h-3 w-3 text-white" />
                    </div>
                    <span className="text-xs">{action.name}</span>
                  </Button>
                )
              })}
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-background border-b border-border px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search code, docs, or ask AI..."
                  className="pl-10 w-96"
                />
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Theme Toggle */}
              <Button
                variant="ghost"
                size="sm"
                onClick={onThemeToggle}
                className="text-foreground"
              >
                {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
              </Button>

              {/* Notifications */}
              <Button variant="ghost" size="sm" className="relative">
                <Bell className="h-4 w-4" />
                <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-500 rounded-full text-xs"></span>
              </Button>

              {/* Settings */}
              <Button variant="ghost" size="sm">
                <Settings className="h-4 w-4" />
              </Button>

              {/* User Avatar */}
              <Avatar className="h-8 w-8">
                <AvatarImage src={user.avatar} alt={user.name} />
                <AvatarFallback>{user.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
              </Avatar>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto bg-background">
          {children}
        </main>
      </div>
    </div>
  )
}

export default DeveloperLayout

