import { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { 
  Home, 
  FolderOpen, 
  CheckSquare, 
  Users, 
  User,
  Bell,
  Plus
} from 'lucide-react'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { 
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from '../ui/drawer'

const BottomNavigation = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('/')
  const [isQuickActionOpen, setIsQuickActionOpen] = useState(false)

  useEffect(() => {
    setActiveTab(location.pathname)
  }, [location.pathname])

  const navigationItems = [
    {
      id: 'home',
      label: 'Home',
      icon: Home,
      path: '/',
      badge: null
    },
    {
      id: 'projects',
      label: 'Projects',
      icon: FolderOpen,
      path: '/projects',
      badge: null
    },
    {
      id: 'tasks',
      label: 'Tasks',
      icon: CheckSquare,
      path: '/tasks',
      badge: 5
    },
    {
      id: 'team',
      label: 'Team',
      icon: Users,
      path: '/team',
      badge: null
    },
    {
      id: 'profile',
      label: 'Profile',
      icon: User,
      path: '/profile',
      badge: null
    }
  ]

  const quickActions = [
    {
      id: 'new-task',
      label: 'New Task',
      icon: '✅',
      action: () => console.log('Create new task')
    },
    {
      id: 'new-project',
      label: 'New Project',
      icon: '📁',
      action: () => console.log('Create new project')
    },
    {
      id: 'scan-document',
      label: 'Scan Document',
      icon: '📷',
      action: () => console.log('Scan document')
    },
    {
      id: 'quick-note',
      label: 'Quick Note',
      icon: '📝',
      action: () => console.log('Create quick note')
    },
    {
      id: 'start-meeting',
      label: 'Start Meeting',
      icon: '🎥',
      action: () => console.log('Start meeting')
    },
    {
      id: 'time-tracker',
      label: 'Time Tracker',
      icon: '⏱️',
      action: () => console.log('Start time tracking')
    }
  ]

  const handleNavigation = (path) => {
    navigate(path)
    setActiveTab(path)
  }

  const handleQuickAction = (action) => {
    action()
    setIsQuickActionOpen(false)
  }

  return (
    <>
      {/* Bottom Navigation Bar */}
      <nav className="fixed bottom-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm border-t">
        <div className="flex items-center justify-around py-2 px-4">
          {navigationItems.map((item) => {
            const Icon = item.icon
            const isActive = activeTab === item.path
            
            return (
              <Button
                key={item.id}
                variant="ghost"
                size="sm"
                className={`flex flex-col items-center space-y-1 h-auto py-2 px-3 relative ${
                  isActive 
                    ? 'text-primary bg-primary/10' 
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => handleNavigation(item.path)}
              >
                <div className="relative">
                  <Icon className="h-5 w-5" />
                  {item.badge && (
                    <Badge 
                      variant="destructive" 
                      className="absolute -top-2 -right-2 h-4 w-4 flex items-center justify-center text-xs p-0"
                    >
                      {item.badge}
                    </Badge>
                  )}
                </div>
                <span className="text-xs font-medium">{item.label}</span>
                {isActive && (
                  <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-primary rounded-full" />
                )}
              </Button>
            )
          })}
        </div>
      </nav>

      {/* Floating Action Button */}
      <Drawer open={isQuickActionOpen} onOpenChange={setIsQuickActionOpen}>
        <DrawerTrigger asChild>
          <Button
            size="lg"
            className="fixed bottom-20 right-4 z-40 h-14 w-14 rounded-full shadow-lg"
          >
            <Plus className="h-6 w-6" />
          </Button>
        </DrawerTrigger>
        <DrawerContent>
          <DrawerHeader>
            <DrawerTitle>Quick Actions</DrawerTitle>
            <DrawerDescription>
              Choose an action to perform quickly
            </DrawerDescription>
          </DrawerHeader>
          
          <div className="p-4 pb-8">
            <div className="grid grid-cols-3 gap-4">
              {quickActions.map((action) => (
                <Button
                  key={action.id}
                  variant="outline"
                  className="flex flex-col items-center space-y-2 h-20 p-4"
                  onClick={() => handleQuickAction(action.action)}
                >
                  <span className="text-2xl">{action.icon}</span>
                  <span className="text-xs text-center">{action.label}</span>
                </Button>
              ))}
            </div>
          </div>
        </DrawerContent>
      </Drawer>

      {/* Notification Bell (when on notifications screen) */}
      {activeTab === '/notifications' && (
        <Button
          variant="ghost"
          size="sm"
          className="fixed top-4 right-4 z-40 bg-background/80 backdrop-blur-sm border"
          onClick={() => navigate('/notifications')}
        >
          <Bell className="h-4 w-4" />
        </Button>
      )}
    </>
  )
}

export default BottomNavigation

