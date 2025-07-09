import { useState } from 'react'
import { 
  Menu, 
  Bell, 
  Search, 
  Settings, 
  User,
  Wifi,
  WifiOff,
  Battery,
  Signal
} from 'lucide-react'
import { Button } from '../ui/button'
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar'
import { Badge } from '../ui/badge'
import { 
  Sheet, 
  SheetContent, 
  SheetDescription, 
  SheetHeader, 
  SheetTitle, 
  SheetTrigger 
} from '../ui/sheet'
import { Input } from '../ui/input'

const MobileLayout = ({ children, currentUser }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [batteryLevel, setBatteryLevel] = useState(85)
  const [signalStrength, setSignalStrength] = useState(4)

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊', path: '/' },
    { id: 'projects', label: 'Projects', icon: '📁', path: '/projects' },
    { id: 'tasks', label: 'Tasks', icon: '✅', path: '/tasks' },
    { id: 'team', label: 'Team', icon: '👥', path: '/team' },
    { id: 'calendar', label: 'Calendar', icon: '📅', path: '/calendar' },
    { id: 'documents', label: 'Documents', icon: '📄', path: '/documents' },
    { id: 'analytics', label: 'Analytics', icon: '📈', path: '/analytics' },
    { id: 'settings', label: 'Settings', icon: '⚙️', path: '/settings' }
  ]

  const StatusBar = () => (
    <div className="flex items-center justify-between px-4 py-1 bg-background/95 backdrop-blur-sm border-b text-xs">
      <div className="flex items-center space-x-2">
        <span className="font-medium">9:41</span>
        <div className="flex items-center space-x-1">
          <Signal className="h-3 w-3" />
          <span>{signalStrength}/4</span>
        </div>
      </div>
      <div className="flex items-center space-x-2">
        <Wifi className="h-3 w-3" />
        <div className="flex items-center space-x-1">
          <Battery className="h-3 w-3" />
          <span>{batteryLevel}%</span>
        </div>
      </div>
    </div>
  )

  const Header = () => (
    <header className="sticky top-0 z-50 bg-background/95 backdrop-blur-sm border-b">
      <StatusBar />
      <div className="flex items-center justify-between p-4">
        <div className="flex items-center space-x-3">
          <Sheet open={isMenuOpen} onOpenChange={setIsMenuOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="sm" className="p-2">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-80">
              <SheetHeader>
                <SheetTitle className="flex items-center space-x-3">
                  <Avatar className="h-10 w-10">
                    <AvatarImage src={currentUser?.avatar} alt={currentUser?.name} />
                    <AvatarFallback>
                      {currentUser?.name?.split(' ').map(n => n[0]).join('')}
                    </AvatarFallback>
                  </Avatar>
                  <div className="text-left">
                    <p className="font-medium">{currentUser?.name}</p>
                    <p className="text-sm text-muted-foreground">{currentUser?.role}</p>
                  </div>
                </SheetTitle>
                <SheetDescription>
                  Navigate through Nexus Architect
                </SheetDescription>
              </SheetHeader>
              
              <div className="mt-6 space-y-2">
                {menuItems.map((item) => (
                  <Button
                    key={item.id}
                    variant="ghost"
                    className="w-full justify-start space-x-3 h-12"
                    onClick={() => {
                      window.location.href = item.path
                      setIsMenuOpen(false)
                    }}
                  >
                    <span className="text-lg">{item.icon}</span>
                    <span>{item.label}</span>
                  </Button>
                ))}
              </div>
              
              <div className="absolute bottom-4 left-4 right-4">
                <Button 
                  variant="outline" 
                  className="w-full"
                  onClick={() => setIsMenuOpen(false)}
                >
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </Button>
              </div>
            </SheetContent>
          </Sheet>
          
          <div className="flex-1 max-w-sm">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 h-9 bg-muted/50"
              />
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button variant="ghost" size="sm" className="relative p-2">
            <Bell className="h-5 w-5" />
            <Badge 
              variant="destructive" 
              className="absolute -top-1 -right-1 h-5 w-5 flex items-center justify-center text-xs p-0"
            >
              3
            </Badge>
          </Button>
          
          <Avatar className="h-8 w-8">
            <AvatarImage src={currentUser?.avatar} alt={currentUser?.name} />
            <AvatarFallback className="text-xs">
              {currentUser?.name?.split(' ').map(n => n[0]).join('')}
            </AvatarFallback>
          </Avatar>
        </div>
      </div>
    </header>
  )

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      {/* Main Content */}
      <main className="pb-20">
        {children}
      </main>
    </div>
  )
}

export default MobileLayout

