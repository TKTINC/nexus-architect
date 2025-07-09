import React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '../atoms/Button'
import { 
  Menu, 
  Search, 
  Bell, 
  Settings, 
  User, 
  Sun, 
  Moon, 
  Monitor,
  ChevronDown,
  LogOut,
  UserCircle,
  HelpCircle
} from 'lucide-react'
import { useTheme } from 'next-themes'

const Header = React.forwardRef(({
  className,
  onMenuToggle,
  user = { name: 'John Doe', email: 'john@example.com', avatar: null },
  notifications = 3,
  showSearch = true,
  showNotifications = true,
  showUserMenu = true,
  showThemeToggle = true,
  ...props
}, ref) => {
  const { theme, setTheme } = useTheme()
  const [userMenuOpen, setUserMenuOpen] = React.useState(false)
  const [themeMenuOpen, setThemeMenuOpen] = React.useState(false)
  const [searchFocused, setSearchFocused] = React.useState(false)

  const themeOptions = [
    { value: 'light', label: 'Light', icon: Sun },
    { value: 'dark', label: 'Dark', icon: Moon },
    { value: 'system', label: 'System', icon: Monitor }
  ]

  const userMenuItems = [
    { label: 'Profile', icon: UserCircle, href: '/profile' },
    { label: 'Settings', icon: Settings, href: '/settings' },
    { label: 'Help', icon: HelpCircle, href: '/help' },
    { type: 'divider' },
    { label: 'Sign Out', icon: LogOut, action: 'logout', variant: 'destructive' }
  ]

  return (
    <header
      ref={ref}
      className={cn(
        "sticky top-0 z-40 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60",
        className
      )}
      {...props}
    >
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Left Section */}
        <div className="flex items-center space-x-4">
          {/* Menu Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={onMenuToggle}
            aria-label="Toggle navigation menu"
            className="md:hidden"
          >
            <Menu className="h-5 w-5" />
          </Button>

          {/* Logo */}
          <div className="flex items-center space-x-2">
            <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
              <span className="text-primary-foreground font-bold text-sm">N</span>
            </div>
            <span className="hidden sm:inline-block font-bold text-lg">
              Nexus Architect
            </span>
          </div>
        </div>

        {/* Center Section - Search */}
        {showSearch && (
          <div className="flex-1 max-w-md mx-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="search"
                placeholder="Search..."
                className={cn(
                  "w-full pl-10 pr-4 py-2 rounded-md border border-input bg-background text-sm",
                  "focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent",
                  "transition-all duration-200",
                  searchFocused && "ring-2 ring-ring"
                )}
                onFocus={() => setSearchFocused(true)}
                onBlur={() => setSearchFocused(false)}
                aria-label="Search"
              />
            </div>
          </div>
        )}

        {/* Right Section */}
        <div className="flex items-center space-x-2">
          {/* Theme Toggle */}
          {showThemeToggle && (
            <div className="relative">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setThemeMenuOpen(!themeMenuOpen)}
                aria-label="Toggle theme"
                aria-expanded={themeMenuOpen}
                aria-haspopup="true"
              >
                {theme === 'light' && <Sun className="h-5 w-5" />}
                {theme === 'dark' && <Moon className="h-5 w-5" />}
                {theme === 'system' && <Monitor className="h-5 w-5" />}
              </Button>

              {themeMenuOpen && (
                <div className="absolute right-0 top-full mt-2 w-48 rounded-md border bg-popover p-1 shadow-md z-50">
                  {themeOptions.map(({ value, label, icon: Icon }) => (
                    <button
                      key={value}
                      onClick={() => {
                        setTheme(value)
                        setThemeMenuOpen(false)
                      }}
                      className={cn(
                        "flex w-full items-center space-x-2 rounded-sm px-2 py-1.5 text-sm",
                        "hover:bg-accent hover:text-accent-foreground",
                        "focus:bg-accent focus:text-accent-foreground focus:outline-none",
                        theme === value && "bg-accent text-accent-foreground"
                      )}
                    >
                      <Icon className="h-4 w-4" />
                      <span>{label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Notifications */}
          {showNotifications && (
            <Button
              variant="ghost"
              size="icon"
              className="relative"
              aria-label={`Notifications ${notifications > 0 ? `(${notifications} unread)` : ''}`}
            >
              <Bell className="h-5 w-5" />
              {notifications > 0 && (
                <span className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-destructive text-destructive-foreground text-xs flex items-center justify-center">
                  {notifications > 9 ? '9+' : notifications}
                </span>
              )}
            </Button>
          )}

          {/* User Menu */}
          {showUserMenu && (
            <div className="relative">
              <Button
                variant="ghost"
                onClick={() => setUserMenuOpen(!userMenuOpen)}
                className="flex items-center space-x-2 px-3"
                aria-label="User menu"
                aria-expanded={userMenuOpen}
                aria-haspopup="true"
              >
                <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center">
                  {user.avatar ? (
                    <img
                      src={user.avatar}
                      alt={user.name}
                      className="h-8 w-8 rounded-full object-cover"
                    />
                  ) : (
                    <User className="h-4 w-4 text-primary-foreground" />
                  )}
                </div>
                <div className="hidden md:block text-left">
                  <div className="text-sm font-medium">{user.name}</div>
                  <div className="text-xs text-muted-foreground">{user.email}</div>
                </div>
                <ChevronDown className="h-4 w-4" />
              </Button>

              {userMenuOpen && (
                <div className="absolute right-0 top-full mt-2 w-56 rounded-md border bg-popover p-1 shadow-md z-50">
                  <div className="px-2 py-1.5 text-sm font-medium border-b border-border mb-1">
                    <div>{user.name}</div>
                    <div className="text-xs text-muted-foreground">{user.email}</div>
                  </div>
                  
                  {userMenuItems.map((item, index) => {
                    if (item.type === 'divider') {
                      return <div key={index} className="my-1 border-t border-border" />
                    }

                    const Icon = item.icon
                    return (
                      <button
                        key={index}
                        onClick={() => {
                          if (item.action === 'logout') {
                            // Handle logout
                            console.log('Logout clicked')
                          } else if (item.href) {
                            // Handle navigation
                            console.log('Navigate to:', item.href)
                          }
                          setUserMenuOpen(false)
                        }}
                        className={cn(
                          "flex w-full items-center space-x-2 rounded-sm px-2 py-1.5 text-sm",
                          "hover:bg-accent hover:text-accent-foreground",
                          "focus:bg-accent focus:text-accent-foreground focus:outline-none",
                          item.variant === 'destructive' && "text-destructive hover:bg-destructive hover:text-destructive-foreground"
                        )}
                      >
                        <Icon className="h-4 w-4" />
                        <span>{item.label}</span>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Mobile Search */}
      {showSearch && (
        <div className="md:hidden border-t px-4 py-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="search"
              placeholder="Search..."
              className="w-full pl-10 pr-4 py-2 rounded-md border border-input bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
              aria-label="Search"
            />
          </div>
        </div>
      )}

      {/* Click outside handlers */}
      {(userMenuOpen || themeMenuOpen) && (
        <div
          className="fixed inset-0 z-30"
          onClick={() => {
            setUserMenuOpen(false)
            setThemeMenuOpen(false)
          }}
        />
      )}
    </header>
  )
})

Header.displayName = "Header"

export { Header }

