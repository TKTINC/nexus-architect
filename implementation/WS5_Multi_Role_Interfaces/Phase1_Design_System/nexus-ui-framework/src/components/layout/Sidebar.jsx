import React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '../atoms/Button'
import { 
  Home, 
  BarChart3, 
  Users, 
  Settings, 
  FileText, 
  Database,
  Shield,
  Zap,
  ChevronDown,
  ChevronRight,
  X,
  Bookmark,
  Star
} from 'lucide-react'
import { useLocation } from 'react-router-dom'

const Sidebar = React.forwardRef(({
  className,
  isOpen = true,
  onClose,
  userRole = 'admin', // admin, developer, manager, viewer
  ...props
}, ref) => {
  const location = useLocation()
  const [expandedSections, setExpandedSections] = React.useState(['main'])
  const [favorites, setFavorites] = React.useState(['/', '/components'])

  // Role-based navigation configuration
  const navigationConfig = {
    admin: {
      sections: [
        {
          id: 'main',
          label: 'Main',
          items: [
            { id: 'dashboard', label: 'Dashboard', icon: Home, href: '/', badge: null },
            { id: 'analytics', label: 'Analytics', icon: BarChart3, href: '/analytics', badge: 'New' },
            { id: 'users', label: 'Users', icon: Users, href: '/users', badge: null },
            { id: 'settings', label: 'Settings', icon: Settings, href: '/settings', badge: null }
          ]
        },
        {
          id: 'development',
          label: 'Development',
          items: [
            { id: 'components', label: 'Components', icon: FileText, href: '/components', badge: null },
            { id: 'database', label: 'Database', icon: Database, href: '/database', badge: null },
            { id: 'security', label: 'Security', icon: Shield, href: '/security', badge: '3' }
          ]
        },
        {
          id: 'tools',
          label: 'Tools',
          items: [
            { id: 'automation', label: 'Automation', icon: Zap, href: '/automation', badge: null },
            { id: 'accessibility', label: 'Accessibility', icon: Shield, href: '/accessibility', badge: null }
          ]
        }
      ]
    },
    developer: {
      sections: [
        {
          id: 'main',
          label: 'Main',
          items: [
            { id: 'dashboard', label: 'Dashboard', icon: Home, href: '/', badge: null },
            { id: 'components', label: 'Components', icon: FileText, href: '/components', badge: null },
            { id: 'database', label: 'Database', icon: Database, href: '/database', badge: null }
          ]
        },
        {
          id: 'tools',
          label: 'Tools',
          items: [
            { id: 'automation', label: 'Automation', icon: Zap, href: '/automation', badge: null },
            { id: 'accessibility', label: 'Accessibility', icon: Shield, href: '/accessibility', badge: null }
          ]
        }
      ]
    },
    manager: {
      sections: [
        {
          id: 'main',
          label: 'Main',
          items: [
            { id: 'dashboard', label: 'Dashboard', icon: Home, href: '/', badge: null },
            { id: 'analytics', label: 'Analytics', icon: BarChart3, href: '/analytics', badge: null },
            { id: 'users', label: 'Users', icon: Users, href: '/users', badge: null }
          ]
        }
      ]
    },
    viewer: {
      sections: [
        {
          id: 'main',
          label: 'Main',
          items: [
            { id: 'dashboard', label: 'Dashboard', icon: Home, href: '/', badge: null },
            { id: 'components', label: 'Components', icon: FileText, href: '/components', badge: null }
          ]
        }
      ]
    }
  }

  const navigation = navigationConfig[userRole] || navigationConfig.viewer

  const toggleSection = (sectionId) => {
    setExpandedSections(prev => 
      prev.includes(sectionId) 
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    )
  }

  const toggleFavorite = (href) => {
    setFavorites(prev => 
      prev.includes(href)
        ? prev.filter(f => f !== href)
        : [...prev, href]
    )
  }

  const isActive = (href) => {
    return location.pathname === href
  }

  const isFavorite = (href) => {
    return favorites.includes(href)
  }

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        ref={ref}
        className={cn(
          "fixed left-0 top-16 z-50 h-[calc(100vh-4rem)] w-64 transform border-r bg-background transition-transform duration-300 ease-in-out md:relative md:top-0 md:h-[calc(100vh-4rem)] md:translate-x-0",
          isOpen ? "translate-x-0" : "-translate-x-full md:-translate-x-48",
          className
        )}
        {...props}
      >
        <div className="flex h-full flex-col">
          {/* Mobile Close Button */}
          <div className="flex items-center justify-between p-4 md:hidden">
            <span className="font-semibold">Navigation</span>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              aria-label="Close navigation"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>

          {/* Navigation Content */}
          <nav className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* Favorites Section */}
            {favorites.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center space-x-2 px-2">
                  <Star className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium text-muted-foreground">Favorites</span>
                </div>
                <div className="space-y-1">
                  {navigation.sections.flatMap(section => section.items)
                    .filter(item => favorites.includes(item.href))
                    .map(item => {
                      const Icon = item.icon
                      return (
                        <a
                          key={`fav-${item.id}`}
                          href={item.href}
                          className={cn(
                            "flex items-center space-x-3 rounded-lg px-3 py-2 text-sm transition-colors",
                            "hover:bg-accent hover:text-accent-foreground",
                            "focus:bg-accent focus:text-accent-foreground focus:outline-none",
                            isActive(item.href) && "bg-accent text-accent-foreground"
                          )}
                        >
                          <Icon className="h-4 w-4" />
                          <span className="flex-1">{item.label}</span>
                        </a>
                      )
                    })}
                </div>
              </div>
            )}

            {/* Main Navigation Sections */}
            {navigation.sections.map(section => (
              <div key={section.id} className="space-y-2">
                {/* Section Header */}
                <button
                  onClick={() => toggleSection(section.id)}
                  className="flex w-full items-center justify-between px-2 py-1 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
                  aria-expanded={expandedSections.includes(section.id)}
                >
                  <span>{section.label}</span>
                  {expandedSections.includes(section.id) ? (
                    <ChevronDown className="h-4 w-4" />
                  ) : (
                    <ChevronRight className="h-4 w-4" />
                  )}
                </button>

                {/* Section Items */}
                {expandedSections.includes(section.id) && (
                  <div className="space-y-1">
                    {section.items.map(item => {
                      const Icon = item.icon
                      const active = isActive(item.href)
                      const favorite = isFavorite(item.href)

                      return (
                        <div key={item.id} className="group relative">
                          <a
                            href={item.href}
                            className={cn(
                              "flex items-center space-x-3 rounded-lg px-3 py-2 text-sm transition-colors",
                              "hover:bg-accent hover:text-accent-foreground",
                              "focus:bg-accent focus:text-accent-foreground focus:outline-none",
                              active && "bg-accent text-accent-foreground"
                            )}
                          >
                            <Icon className="h-4 w-4" />
                            <span className="flex-1">{item.label}</span>
                            
                            {/* Badge */}
                            {item.badge && (
                              <span className={cn(
                                "rounded-full px-2 py-0.5 text-xs font-medium",
                                item.badge === 'New' && "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
                                /^\d+$/.test(item.badge) && "bg-destructive text-destructive-foreground"
                              )}>
                                {item.badge}
                              </span>
                            )}
                          </a>

                          {/* Favorite Toggle */}
                          <button
                            onClick={(e) => {
                              e.preventDefault()
                              toggleFavorite(item.href)
                            }}
                            className={cn(
                              "absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity",
                              "hover:bg-accent-foreground/10",
                              favorite && "opacity-100"
                            )}
                            aria-label={favorite ? 'Remove from favorites' : 'Add to favorites'}
                          >
                            {favorite ? (
                              <Star className="h-3 w-3 fill-current text-yellow-500" />
                            ) : (
                              <Bookmark className="h-3 w-3" />
                            )}
                          </button>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            ))}
          </nav>

          {/* Sidebar Footer */}
          <div className="border-t p-4">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Role: {userRole}</span>
              <span className="capitalize">{isOpen ? 'Expanded' : 'Collapsed'}</span>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
})

Sidebar.displayName = "Sidebar"

// Sidebar context for managing state
const SidebarContext = React.createContext({
  isOpen: true,
  toggle: () => {},
  close: () => {},
  open: () => {}
})

const useSidebar = () => {
  const context = React.useContext(SidebarContext)
  if (!context) {
    throw new Error('useSidebar must be used within a SidebarProvider')
  }
  return context
}

const SidebarProvider = ({ children, defaultOpen = true }) => {
  const [isOpen, setIsOpen] = React.useState(defaultOpen)

  const toggle = React.useCallback(() => {
    setIsOpen(prev => !prev)
  }, [])

  const close = React.useCallback(() => {
    setIsOpen(false)
  }, [])

  const open = React.useCallback(() => {
    setIsOpen(true)
  }, [])

  const value = React.useMemo(() => ({
    isOpen,
    toggle,
    close,
    open
  }), [isOpen, toggle, close, open])

  return (
    <SidebarContext.Provider value={value}>
      {children}
    </SidebarContext.Provider>
  )
}

export { Sidebar, SidebarProvider, useSidebar }

