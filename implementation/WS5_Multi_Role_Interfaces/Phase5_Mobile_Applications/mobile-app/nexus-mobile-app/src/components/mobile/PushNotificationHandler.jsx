import { useState, useEffect } from 'react'
import { 
  Bell, 
  X, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  MessageSquare,
  Calendar,
  Users,
  FileText
} from 'lucide-react'
import { Card, CardContent } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar'
import { 
  Toast,
  ToastAction,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from '../ui/toast'
import { useToast } from '../ui/use-toast'

const PushNotificationHandler = ({ notifications, onNotificationReceived }) => {
  const { toast } = useToast()
  const [activeNotifications, setActiveNotifications] = useState([])
  const [notificationQueue, setNotificationQueue] = useState([])

  // Initialize service worker for push notifications
  useEffect(() => {
    if ('serviceWorker' in navigator && 'PushManager' in window) {
      initializePushNotifications()
    }
  }, [])

  // Handle incoming notifications
  useEffect(() => {
    if (notifications.length > 0) {
      const newNotifications = notifications.filter(
        notif => !activeNotifications.find(active => active.id === notif.id)
      )
      
      newNotifications.forEach(notification => {
        handleIncomingNotification(notification)
      })
    }
  }, [notifications, activeNotifications])

  const initializePushNotifications = async () => {
    try {
      // Register service worker
      const registration = await navigator.serviceWorker.register('/sw.js')
      
      // Listen for push events
      navigator.serviceWorker.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'PUSH_NOTIFICATION') {
          handleIncomingNotification(event.data.notification)
        }
      })

      console.log('Push notifications initialized')
    } catch (error) {
      console.error('Failed to initialize push notifications:', error)
    }
  }

  const handleIncomingNotification = (notification) => {
    const processedNotification = {
      ...notification,
      id: notification.id || Date.now() + Math.random(),
      timestamp: notification.timestamp || new Date().toISOString(),
      read: false,
      dismissed: false
    }

    // Add to active notifications
    setActiveNotifications(prev => [processedNotification, ...prev])

    // Show toast notification
    showToastNotification(processedNotification)

    // Play notification sound (if enabled)
    playNotificationSound(processedNotification.priority)

    // Vibrate device (if supported)
    vibrateForNotification(processedNotification.priority)

    // Update parent component
    onNotificationReceived?.(prev => [processedNotification, ...prev])
  }

  const showToastNotification = (notification) => {
    const icon = getNotificationIcon(notification.type)
    const variant = getNotificationVariant(notification.priority)

    toast({
      title: notification.title,
      description: notification.body,
      variant,
      action: notification.actionUrl ? (
        <ToastAction 
          altText="View" 
          onClick={() => handleNotificationAction(notification)}
        >
          View
        </ToastAction>
      ) : undefined,
      duration: getToastDuration(notification.priority)
    })
  }

  const getNotificationIcon = (type) => {
    const icons = {
      message: MessageSquare,
      task: CheckCircle,
      meeting: Calendar,
      team: Users,
      document: FileText,
      alert: AlertTriangle,
      info: Info,
      default: Bell
    }
    return icons[type] || icons.default
  }

  const getNotificationVariant = (priority) => {
    switch (priority) {
      case 'high':
      case 'urgent':
        return 'destructive'
      case 'medium':
        return 'default'
      case 'low':
        return 'secondary'
      default:
        return 'default'
    }
  }

  const getToastDuration = (priority) => {
    switch (priority) {
      case 'urgent':
        return 10000 // 10 seconds
      case 'high':
        return 7000  // 7 seconds
      case 'medium':
        return 5000  // 5 seconds
      case 'low':
        return 3000  // 3 seconds
      default:
        return 5000
    }
  }

  const playNotificationSound = (priority) => {
    try {
      const audio = new Audio()
      
      switch (priority) {
        case 'urgent':
          audio.src = '/sounds/urgent.mp3'
          break
        case 'high':
          audio.src = '/sounds/high.mp3'
          break
        default:
          audio.src = '/sounds/default.mp3'
      }
      
      audio.volume = 0.5
      audio.play().catch(error => {
        console.log('Could not play notification sound:', error)
      })
    } catch (error) {
      console.log('Notification sound not available:', error)
    }
  }

  const vibrateForNotification = (priority) => {
    if ('vibrate' in navigator) {
      const patterns = {
        urgent: [200, 100, 200, 100, 200],
        high: [100, 50, 100],
        medium: [100],
        low: [50]
      }
      
      navigator.vibrate(patterns[priority] || patterns.medium)
    }
  }

  const handleNotificationAction = (notification) => {
    if (notification.actionUrl) {
      window.open(notification.actionUrl, '_blank')
    }
    
    markAsRead(notification.id)
  }

  const markAsRead = (notificationId) => {
    setActiveNotifications(prev =>
      prev.map(notif =>
        notif.id === notificationId
          ? { ...notif, read: true }
          : notif
      )
    )
  }

  const dismissNotification = (notificationId) => {
    setActiveNotifications(prev =>
      prev.filter(notif => notif.id !== notificationId)
    )
  }

  const clearAllNotifications = () => {
    setActiveNotifications([])
  }

  // Notification permission status
  const [permissionStatus, setPermissionStatus] = useState(
    'Notification' in window ? Notification.permission : 'unsupported'
  )

  const requestPermission = async () => {
    if ('Notification' in window) {
      const permission = await Notification.requestPermission()
      setPermissionStatus(permission)
      return permission
    }
    return 'unsupported'
  }

  // In-app notification overlay for urgent notifications
  const UrgentNotificationOverlay = ({ notification, onDismiss }) => {
    const Icon = getNotificationIcon(notification.type)
    
    return (
      <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
        <Card className="w-full max-w-sm bg-red-50 border-red-200">
          <CardContent className="p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                  <Icon className="h-5 w-5 text-red-600" />
                </div>
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-2">
                  <Badge variant="destructive" className="text-xs">
                    URGENT
                  </Badge>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="p-1 h-auto"
                    onClick={onDismiss}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                
                <h3 className="font-semibold text-sm mb-1">
                  {notification.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-3">
                  {notification.body}
                </p>
                
                <div className="flex space-x-2">
                  {notification.actionUrl && (
                    <Button 
                      size="sm" 
                      className="flex-1"
                      onClick={() => handleNotificationAction(notification)}
                    >
                      View Details
                    </Button>
                  )}
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={onDismiss}
                  >
                    Dismiss
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Show urgent notification overlay
  const urgentNotification = activeNotifications.find(
    notif => notif.priority === 'urgent' && !notif.dismissed
  )

  return (
    <>
      <ToastProvider>
        <ToastViewport />
      </ToastProvider>

      {/* Urgent Notification Overlay */}
      {urgentNotification && (
        <UrgentNotificationOverlay
          notification={urgentNotification}
          onDismiss={() => dismissNotification(urgentNotification.id)}
        />
      )}

      {/* Notification Permission Request */}
      {permissionStatus === 'default' && (
        <div className="fixed top-20 left-4 right-4 z-40">
          <Card className="bg-blue-50 border-blue-200">
            <CardContent className="p-3">
              <div className="flex items-center space-x-3">
                <Bell className="h-5 w-5 text-blue-600" />
                <div className="flex-1">
                  <p className="font-medium text-sm">Enable Notifications</p>
                  <p className="text-xs text-muted-foreground">
                    Stay updated with important project updates
                  </p>
                </div>
                <div className="flex space-x-2">
                  <Button size="sm" onClick={requestPermission}>
                    Enable
                  </Button>
                  <Button 
                    size="sm" 
                    variant="ghost"
                    onClick={() => setPermissionStatus('denied')}
                  >
                    Later
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </>
  )
}

// Mock notification data for testing
export const mockNotifications = [
  {
    id: 1,
    type: 'task',
    title: 'Task Due Soon',
    body: 'UI Design Review is due in 30 minutes',
    priority: 'high',
    actionUrl: '/tasks/123',
    timestamp: new Date().toISOString()
  },
  {
    id: 2,
    type: 'message',
    title: 'New Message',
    body: 'Alice Johnson commented on your task',
    priority: 'medium',
    actionUrl: '/messages/456',
    timestamp: new Date().toISOString()
  },
  {
    id: 3,
    type: 'meeting',
    title: 'Meeting Starting',
    body: 'Sprint Planning meeting starts in 5 minutes',
    priority: 'urgent',
    actionUrl: '/meetings/789',
    timestamp: new Date().toISOString()
  }
]

export default PushNotificationHandler

