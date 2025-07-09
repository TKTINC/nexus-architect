import { useState, useEffect } from 'react'
import { 
  Wifi, 
  WifiOff, 
  RefreshCw, 
  CheckCircle, 
  AlertTriangle, 
  Clock,
  Database,
  Sync
} from 'lucide-react'
import { Card, CardContent } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Progress } from '../ui/progress'
import { 
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '../ui/collapsible'

const OfflineIndicator = ({ 
  isOnline, 
  syncStatus, 
  lastSync, 
  pendingChanges,
  syncProgress = 0,
  onForceSync
}) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showIndicator, setShowIndicator] = useState(true)

  // Auto-hide indicator when online and synced
  useEffect(() => {
    if (isOnline && syncStatus === 'idle' && pendingChanges.length === 0) {
      const timer = setTimeout(() => setShowIndicator(false), 5000)
      return () => clearTimeout(timer)
    } else {
      setShowIndicator(true)
    }
  }, [isOnline, syncStatus, pendingChanges])

  // Don't show if everything is normal and no pending changes
  if (!showIndicator && isOnline && syncStatus === 'idle' && pendingChanges.length === 0) {
    return null
  }

  const getStatusInfo = () => {
    if (!isOnline) {
      return {
        icon: WifiOff,
        color: 'text-red-500',
        bgColor: 'bg-red-50 border-red-200',
        title: 'Offline Mode',
        description: 'Working offline. Changes will sync when connected.',
        variant: 'destructive'
      }
    }

    switch (syncStatus) {
      case 'syncing':
        return {
          icon: RefreshCw,
          color: 'text-blue-500',
          bgColor: 'bg-blue-50 border-blue-200',
          title: 'Syncing...',
          description: `Syncing ${pendingChanges.length} changes`,
          variant: 'default',
          animated: true
        }
      case 'success':
        return {
          icon: CheckCircle,
          color: 'text-green-500',
          bgColor: 'bg-green-50 border-green-200',
          title: 'Synced',
          description: 'All changes synchronized',
          variant: 'success'
        }
      case 'error':
        return {
          icon: AlertTriangle,
          color: 'text-red-500',
          bgColor: 'bg-red-50 border-red-200',
          title: 'Sync Failed',
          description: 'Unable to sync changes. Tap to retry.',
          variant: 'destructive'
        }
      default:
        if (pendingChanges.length > 0) {
          return {
            icon: Clock,
            color: 'text-yellow-500',
            bgColor: 'bg-yellow-50 border-yellow-200',
            title: 'Pending Changes',
            description: `${pendingChanges.length} changes waiting to sync`,
            variant: 'warning'
          }
        }
        return {
          icon: Wifi,
          color: 'text-green-500',
          bgColor: 'bg-green-50 border-green-200',
          title: 'Online',
          description: 'Connected and synchronized',
          variant: 'success'
        }
    }
  }

  const statusInfo = getStatusInfo()
  const Icon = statusInfo.icon

  const formatLastSync = (date) => {
    if (!date) return 'Never'
    
    const now = new Date()
    const diff = now - new Date(date)
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)

    if (days > 0) return `${days}d ago`
    if (hours > 0) return `${hours}h ago`
    if (minutes > 0) return `${minutes}m ago`
    return 'Just now'
  }

  const handleIndicatorClick = () => {
    if (syncStatus === 'error' || (isOnline && pendingChanges.length > 0)) {
      onForceSync?.()
    } else {
      setIsExpanded(!isExpanded)
    }
  }

  return (
    <div className="fixed top-16 left-4 right-4 z-40">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <Card className={`${statusInfo.bgColor} border transition-all duration-300`}>
          <CollapsibleTrigger asChild>
            <CardContent 
              className="p-3 cursor-pointer"
              onClick={handleIndicatorClick}
            >
              <div className="flex items-center space-x-3">
                <Icon 
                  className={`h-5 w-5 ${statusInfo.color} ${
                    statusInfo.animated ? 'animate-spin' : ''
                  }`} 
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-sm">{statusInfo.title}</p>
                    <Badge variant={statusInfo.variant} className="text-xs">
                      {isOnline ? 'Online' : 'Offline'}
                    </Badge>
                  </div>
                  <p className="text-xs text-muted-foreground truncate">
                    {statusInfo.description}
                  </p>
                </div>
              </div>

              {/* Sync Progress Bar */}
              {syncStatus === 'syncing' && (
                <div className="mt-2">
                  <Progress value={syncProgress} className="h-1" />
                  <p className="text-xs text-muted-foreground mt-1">
                    {Math.round(syncProgress)}% complete
                  </p>
                </div>
              )}
            </CardContent>
          </CollapsibleTrigger>

          <CollapsibleContent>
            <CardContent className="pt-0 pb-3 px-3">
              <div className="space-y-3 border-t pt-3">
                {/* Detailed Status */}
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <Database className="h-3 w-3" />
                      <span className="font-medium">Local Data</span>
                    </div>
                    <p className="text-muted-foreground">
                      {pendingChanges.length} pending changes
                    </p>
                  </div>
                  
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <Sync className="h-3 w-3" />
                      <span className="font-medium">Last Sync</span>
                    </div>
                    <p className="text-muted-foreground">
                      {formatLastSync(lastSync)}
                    </p>
                  </div>
                </div>

                {/* Pending Changes List */}
                {pendingChanges.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-medium">Pending Changes:</p>
                    <div className="space-y-1 max-h-20 overflow-y-auto">
                      {pendingChanges.slice(0, 3).map((change, index) => (
                        <div key={index} className="flex items-center justify-between text-xs">
                          <span className="truncate">
                            {change.type}: {change.data?.title || change.data?.name || 'Untitled'}
                          </span>
                          <Badge variant="outline" className="text-xs">
                            {change.operation}
                          </Badge>
                        </div>
                      ))}
                      {pendingChanges.length > 3 && (
                        <p className="text-xs text-muted-foreground">
                          +{pendingChanges.length - 3} more changes
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex space-x-2">
                  {isOnline && pendingChanges.length > 0 && (
                    <Button 
                      size="sm" 
                      variant="outline" 
                      className="flex-1 h-8 text-xs"
                      onClick={onForceSync}
                      disabled={syncStatus === 'syncing'}
                    >
                      <RefreshCw className="h-3 w-3 mr-1" />
                      Sync Now
                    </Button>
                  )}
                  
                  <Button 
                    size="sm" 
                    variant="ghost" 
                    className="h-8 text-xs"
                    onClick={() => setShowIndicator(false)}
                  >
                    Dismiss
                  </Button>
                </div>
              </div>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  )
}

export default OfflineIndicator

