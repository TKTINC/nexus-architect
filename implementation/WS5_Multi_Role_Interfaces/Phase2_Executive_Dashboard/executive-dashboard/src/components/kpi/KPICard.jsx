import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'

export default function KPICard({ 
  title, 
  value, 
  unit = '', 
  previousValue, 
  trend, 
  target, 
  description,
  className = '' 
}) {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-green-500" />
      case 'down':
        return <TrendingDown className="w-4 h-4 text-red-500" />
      default:
        return <Minus className="w-4 h-4 text-gray-500" />
    }
  }

  const getTrendColor = () => {
    switch (trend) {
      case 'up':
        return 'text-green-600'
      case 'down':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  const getTargetStatus = () => {
    if (!target) return null
    
    const isOnTarget = trend === 'up' ? value >= target : value <= target
    return (
      <Badge variant={isOnTarget ? 'default' : 'secondary'} className="text-xs">
        Target: {target}{unit}
      </Badge>
    )
  }

  const calculateChange = () => {
    if (!previousValue) return null
    const change = ((value - previousValue) / previousValue * 100).toFixed(1)
    return Math.abs(change)
  }

  return (
    <Card className={`transition-all duration-200 hover:shadow-lg ${className}`}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        {getTrendIcon()}
      </CardHeader>
      <CardContent>
        <div className="flex items-baseline space-x-2">
          <div className="text-2xl font-bold text-foreground">
            {typeof value === 'number' ? value.toLocaleString() : value}
            <span className="text-sm font-normal text-muted-foreground ml-1">
              {unit}
            </span>
          </div>
        </div>
        
        <div className="flex items-center justify-between mt-2">
          {previousValue && (
            <div className={`flex items-center space-x-1 text-xs ${getTrendColor()}`}>
              <span>
                {trend === 'up' ? '+' : trend === 'down' ? '-' : ''}
                {calculateChange()}%
              </span>
              <span className="text-muted-foreground">vs last period</span>
            </div>
          )}
          
          {getTargetStatus()}
        </div>
        
        {description && (
          <p className="text-xs text-muted-foreground mt-2">
            {description}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

