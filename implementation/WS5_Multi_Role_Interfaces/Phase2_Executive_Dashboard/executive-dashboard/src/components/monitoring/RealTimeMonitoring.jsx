import { useState, useEffect } from 'react'
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Server, 
  Zap,
  Bell,
  Settings,
  RefreshCw
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Progress } from '../ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import KPICard from '../kpi/KPICard'
import TrendChart from '../charts/TrendChart'
import { 
  systemHealth, 
  incidentData, 
  capacityPlanning 
} from '../../data/mockData'

export default function RealTimeMonitoring() {
  const [isLive, setIsLive] = useState(true)
  const [lastUpdate, setLastUpdate] = useState(new Date())
  const [alerts, setAlerts] = useState([
    {
      id: 1,
      severity: 'warning',
      title: 'High CPU Usage',
      description: 'Production server CPU usage above 80%',
      timestamp: new Date(Date.now() - 300000), // 5 minutes ago
      status: 'active'
    },
    {
      id: 2,
      severity: 'info',
      title: 'Deployment Completed',
      description: 'Version 2.1.4 deployed successfully',
      timestamp: new Date(Date.now() - 900000), // 15 minutes ago
      status: 'resolved'
    },
    {
      id: 3,
      severity: 'critical',
      title: 'Database Connection Pool',
      description: 'Connection pool utilization at 95%',
      timestamp: new Date(Date.now() - 1800000), // 30 minutes ago
      status: 'investigating'
    }
  ])

  // Real-time system metrics
  const [metrics, setMetrics] = useState({
    activeUsers: 1247,
    requestsPerMinute: 12500,
    responseTime: 145,
    errorRate: 0.03,
    uptime: 99.97
  })

  // Simulate real-time updates
  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(() => {
      setMetrics(prev => ({
        activeUsers: prev.activeUsers + Math.floor(Math.random() * 20 - 10),
        requestsPerMinute: prev.requestsPerMinute + Math.floor(Math.random() * 200 - 100),
        responseTime: Math.max(100, prev.responseTime + Math.floor(Math.random() * 20 - 10)),
        errorRate: Math.max(0, prev.errorRate + (Math.random() * 0.02 - 0.01)),
        uptime: prev.uptime
      }))
      setLastUpdate(new Date())
    }, 5000)

    return () => clearInterval(interval)
  }, [isLive])

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical': return 'destructive'
      case 'warning': return 'default'
      case 'info': return 'secondary'
      default: return 'outline'
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'bg-red-100 text-red-800'
      case 'investigating': return 'bg-yellow-100 text-yellow-800'
      case 'resolved': return 'bg-green-100 text-green-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const incidentTrend = incidentData.map(item => ({
    date: new Date(item.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    critical: item.critical,
    high: item.high,
    medium: item.medium,
    low: item.low,
    total: item.critical + item.high + item.medium + item.low
  }))

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Real-Time Monitoring</h1>
          <p className="text-muted-foreground mt-1">
            System health, incidents, and capacity monitoring
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-500' : 'bg-gray-400'}`} />
            <span className="text-sm text-muted-foreground">
              {isLive ? 'Live' : 'Paused'}
            </span>
          </div>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setIsLive(!isLive)}
          >
            {isLive ? 'Pause' : 'Resume'}
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="w-4 h-4 mr-2" />
            Configure
          </Button>
        </div>
      </div>

      {/* Live Status */}
      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>Last updated: {lastUpdate.toLocaleTimeString()}</span>
        <div className="flex items-center space-x-4">
          <Badge variant="outline" className="text-green-600 border-green-600">
            <CheckCircle className="w-3 h-3 mr-1" />
            All systems operational
          </Badge>
          <span>{alerts.filter(a => a.status === 'active').length} active alerts</span>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">System Overview</TabsTrigger>
          <TabsTrigger value="incidents">Incidents</TabsTrigger>
          <TabsTrigger value="capacity">Capacity</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        {/* System Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Real-time Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            <KPICard
              title="Active Users"
              value={metrics.activeUsers}
              trend="up"
              description="Currently online users"
            />
            <KPICard
              title="Requests/Min"
              value={metrics.requestsPerMinute}
              trend="up"
              description="API requests per minute"
            />
            <KPICard
              title="Response Time"
              value={metrics.responseTime}
              unit="ms"
              trend="down"
              target={150}
              description="Average response latency"
            />
            <KPICard
              title="Error Rate"
              value={metrics.errorRate.toFixed(3)}
              unit="%"
              trend="down"
              target={0.02}
              description="Application error frequency"
            />
            <KPICard
              title="Uptime"
              value={metrics.uptime}
              unit="%"
              trend="up"
              target={99.95}
              description="System availability"
            />
          </div>

          {/* System Health Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Web Servers</CardTitle>
                <Server className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">8/8</div>
                <p className="text-xs text-muted-foreground">All servers healthy</p>
                <div className="mt-2">
                  <Badge variant="outline" className="text-green-600 border-green-600">
                    Healthy
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Databases</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">3/3</div>
                <p className="text-xs text-muted-foreground">Primary and replicas</p>
                <div className="mt-2">
                  <Badge variant="outline" className="text-green-600 border-green-600">
                    Healthy
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Load Balancers</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-yellow-600">2/3</div>
                <p className="text-xs text-muted-foreground">One under maintenance</p>
                <div className="mt-2">
                  <Badge variant="outline" className="text-yellow-600 border-yellow-600">
                    Degraded
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">CDN</CardTitle>
                <RefreshCw className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">12/12</div>
                <p className="text-xs text-muted-foreground">Global edge locations</p>
                <div className="mt-2">
                  <Badge variant="outline" className="text-green-600 border-green-600">
                    Healthy
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Performance Trends */}
          <TrendChart
            title="Performance Trends (Last 7 Days)"
            data={incidentTrend}
            lines={[
              { dataKey: 'total', name: 'Total Incidents' }
            ]}
            height={300}
          />
        </TabsContent>

        {/* Incidents Tab */}
        <TabsContent value="incidents" className="space-y-6">
          <TrendChart
            title="Incident Trends by Severity"
            data={incidentTrend}
            lines={[
              { dataKey: 'critical', name: 'Critical' },
              { dataKey: 'high', name: 'High' },
              { dataKey: 'medium', name: 'Medium' },
              { dataKey: 'low', name: 'Low' }
            ]}
            height={350}
          />

          <Card>
            <CardHeader>
              <CardTitle>Recent Incidents</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.map((alert) => (
                  <div key={alert.id} className="flex items-start space-x-4 p-4 border border-border rounded-lg">
                    <div className="flex-shrink-0">
                      {alert.severity === 'critical' && (
                        <AlertTriangle className="w-5 h-5 text-red-500" />
                      )}
                      {alert.severity === 'warning' && (
                        <AlertTriangle className="w-5 h-5 text-yellow-500" />
                      )}
                      {alert.severity === 'info' && (
                        <CheckCircle className="w-5 h-5 text-blue-500" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-foreground">{alert.title}</h4>
                        <div className="flex items-center space-x-2">
                          <Badge variant={getSeverityColor(alert.severity)}>
                            {alert.severity}
                          </Badge>
                          <span className={`px-2 py-1 rounded text-xs ${getStatusColor(alert.status)}`}>
                            {alert.status}
                          </span>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        {alert.description}
                      </p>
                      <div className="flex items-center space-x-4 mt-2 text-xs text-muted-foreground">
                        <span className="flex items-center">
                          <Clock className="w-3 h-3 mr-1" />
                          {alert.timestamp.toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Capacity Tab */}
        <TabsContent value="capacity" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {Object.entries(capacityPlanning).map(([resource, data]) => (
              <Card key={resource}>
                <CardHeader>
                  <CardTitle className="capitalize">{resource} Utilization</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Current Usage</span>
                        <span>{data.current}%</span>
                      </div>
                      <Progress 
                        value={data.current} 
                        className={`h-2 ${data.current > data.alert ? 'bg-red-100' : 'bg-green-100'}`}
                      />
                    </div>
                    
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Projected (30 days)</span>
                        <span>{data.projected}%</span>
                      </div>
                      <Progress 
                        value={data.projected} 
                        className={`h-2 ${data.projected > data.alert ? 'bg-red-100' : 'bg-yellow-100'}`}
                      />
                    </div>

                    <div className="flex justify-between items-center text-xs text-muted-foreground">
                      <span>Alert threshold: {data.alert}%</span>
                      <Badge 
                        variant={data.current > data.alert ? 'destructive' : 'secondary'}
                      >
                        {data.current > data.alert ? 'Action Required' : 'Normal'}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Capacity Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border border-border rounded-lg">
                  <h4 className="font-medium text-foreground">Memory Scaling Required</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Current memory utilization at 72% with projected growth to 84%. 
                    Consider adding 2 additional servers within 2 weeks.
                  </p>
                  <Badge className="mt-2" variant="destructive">High Priority</Badge>
                </div>
                <div className="p-4 border border-border rounded-lg">
                  <h4 className="font-medium text-foreground">Storage Optimization</h4>
                  <p className="text-sm text-muted-foreground mt-1">
                    Storage growth trending upward. Implement data archiving strategy 
                    to maintain current capacity levels.
                  </p>
                  <Badge className="mt-2" variant="default">Medium Priority</Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Active Alerts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-red-600 mb-2">
                    {alerts.filter(a => a.status === 'active').length}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Require immediate attention
                  </p>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Under Investigation</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-yellow-600 mb-2">
                    {alerts.filter(a => a.status === 'investigating').length}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Being actively investigated
                  </p>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Resolved Today</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600 mb-2">
                    {alerts.filter(a => a.status === 'resolved').length}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Successfully resolved
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Bell className="w-5 h-5" />
                <span>Alert Configuration</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border border-border rounded-lg">
                    <h4 className="font-medium text-foreground">Response Time</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Alert when average response time exceeds 200ms for 5 minutes
                    </p>
                    <Badge className="mt-2" variant="outline">Enabled</Badge>
                  </div>
                  <div className="p-4 border border-border rounded-lg">
                    <h4 className="font-medium text-foreground">Error Rate</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Alert when error rate exceeds 0.1% for 3 minutes
                    </p>
                    <Badge className="mt-2" variant="outline">Enabled</Badge>
                  </div>
                  <div className="p-4 border border-border rounded-lg">
                    <h4 className="font-medium text-foreground">CPU Usage</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Alert when CPU usage exceeds 80% for 10 minutes
                    </p>
                    <Badge className="mt-2" variant="outline">Enabled</Badge>
                  </div>
                  <div className="p-4 border border-border rounded-lg">
                    <h4 className="font-medium text-foreground">Memory Usage</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Alert when memory usage exceeds 90% for 5 minutes
                    </p>
                    <Badge className="mt-2" variant="outline">Enabled</Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

