import { useState, useEffect } from 'react'
import { Calendar, Download, RefreshCw, Filter } from 'lucide-react'
import { Button } from '../ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'
import KPICard from '../kpi/KPICard'
import TrendChart from '../charts/TrendChart'
import BarChart from '../charts/BarChart'
import { 
  executiveKPIs, 
  businessMetrics, 
  systemHealth, 
  developmentTrends,
  teamPerformance,
  executiveSummary 
} from '../../data/mockData'

export default function ExecutiveDashboard() {
  const [lastUpdated, setLastUpdated] = useState(new Date())
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleRefresh = async () => {
    setIsRefreshing(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    setLastUpdated(new Date())
    setIsRefreshing(false)
  }

  const trendLines = [
    { dataKey: 'velocity', name: 'Development Velocity' },
    { dataKey: 'quality', name: 'Code Quality' },
    { dataKey: 'deployment', name: 'Deployment Success' }
  ]

  const teamBars = [
    { dataKey: 'productivity', name: 'Productivity' },
    { dataKey: 'satisfaction', name: 'Satisfaction' },
    { dataKey: 'velocity', name: 'Velocity' }
  ]

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Executive Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Strategic insights and organizational performance overview
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Button variant="outline" size="sm">
            <Filter className="w-4 h-4 mr-2" />
            Filter
          </Button>
          <Button variant="outline" size="sm">
            <Calendar className="w-4 h-4 mr-2" />
            Q1 2024
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button size="sm">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Last updated indicator */}
      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>Last updated: {lastUpdated.toLocaleString()}</span>
        <Badge variant="outline" className="text-green-600 border-green-600">
          All systems operational
        </Badge>
      </div>

      {/* Executive Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <span>Executive Summary</span>
            <Badge variant="secondary">AI Generated</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3 text-foreground">Key Insights</h4>
              <ul className="space-y-2">
                {executiveSummary.keyInsights.map((insight, index) => (
                  <li key={index} className="text-sm text-muted-foreground flex items-start">
                    <span className="w-2 h-2 bg-primary rounded-full mt-2 mr-3 flex-shrink-0" />
                    {insight}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-3 text-foreground">Recommendations</h4>
              <ul className="space-y-2">
                {executiveSummary.recommendations.map((rec, index) => (
                  <li key={index} className="text-sm text-muted-foreground flex items-start">
                    <span className="w-2 h-2 bg-chart-2 rounded-full mt-2 mr-3 flex-shrink-0" />
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Strategic KPIs */}
      <div>
        <h2 className="text-xl font-semibold mb-4 text-foreground">Strategic KPIs</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Development Velocity"
            value={executiveKPIs.developmentVelocity.current}
            unit={executiveKPIs.developmentVelocity.unit}
            previousValue={executiveKPIs.developmentVelocity.previous}
            trend={executiveKPIs.developmentVelocity.trend}
            target={executiveKPIs.developmentVelocity.target}
            description="Sprint completion rate and feature delivery"
          />
          <KPICard
            title="Technical Debt"
            value={executiveKPIs.technicalDebt.current}
            unit={executiveKPIs.technicalDebt.unit}
            previousValue={executiveKPIs.technicalDebt.previous}
            trend={executiveKPIs.technicalDebt.trend}
            target={executiveKPIs.technicalDebt.target}
            description="Code quality and maintainability score"
          />
          <KPICard
            title="Security Posture"
            value={executiveKPIs.securityPosture.current}
            unit={executiveKPIs.securityPosture.unit}
            previousValue={executiveKPIs.securityPosture.previous}
            trend={executiveKPIs.securityPosture.trend}
            target={executiveKPIs.securityPosture.target}
            description="Compliance and vulnerability management"
          />
          <KPICard
            title="Team Productivity"
            value={executiveKPIs.teamProductivity.current}
            unit={executiveKPIs.teamProductivity.unit}
            previousValue={executiveKPIs.teamProductivity.previous}
            trend={executiveKPIs.teamProductivity.trend}
            target={executiveKPIs.teamProductivity.target}
            description="Overall team efficiency and output"
          />
        </div>
      </div>

      {/* Business Impact */}
      <div>
        <h2 className="text-xl font-semibold mb-4 text-foreground">Business Impact</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="ROI"
            value={businessMetrics.roi.current}
            unit={businessMetrics.roi.unit}
            previousValue={businessMetrics.roi.previous}
            trend={businessMetrics.roi.trend}
            target={businessMetrics.roi.target}
            description="Return on technology investments"
          />
          <KPICard
            title="Cost Savings"
            value={businessMetrics.costSavings.current}
            unit={`$${businessMetrics.costSavings.unit}`}
            previousValue={businessMetrics.costSavings.previous}
            trend={businessMetrics.costSavings.trend}
            target={businessMetrics.costSavings.target}
            description="Quarterly operational cost reduction"
          />
          <KPICard
            title="Time to Market"
            value={businessMetrics.timeToMarket.current}
            unit={businessMetrics.timeToMarket.unit}
            previousValue={businessMetrics.timeToMarket.previous}
            trend={businessMetrics.timeToMarket.trend}
            target={businessMetrics.timeToMarket.target}
            description="Average feature delivery time"
          />
          <KPICard
            title="Customer Satisfaction"
            value={businessMetrics.customerSatisfaction.current}
            unit={businessMetrics.customerSatisfaction.unit}
            previousValue={businessMetrics.customerSatisfaction.previous}
            trend={businessMetrics.customerSatisfaction.trend}
            target={businessMetrics.customerSatisfaction.target}
            description="User experience and product quality"
          />
        </div>
      </div>

      {/* System Health */}
      <div>
        <h2 className="text-xl font-semibold mb-4 text-foreground">System Health</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <KPICard
            title="Uptime"
            value={systemHealth.uptime.current}
            unit={systemHealth.uptime.unit}
            previousValue={systemHealth.uptime.previous}
            trend={systemHealth.uptime.trend}
            target={systemHealth.uptime.target}
            description="System availability and reliability"
          />
          <KPICard
            title="Response Time"
            value={systemHealth.responseTime.current}
            unit={systemHealth.responseTime.unit}
            previousValue={systemHealth.responseTime.previous}
            trend={systemHealth.responseTime.trend}
            target={systemHealth.responseTime.target}
            description="Average API response latency"
          />
          <KPICard
            title="Error Rate"
            value={systemHealth.errorRate.current}
            unit={systemHealth.errorRate.unit}
            previousValue={systemHealth.errorRate.previous}
            trend={systemHealth.errorRate.trend}
            target={systemHealth.errorRate.target}
            description="Application error frequency"
          />
          <KPICard
            title="Throughput"
            value={systemHealth.throughput.current}
            unit={systemHealth.throughput.unit}
            previousValue={systemHealth.throughput.previous}
            trend={systemHealth.throughput.trend}
            target={systemHealth.throughput.target}
            description="Request processing capacity"
          />
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TrendChart
          title="Development Trends"
          data={developmentTrends}
          lines={trendLines}
          height={350}
        />
        <BarChart
          title="Team Performance"
          data={teamPerformance}
          bars={teamBars}
          height={350}
        />
      </div>

      {/* Action Items */}
      <Card>
        <CardHeader>
          <CardTitle>Priority Action Items</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {executiveSummary.actionItems.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-4 border border-border rounded-lg">
                <div className="flex-1">
                  <h4 className="font-medium text-foreground">{item.item}</h4>
                  <p className="text-sm text-muted-foreground">
                    Owner: {item.owner} • Due: {new Date(item.dueDate).toLocaleDateString()}
                  </p>
                </div>
                <Badge 
                  variant={item.priority === 'Critical' ? 'destructive' : 
                          item.priority === 'High' ? 'default' : 'secondary'}
                >
                  {item.priority}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

