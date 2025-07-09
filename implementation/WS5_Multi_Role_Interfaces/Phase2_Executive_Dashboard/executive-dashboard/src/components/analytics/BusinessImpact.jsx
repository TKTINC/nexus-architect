import { useState } from 'react'
import { DollarSign, TrendingUp, AlertTriangle, Target, Calculator, Download } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import KPICard from '../kpi/KPICard'
import BarChart from '../charts/BarChart'
import TrendChart from '../charts/TrendChart'
import { 
  businessMetrics, 
  costAnalysis, 
  riskAssessment,
  developmentTrends 
} from '../../data/mockData'

export default function BusinessImpact() {
  const [selectedTimeframe, setSelectedTimeframe] = useState('quarterly')

  // ROI Calculation Data
  const roiData = {
    investment: 2.8, // Million
    returns: 6.86, // Million
    roi: 245, // Percentage
    paybackPeriod: 8, // Months
    npv: 4.06, // Million
    irr: 42 // Percentage
  }

  // Cost breakdown data
  const costBreakdown = [
    { category: 'Personnel', amount: 1.8, percentage: 64 },
    { category: 'Infrastructure', amount: 0.5, percentage: 18 },
    { category: 'Software Licenses', amount: 0.3, percentage: 11 },
    { category: 'Training', amount: 0.2, percentage: 7 }
  ]

  // Market positioning data
  const marketData = [
    { metric: 'Time to Market', us: 14, competitor: 21, industry: 18 },
    { metric: 'Development Cost', us: 2.8, competitor: 4.2, industry: 3.5 },
    { metric: 'Quality Score', us: 94, competitor: 87, industry: 89 },
    { metric: 'Customer Satisfaction', us: 4.7, competitor: 4.2, industry: 4.3 }
  ]

  const costBars = [
    { dataKey: 'current', name: 'Current Quarter' },
    { dataKey: 'previous', name: 'Previous Quarter' },
    { dataKey: 'budget', name: 'Budget' }
  ]

  const marketBars = [
    { dataKey: 'us', name: 'Our Performance' },
    { dataKey: 'competitor', name: 'Top Competitor' },
    { dataKey: 'industry', name: 'Industry Average' }
  ]

  const getRiskColor = (level) => {
    switch (level.toLowerCase()) {
      case 'high': return 'destructive'
      case 'medium': return 'default'
      case 'low': return 'secondary'
      default: return 'outline'
    }
  }

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'in progress': return 'bg-blue-100 text-blue-800'
      case 'planned': return 'bg-yellow-100 text-yellow-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Business Impact Analysis</h1>
          <p className="text-muted-foreground mt-1">
            ROI calculations, cost analysis, and strategic positioning
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Button variant="outline" size="sm">
            <Calculator className="w-4 h-4 mr-2" />
            ROI Calculator
          </Button>
          <Button size="sm">
            <Download className="w-4 h-4 mr-2" />
            Export Analysis
          </Button>
        </div>
      </div>

      <Tabs defaultValue="roi" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="roi">ROI Analysis</TabsTrigger>
          <TabsTrigger value="costs">Cost Analysis</TabsTrigger>
          <TabsTrigger value="risks">Risk Assessment</TabsTrigger>
          <TabsTrigger value="market">Market Position</TabsTrigger>
        </TabsList>

        {/* ROI Analysis Tab */}
        <TabsContent value="roi" className="space-y-6">
          {/* ROI Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <KPICard
              title="Total ROI"
              value={roiData.roi}
              unit="%"
              trend="up"
              description="Return on technology investments"
              className="border-green-200"
            />
            <KPICard
              title="Payback Period"
              value={roiData.paybackPeriod}
              unit=" months"
              trend="down"
              description="Time to recover initial investment"
            />
            <KPICard
              title="Net Present Value"
              value={roiData.npv}
              unit="M"
              trend="up"
              description="Present value of future cash flows"
            />
            <KPICard
              title="Internal Rate of Return"
              value={roiData.irr}
              unit="%"
              trend="up"
              description="Annualized effective compound return"
            />
          </div>

          {/* ROI Breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <DollarSign className="w-5 h-5" />
                  <span>Investment vs Returns</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-4 bg-red-50 rounded-lg">
                    <span className="font-medium">Total Investment</span>
                    <span className="text-xl font-bold text-red-600">
                      ${roiData.investment}M
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-green-50 rounded-lg">
                    <span className="font-medium">Total Returns</span>
                    <span className="text-xl font-bold text-green-600">
                      ${roiData.returns}M
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-blue-50 rounded-lg">
                    <span className="font-medium">Net Profit</span>
                    <span className="text-xl font-bold text-blue-600">
                      ${(roiData.returns - roiData.investment).toFixed(2)}M
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cost Breakdown</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {costBreakdown.map((item, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <span className="text-sm font-medium">{item.category}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-muted-foreground">
                          ${item.amount}M
                        </span>
                        <Badge variant="outline">{item.percentage}%</Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* ROI Trend */}
          <TrendChart
            title="ROI Trend Analysis"
            data={[
              { month: 'Q1 2023', roi: 180, investment: 2.2, returns: 3.96 },
              { month: 'Q2 2023', roi: 195, investment: 2.4, returns: 4.68 },
              { month: 'Q3 2023', roi: 210, investment: 2.6, returns: 5.46 },
              { month: 'Q4 2023', roi: 225, investment: 2.7, returns: 6.08 },
              { month: 'Q1 2024', roi: 245, investment: 2.8, returns: 6.86 }
            ]}
            lines={[
              { dataKey: 'roi', name: 'ROI %' }
            ]}
            height={300}
          />
        </TabsContent>

        {/* Cost Analysis Tab */}
        <TabsContent value="costs" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <BarChart
              title="Cost Analysis by Category"
              data={costAnalysis}
              bars={costBars}
              height={350}
            />
            
            <Card>
              <CardHeader>
                <CardTitle>Cost Optimization Opportunities</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 border border-border rounded-lg">
                    <h4 className="font-medium text-foreground">Infrastructure Optimization</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Potential savings: $150K annually through cloud optimization
                    </p>
                    <Badge className="mt-2" variant="default">High Impact</Badge>
                  </div>
                  <div className="p-4 border border-border rounded-lg">
                    <h4 className="font-medium text-foreground">Process Automation</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Reduce operational costs by $200K through automation
                    </p>
                    <Badge className="mt-2" variant="default">Medium Impact</Badge>
                  </div>
                  <div className="p-4 border border-border rounded-lg">
                    <h4 className="font-medium text-foreground">License Consolidation</h4>
                    <p className="text-sm text-muted-foreground mt-1">
                      Save $75K annually by consolidating software licenses
                    </p>
                    <Badge className="mt-2" variant="secondary">Low Impact</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Risk Assessment Tab */}
        <TabsContent value="risks" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5" />
                <span>Risk Assessment Matrix</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {riskAssessment.map((risk, index) => (
                  <div key={index} className="p-4 border border-border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-foreground">{risk.category}</h4>
                      <div className="flex items-center space-x-2">
                        <Badge variant={getRiskColor(risk.level)}>
                          {risk.level} Risk
                        </Badge>
                        <span className={`px-2 py-1 rounded text-xs ${getStatusColor(risk.status)}`}>
                          {risk.status}
                        </span>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Impact: </span>
                        <span className="font-medium">{risk.impact}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Probability: </span>
                        <span className="font-medium">{risk.probability}</span>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      <span className="font-medium">Mitigation: </span>
                      {risk.mitigation}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Market Position Tab */}
        <TabsContent value="market" className="space-y-6">
          <BarChart
            title="Competitive Analysis"
            data={marketData}
            bars={marketBars}
            height={400}
          />
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Market Leadership</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600 mb-2">
                    #2
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Market position in development efficiency
                  </p>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Competitive Advantage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600 mb-2">
                    33%
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Faster than industry average
                  </p>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Cost Efficiency</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-600 mb-2">
                    20%
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Lower costs than competitors
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

