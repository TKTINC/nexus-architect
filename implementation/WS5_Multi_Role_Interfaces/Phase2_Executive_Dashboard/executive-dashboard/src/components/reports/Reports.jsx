import { useState } from 'react'
import { 
  FileText, 
  Download, 
  Calendar, 
  Share2, 
  Mail, 
  Settings,
  Clock,
  Users,
  Eye,
  Edit,
  Trash2,
  Plus
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Textarea } from '../ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '../ui/dialog'

export default function Reports() {
  const [reports] = useState([
    {
      id: 1,
      title: 'Q1 2024 Executive Summary',
      type: 'Executive',
      status: 'Published',
      lastGenerated: new Date('2024-01-15'),
      schedule: 'Quarterly',
      recipients: ['CEO', 'CTO', 'Board'],
      format: 'PDF',
      size: '2.4 MB'
    },
    {
      id: 2,
      title: 'Weekly Development Metrics',
      type: 'Development',
      status: 'Scheduled',
      lastGenerated: new Date('2024-01-08'),
      schedule: 'Weekly',
      recipients: ['Engineering Team', 'Product Managers'],
      format: 'PDF + Excel',
      size: '1.8 MB'
    },
    {
      id: 3,
      title: 'Security Compliance Report',
      type: 'Security',
      status: 'Draft',
      lastGenerated: new Date('2024-01-10'),
      schedule: 'Monthly',
      recipients: ['CISO', 'Compliance Team'],
      format: 'PDF',
      size: '3.2 MB'
    },
    {
      id: 4,
      title: 'Cost Analysis Dashboard',
      type: 'Financial',
      status: 'Published',
      lastGenerated: new Date('2024-01-12'),
      schedule: 'Monthly',
      recipients: ['CFO', 'Finance Team'],
      format: 'Excel + PowerPoint',
      size: '4.1 MB'
    }
  ])

  const [templates] = useState([
    {
      id: 1,
      name: 'Executive Summary',
      description: 'High-level overview for C-suite and board members',
      sections: ['KPI Overview', 'Business Impact', 'Strategic Insights', 'Action Items'],
      estimatedTime: '5 minutes'
    },
    {
      id: 2,
      name: 'Technical Performance',
      description: 'Detailed technical metrics and system health',
      sections: ['System Health', 'Performance Metrics', 'Incident Analysis', 'Capacity Planning'],
      estimatedTime: '8 minutes'
    },
    {
      id: 3,
      name: 'Business Intelligence',
      description: 'ROI analysis and business impact assessment',
      sections: ['ROI Analysis', 'Cost Breakdown', 'Market Position', 'Recommendations'],
      estimatedTime: '7 minutes'
    }
  ])

  const [collaborations] = useState([
    {
      id: 1,
      reportTitle: 'Q1 2024 Executive Summary',
      user: 'Sarah Chen',
      action: 'Added comment',
      comment: 'The ROI figures look impressive. Can we include a comparison with Q4 2023?',
      timestamp: new Date(Date.now() - 3600000), // 1 hour ago
      status: 'pending'
    },
    {
      id: 2,
      reportTitle: 'Security Compliance Report',
      user: 'Mike Rodriguez',
      action: 'Approved section',
      comment: 'Security metrics section approved. Ready for publication.',
      timestamp: new Date(Date.now() - 7200000), // 2 hours ago
      status: 'approved'
    },
    {
      id: 3,
      reportTitle: 'Weekly Development Metrics',
      user: 'Lisa Wang',
      action: 'Requested changes',
      comment: 'Please update the velocity calculations to include bug fixes.',
      timestamp: new Date(Date.now() - 10800000), // 3 hours ago
      status: 'changes_requested'
    }
  ])

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'published': return 'default'
      case 'scheduled': return 'secondary'
      case 'draft': return 'outline'
      case 'generating': return 'default'
      default: return 'outline'
    }
  }

  const getActionColor = (status) => {
    switch (status) {
      case 'approved': return 'bg-green-100 text-green-800'
      case 'pending': return 'bg-yellow-100 text-yellow-800'
      case 'changes_requested': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Reports & Communication</h1>
          <p className="text-muted-foreground mt-1">
            Automated reporting, export capabilities, and collaboration tools
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <Dialog>
            <DialogTrigger asChild>
              <Button size="sm">
                <Plus className="w-4 h-4 mr-2" />
                New Report
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle>Create New Report</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="title">Report Title</Label>
                  <Input id="title" placeholder="Enter report title" />
                </div>
                <div>
                  <Label htmlFor="template">Template</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select template" />
                    </SelectTrigger>
                    <SelectContent>
                      {templates.map(template => (
                        <SelectItem key={template.id} value={template.id.toString()}>
                          {template.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label htmlFor="schedule">Schedule</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select schedule" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="once">One-time</SelectItem>
                      <SelectItem value="daily">Daily</SelectItem>
                      <SelectItem value="weekly">Weekly</SelectItem>
                      <SelectItem value="monthly">Monthly</SelectItem>
                      <SelectItem value="quarterly">Quarterly</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex justify-end space-x-2">
                  <Button variant="outline">Cancel</Button>
                  <Button>Create Report</Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <Tabs defaultValue="reports" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="reports">Reports</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="collaboration">Collaboration</TabsTrigger>
          <TabsTrigger value="exports">Export Center</TabsTrigger>
        </TabsList>

        {/* Reports Tab */}
        <TabsContent value="reports" className="space-y-6">
          <div className="grid grid-cols-1 gap-4">
            {reports.map((report) => (
              <Card key={report.id}>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h3 className="text-lg font-semibold text-foreground">
                          {report.title}
                        </h3>
                        <Badge variant={getStatusColor(report.status)}>
                          {report.status}
                        </Badge>
                        <Badge variant="outline">{report.type}</Badge>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-muted-foreground">
                        <div className="flex items-center">
                          <Clock className="w-4 h-4 mr-2" />
                          <span>Last: {report.lastGenerated.toLocaleDateString()}</span>
                        </div>
                        <div className="flex items-center">
                          <Calendar className="w-4 h-4 mr-2" />
                          <span>{report.schedule}</span>
                        </div>
                        <div className="flex items-center">
                          <Users className="w-4 h-4 mr-2" />
                          <span>{report.recipients.length} recipients</span>
                        </div>
                        <div className="flex items-center">
                          <FileText className="w-4 h-4 mr-2" />
                          <span>{report.format} • {report.size}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <Button variant="outline" size="sm">
                        <Eye className="w-4 h-4 mr-2" />
                        Preview
                      </Button>
                      <Button variant="outline" size="sm">
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                      <Button variant="outline" size="sm">
                        <Share2 className="w-4 h-4 mr-2" />
                        Share
                      </Button>
                      <Button variant="outline" size="sm">
                        <Edit className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Templates Tab */}
        <TabsContent value="templates" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {templates.map((template) => (
              <Card key={template.id}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{template.name}</span>
                    <Badge variant="outline">{template.estimatedTime}</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">
                    {template.description}
                  </p>
                  
                  <div className="space-y-2 mb-4">
                    <h4 className="text-sm font-medium">Sections:</h4>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      {template.sections.map((section, index) => (
                        <li key={index} className="flex items-center">
                          <span className="w-2 h-2 bg-primary rounded-full mr-2" />
                          {section}
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="flex space-x-2">
                    <Button size="sm" className="flex-1">
                      Use Template
                    </Button>
                    <Button variant="outline" size="sm">
                      <Edit className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Collaboration Tab */}
        <TabsContent value="collaboration" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {collaborations.map((activity) => (
                  <div key={activity.id} className="flex items-start space-x-4 p-4 border border-border rounded-lg">
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center text-primary-foreground text-sm font-medium">
                        {activity.user.split(' ').map(n => n[0]).join('')}
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h4 className="font-medium text-foreground">
                          {activity.user} {activity.action}
                        </h4>
                        <div className="flex items-center space-x-2">
                          <span className={`px-2 py-1 rounded text-xs ${getActionColor(activity.status)}`}>
                            {activity.status.replace('_', ' ')}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {activity.timestamp.toLocaleString()}
                          </span>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground mt-1">
                        Report: {activity.reportTitle}
                      </p>
                      {activity.comment && (
                        <div className="mt-2 p-3 bg-muted rounded-lg">
                          <p className="text-sm">{activity.comment}</p>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Collaboration Settings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Email Notifications</h4>
                    <p className="text-sm text-muted-foreground">
                      Receive email notifications for comments and approvals
                    </p>
                  </div>
                  <Button variant="outline" size="sm">
                    Configure
                  </Button>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Auto-sharing</h4>
                    <p className="text-sm text-muted-foreground">
                      Automatically share reports with designated recipients
                    </p>
                  </div>
                  <Button variant="outline" size="sm">
                    Manage
                  </Button>
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Approval Workflow</h4>
                    <p className="text-sm text-muted-foreground">
                      Require approval before publishing reports
                    </p>
                  </div>
                  <Button variant="outline" size="sm">
                    Setup
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Export Center Tab */}
        <TabsContent value="exports" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>PDF Export</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Generate professional PDF reports with charts and formatting
                </p>
                <div className="space-y-2">
                  <Label htmlFor="pdf-template">Template</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select template" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="executive">Executive Summary</SelectItem>
                      <SelectItem value="technical">Technical Report</SelectItem>
                      <SelectItem value="financial">Financial Analysis</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button className="w-full mt-4">
                  <Download className="w-4 h-4 mr-2" />
                  Export PDF
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>Excel Export</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Export data tables and metrics for further analysis
                </p>
                <div className="space-y-2">
                  <Label htmlFor="excel-data">Data Range</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select range" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="current">Current Quarter</SelectItem>
                      <SelectItem value="ytd">Year to Date</SelectItem>
                      <SelectItem value="custom">Custom Range</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button className="w-full mt-4">
                  <Download className="w-4 h-4 mr-2" />
                  Export Excel
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="w-5 h-5" />
                  <span>PowerPoint Export</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Create presentation-ready slides with charts and insights
                </p>
                <div className="space-y-2">
                  <Label htmlFor="ppt-style">Presentation Style</Label>
                  <Select>
                    <SelectTrigger>
                      <SelectValue placeholder="Select style" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="corporate">Corporate</SelectItem>
                      <SelectItem value="minimal">Minimal</SelectItem>
                      <SelectItem value="detailed">Detailed</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button className="w-full mt-4">
                  <Download className="w-4 h-4 mr-2" />
                  Export PowerPoint
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Scheduled Exports</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border border-border rounded-lg">
                  <div>
                    <h4 className="font-medium">Weekly Executive Summary</h4>
                    <p className="text-sm text-muted-foreground">
                      PDF export every Monday at 9:00 AM
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline">Active</Badge>
                    <Button variant="outline" size="sm">
                      <Settings className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
                
                <div className="flex items-center justify-between p-4 border border-border rounded-lg">
                  <div>
                    <h4 className="font-medium">Monthly Financial Report</h4>
                    <p className="text-sm text-muted-foreground">
                      Excel export on the 1st of each month
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline">Active</Badge>
                    <Button variant="outline" size="sm">
                      <Settings className="w-4 h-4" />
                    </Button>
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

