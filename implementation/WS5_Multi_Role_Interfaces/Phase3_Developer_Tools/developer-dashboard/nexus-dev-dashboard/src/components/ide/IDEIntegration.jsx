import { useState } from 'react'
import { 
  Puzzle, 
  Download, 
  Settings, 
  CheckCircle,
  AlertCircle,
  ExternalLink,
  Code,
  Zap,
  Eye,
  GitBranch,
  FileText,
  Terminal,
  Smartphone,
  Monitor,
  Cloud,
  RefreshCw,
  Play,
  Pause
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Progress } from '../ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { Switch } from '../ui/switch'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

import { ideIntegrations } from '../../data/mockData'

const IDEIntegration = () => {
  const [extensionSettings, setExtensionSettings] = useState({
    realTimeAnalysis: true,
    autoSuggestions: true,
    codeReview: false,
    performanceHints: true,
    securityScanning: true
  })

  const extensions = [
    {
      name: 'Nexus Code Assistant',
      description: 'AI-powered code completion and analysis',
      version: '2.1.3',
      status: 'installed',
      ide: 'VS Code',
      icon: Code,
      features: ['Code completion', 'Error detection', 'Refactoring suggestions'],
      usage: { daily: 8.5, suggestions: 156, accepted: 89 }
    },
    {
      name: 'Nexus Quality Checker',
      description: 'Real-time code quality analysis and metrics',
      version: '1.8.2',
      status: 'available',
      ide: 'IntelliJ IDEA',
      icon: CheckCircle,
      features: ['Quality metrics', 'Technical debt detection', 'Performance analysis'],
      usage: { daily: 0, suggestions: 0, accepted: 0 }
    },
    {
      name: 'Nexus Workflow Optimizer',
      description: 'Automated workflow optimization and task management',
      version: '1.5.1',
      status: 'beta',
      ide: 'VS Code',
      icon: Zap,
      features: ['Task automation', 'Workflow analysis', 'Time tracking'],
      usage: { daily: 3.2, suggestions: 45, accepted: 32 }
    },
    {
      name: 'Nexus Security Scanner',
      description: 'Security vulnerability detection and remediation',
      version: '2.0.0',
      status: 'installed',
      ide: 'VS Code',
      icon: AlertCircle,
      features: ['Vulnerability scanning', 'Security recommendations', 'Compliance checks'],
      usage: { daily: 1.8, suggestions: 23, accepted: 18 }
    }
  ]

  const webIDEFeatures = [
    {
      name: 'Cloud Workspace',
      description: 'Access your development environment from anywhere',
      icon: Cloud,
      enabled: true,
      usage: '2.1 hours/day'
    },
    {
      name: 'Real-time Collaboration',
      description: 'Collaborate with team members in real-time',
      icon: Users,
      enabled: true,
      usage: '45 min/day'
    },
    {
      name: 'Live Preview',
      description: 'See changes instantly with live preview',
      icon: Eye,
      enabled: false,
      usage: '0 min/day'
    },
    {
      name: 'Version Control',
      description: 'Integrated Git workflow and branch management',
      icon: GitBranch,
      enabled: true,
      usage: '1.2 hours/day'
    },
    {
      name: 'Terminal Access',
      description: 'Full terminal access for command-line operations',
      icon: Terminal,
      enabled: true,
      usage: '30 min/day'
    },
    {
      name: 'Mobile Support',
      description: 'Code review and light editing on mobile devices',
      icon: Smartphone,
      enabled: false,
      usage: '0 min/day'
    }
  ]

  const usageData = [
    { date: '2024-01-01', vsCode: 6.2, webIDE: 1.8, intellij: 0.5 },
    { date: '2024-01-02', vsCode: 7.1, webIDE: 2.1, intellij: 0.3 },
    { date: '2024-01-03', vsCode: 8.3, webIDE: 1.9, intellij: 0.8 },
    { date: '2024-01-04', vsCode: 7.8, webIDE: 2.3, intellij: 0.6 },
    { date: '2024-01-05', vsCode: 8.9, webIDE: 2.0, intellij: 0.4 },
    { date: '2024-01-06', vsCode: 8.1, webIDE: 2.2, intellij: 0.7 },
    { date: '2024-01-07', vsCode: 8.5, webIDE: 2.1, intellij: 0.5 }
  ]

  const suggestionData = [
    { category: 'Code Quality', suggestions: 45, accepted: 38 },
    { category: 'Performance', suggestions: 23, accepted: 19 },
    { category: 'Security', suggestions: 18, accepted: 15 },
    { category: 'Refactoring', suggestions: 32, accepted: 24 },
    { category: 'Documentation', suggestions: 12, accepted: 8 }
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case 'installed':
        return 'bg-green-500'
      case 'available':
        return 'bg-blue-500'
      case 'beta':
        return 'bg-yellow-500'
      case 'updating':
        return 'bg-purple-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getStatusText = (status) => {
    switch (status) {
      case 'installed':
        return 'Installed'
      case 'available':
        return 'Available'
      case 'beta':
        return 'Beta'
      case 'updating':
        return 'Updating'
      default:
        return 'Unknown'
    }
  }

  const toggleSetting = (setting) => {
    setExtensionSettings(prev => ({
      ...prev,
      [setting]: !prev[setting]
    }))
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">IDE Integration</h1>
          <p className="text-muted-foreground">
            Manage IDE extensions, configure tools, and optimize your development environment
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Sync Settings
          </Button>
          <Button size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Configure
          </Button>
        </div>
      </div>

      {/* Integration Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {ideIntegrations.map((integration, index) => (
          <Card key={index}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{integration.name}</CardTitle>
                <div className={`w-3 h-3 rounded-full ${getStatusColor(integration.status)}`}></div>
              </div>
              <CardDescription>Version {integration.version}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Daily Usage</span>
                    <span className="font-medium">{integration.usage.daily} hours</span>
                  </div>
                  <Progress value={(integration.usage.daily / 10) * 100} className="h-2" />
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-center">
                  <div>
                    <div className="text-lg font-bold">{integration.usage.suggestions}</div>
                    <div className="text-xs text-muted-foreground">Suggestions</div>
                  </div>
                  <div>
                    <div className="text-lg font-bold">{integration.usage.accepted}</div>
                    <div className="text-xs text-muted-foreground">Accepted</div>
                  </div>
                </div>
                
                <div className="space-y-1">
                  {integration.features.map((feature, idx) => (
                    <div key={idx} className="text-xs text-muted-foreground">
                      • {feature}
                    </div>
                  ))}
                </div>
                
                <div className="flex space-x-2">
                  {integration.status === 'installed' ? (
                    <>
                      <Button size="sm" variant="outline" className="flex-1">
                        <Settings className="h-4 w-4 mr-1" />
                        Configure
                      </Button>
                      <Button size="sm" variant="outline">
                        <ExternalLink className="h-4 w-4" />
                      </Button>
                    </>
                  ) : integration.status === 'available' ? (
                    <Button size="sm" className="flex-1">
                      <Download className="h-4 w-4 mr-1" />
                      Install
                    </Button>
                  ) : (
                    <Button size="sm" variant="outline" className="flex-1">
                      <Eye className="h-4 w-4 mr-1" />
                      Preview
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="extensions" className="space-y-6">
        <TabsList>
          <TabsTrigger value="extensions">Extensions</TabsTrigger>
          <TabsTrigger value="web-ide">Web IDE</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="extensions" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {extensions.map((extension, index) => {
              const Icon = extension.icon
              return (
                <Card key={index}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-3">
                        <Icon className="h-6 w-6 text-muted-foreground" />
                        <div>
                          <CardTitle className="text-lg">{extension.name}</CardTitle>
                          <CardDescription>{extension.description}</CardDescription>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">{extension.ide}</Badge>
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(extension.status)}`}></div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Version</span>
                        <span>{extension.version}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Status</span>
                        <Badge variant="secondary">{getStatusText(extension.status)}</Badge>
                      </div>
                      
                      {extension.usage.daily > 0 && (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between text-sm">
                            <span className="text-muted-foreground">Daily Usage</span>
                            <span>{extension.usage.daily} hours</span>
                          </div>
                          <Progress value={(extension.usage.daily / 10) * 100} className="h-2" />
                        </div>
                      )}
                      
                      <div className="space-y-1">
                        {extension.features.map((feature, idx) => (
                          <div key={idx} className="flex items-center space-x-2 text-sm">
                            <CheckCircle className="h-3 w-3 text-green-500" />
                            <span>{feature}</span>
                          </div>
                        ))}
                      </div>
                      
                      <div className="flex space-x-2">
                        {extension.status === 'installed' ? (
                          <>
                            <Button size="sm" variant="outline" className="flex-1">
                              <Settings className="h-4 w-4 mr-1" />
                              Settings
                            </Button>
                            <Button size="sm" variant="outline">
                              <RefreshCw className="h-4 w-4" />
                            </Button>
                          </>
                        ) : extension.status === 'available' ? (
                          <Button size="sm" className="flex-1">
                            <Download className="h-4 w-4 mr-1" />
                            Install
                          </Button>
                        ) : (
                          <Button size="sm" variant="outline" className="flex-1">
                            <Eye className="h-4 w-4 mr-1" />
                            Preview
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </TabsContent>

        <TabsContent value="web-ide" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Web IDE Features</CardTitle>
              <CardDescription>Configure and manage web-based development environment</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {webIDEFeatures.map((feature, index) => {
                  const Icon = feature.icon
                  return (
                    <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center space-x-4">
                        <Icon className="h-5 w-5 text-muted-foreground" />
                        <div className="flex-1">
                          <h4 className="font-medium">{feature.name}</h4>
                          <p className="text-sm text-muted-foreground">{feature.description}</p>
                          <div className="text-xs text-muted-foreground mt-1">
                            Usage: {feature.usage}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Switch checked={feature.enabled} />
                        <Button size="sm" variant="outline">
                          Configure
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Web IDE Access</CardTitle>
              <CardDescription>Quick access to your cloud development environment</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <Monitor className="h-8 w-8 text-blue-500" />
                  <div>
                    <h4 className="font-medium">Nexus Web IDE</h4>
                    <p className="text-sm text-muted-foreground">
                      Full-featured development environment in your browser
                    </p>
                    <div className="text-xs text-muted-foreground mt-1">
                      Last accessed: 2 hours ago
                    </div>
                  </div>
                </div>
                <div className="flex space-x-2">
                  <Button size="sm" variant="outline">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Open in New Tab
                  </Button>
                  <Button size="sm">
                    <Play className="h-4 w-4 mr-2" />
                    Launch IDE
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Extension Settings</CardTitle>
              <CardDescription>Configure global settings for all Nexus extensions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Real-time Code Analysis</h4>
                    <p className="text-sm text-muted-foreground">
                      Analyze code as you type for immediate feedback
                    </p>
                  </div>
                  <Switch 
                    checked={extensionSettings.realTimeAnalysis}
                    onCheckedChange={() => toggleSetting('realTimeAnalysis')}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Auto Suggestions</h4>
                    <p className="text-sm text-muted-foreground">
                      Automatically show code completion suggestions
                    </p>
                  </div>
                  <Switch 
                    checked={extensionSettings.autoSuggestions}
                    onCheckedChange={() => toggleSetting('autoSuggestions')}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Code Review Assistance</h4>
                    <p className="text-sm text-muted-foreground">
                      Provide suggestions during code review process
                    </p>
                  </div>
                  <Switch 
                    checked={extensionSettings.codeReview}
                    onCheckedChange={() => toggleSetting('codeReview')}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Performance Hints</h4>
                    <p className="text-sm text-muted-foreground">
                      Show performance optimization suggestions
                    </p>
                  </div>
                  <Switch 
                    checked={extensionSettings.performanceHints}
                    onCheckedChange={() => toggleSetting('performanceHints')}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium">Security Scanning</h4>
                    <p className="text-sm text-muted-foreground">
                      Automatically scan for security vulnerabilities
                    </p>
                  </div>
                  <Switch 
                    checked={extensionSettings.securityScanning}
                    onCheckedChange={() => toggleSetting('securityScanning')}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Usage Trends */}
            <Card>
              <CardHeader>
                <CardTitle>IDE Usage Trends</CardTitle>
                <CardDescription>Daily usage across different development environments</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={usageData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tickFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <YAxis />
                    <Tooltip labelFormatter={(date) => new Date(date).toLocaleDateString()} />
                    <Line type="monotone" dataKey="vsCode" stroke="#007ACC" strokeWidth={2} name="VS Code" />
                    <Line type="monotone" dataKey="webIDE" stroke="#22c55e" strokeWidth={2} name="Web IDE" />
                    <Line type="monotone" dataKey="intellij" stroke="#ff6b35" strokeWidth={2} name="IntelliJ" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Suggestion Analytics */}
            <Card>
              <CardHeader>
                <CardTitle>AI Suggestions</CardTitle>
                <CardDescription>Suggestions provided and accepted by category</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={suggestionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="suggestions" fill="#8b5cf6" name="Suggestions" />
                    <Bar dataKey="accepted" fill="#22c55e" name="Accepted" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Performance Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Integration Performance</CardTitle>
              <CardDescription>Overall performance metrics for IDE integrations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-500">8.5</div>
                  <div className="text-sm text-muted-foreground">Hours/Day</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Average IDE usage
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-500">89%</div>
                  <div className="text-sm text-muted-foreground">Acceptance Rate</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    AI suggestions accepted
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-500">156</div>
                  <div className="text-sm text-muted-foreground">Suggestions/Week</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    AI-powered recommendations
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-orange-500">25%</div>
                  <div className="text-sm text-muted-foreground">Productivity Gain</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Estimated improvement
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

export default IDEIntegration

