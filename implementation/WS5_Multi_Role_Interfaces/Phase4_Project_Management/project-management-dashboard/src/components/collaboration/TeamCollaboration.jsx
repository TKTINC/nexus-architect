import { useState } from 'react'
import { 
  MessageSquare, 
  Video, 
  Phone, 
  FileText, 
  Share2, 
  Users, 
  Calendar, 
  Bell,
  Search,
  Plus,
  Send,
  Paperclip,
  Smile,
  MoreHorizontal,
  Download,
  Edit,
  Trash2,
  Eye,
  Clock,
  CheckCircle,
  AlertCircle,
  Star,
  Hash,
  AtSign,
  Mic,
  Camera,
  Screen,
  Settings
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { Avatar, AvatarFallback, AvatarImage } from '../ui/avatar'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs'
import { Textarea } from '../ui/textarea'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuLabel, 
  DropdownMenuSeparator, 
  DropdownMenuTrigger 
} from '../ui/dropdown-menu'
import { Separator } from '../ui/separator'
import { ScrollArea } from '../ui/scroll-area'
import { teamMembers, communicationData } from '../../data/mockData'

const TeamCollaboration = () => {
  const [selectedChannel, setSelectedChannel] = useState('general')
  const [messageInput, setMessageInput] = useState('')
  const [selectedDocument, setSelectedDocument] = useState(null)
  const [activeCall, setActiveCall] = useState(null)

  const channels = [
    { id: 'general', name: 'General', type: 'public', members: 24, unread: 3 },
    { id: 'development', name: 'Development', type: 'public', members: 12, unread: 0 },
    { id: 'design', name: 'Design', type: 'public', members: 8, unread: 1 },
    { id: 'project-alpha', name: 'Project Alpha', type: 'private', members: 6, unread: 5 },
    { id: 'announcements', name: 'Announcements', type: 'public', members: 24, unread: 0 }
  ]

  const messages = [
    {
      id: 1,
      user: 'Sarah Chen',
      avatar: '/avatars/sarah-chen.jpg',
      message: 'Good morning team! Ready for our sprint planning session at 10 AM?',
      timestamp: '09:15 AM',
      reactions: [{ emoji: '👍', count: 3 }, { emoji: '✅', count: 2 }],
      replies: 2
    },
    {
      id: 2,
      user: 'John Doe',
      avatar: '/avatars/john-doe.jpg',
      message: 'The authentication system is now ready for testing. I\'ve deployed it to the staging environment.',
      timestamp: '09:32 AM',
      reactions: [{ emoji: '🎉', count: 4 }],
      replies: 0,
      attachments: [{ name: 'auth-system-docs.pdf', size: '2.3 MB' }]
    },
    {
      id: 3,
      user: 'Alice Johnson',
      avatar: '/avatars/alice-johnson.jpg',
      message: 'Great work John! I\'ll start the UI integration this afternoon.',
      timestamp: '09:45 AM',
      reactions: [],
      replies: 0
    },
    {
      id: 4,
      user: 'Bob Wilson',
      avatar: '/avatars/bob-wilson.jpg',
      message: 'Database optimization is complete. Performance improved by 40%! 📊',
      timestamp: '10:12 AM',
      reactions: [{ emoji: '🚀', count: 5 }, { emoji: '💪', count: 3 }],
      replies: 1
    }
  ]

  const documents = [
    {
      id: 1,
      name: 'Project Requirements Document',
      type: 'PDF',
      size: '4.2 MB',
      author: 'Sarah Chen',
      lastModified: '2 hours ago',
      collaborators: ['John Doe', 'Alice Johnson', 'Bob Wilson'],
      status: 'review'
    },
    {
      id: 2,
      name: 'API Documentation',
      type: 'Markdown',
      size: '1.8 MB',
      author: 'John Doe',
      lastModified: '1 day ago',
      collaborators: ['Sarah Chen', 'Bob Wilson'],
      status: 'approved'
    },
    {
      id: 3,
      name: 'UI Design System',
      type: 'Figma',
      size: '12.5 MB',
      author: 'Alice Johnson',
      lastModified: '3 hours ago',
      collaborators: ['Sarah Chen', 'Lisa Wang'],
      status: 'draft'
    },
    {
      id: 4,
      name: 'Database Schema',
      type: 'SQL',
      size: '856 KB',
      author: 'Bob Wilson',
      lastModified: '5 hours ago',
      collaborators: ['John Doe'],
      status: 'approved'
    }
  ]

  const meetings = [
    {
      id: 1,
      title: 'Sprint Planning',
      time: '10:00 AM - 11:00 AM',
      participants: ['Sarah Chen', 'John Doe', 'Alice Johnson', 'Bob Wilson'],
      status: 'upcoming',
      type: 'recurring'
    },
    {
      id: 2,
      title: 'Design Review',
      time: '2:00 PM - 3:00 PM',
      participants: ['Alice Johnson', 'Lisa Wang', 'Sarah Chen'],
      status: 'upcoming',
      type: 'one-time'
    },
    {
      id: 3,
      title: 'Code Review Session',
      time: '4:00 PM - 4:30 PM',
      participants: ['John Doe', 'Bob Wilson'],
      status: 'upcoming',
      type: 'one-time'
    }
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case 'approved':
        return 'bg-green-500'
      case 'review':
        return 'bg-yellow-500'
      case 'draft':
        return 'bg-blue-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getFileIcon = (type) => {
    switch (type.toLowerCase()) {
      case 'pdf':
        return <FileText className="h-4 w-4 text-red-500" />
      case 'markdown':
        return <FileText className="h-4 w-4 text-blue-500" />
      case 'figma':
        return <FileText className="h-4 w-4 text-purple-500" />
      case 'sql':
        return <FileText className="h-4 w-4 text-green-500" />
      default:
        return <FileText className="h-4 w-4 text-gray-500" />
    }
  }

  const MessageItem = ({ message }) => (
    <div className="flex space-x-3 p-3 hover:bg-accent rounded-lg">
      <Avatar className="h-8 w-8">
        <AvatarImage src={message.avatar} alt={message.user} />
        <AvatarFallback>{message.user.split(' ').map(n => n[0]).join('')}</AvatarFallback>
      </Avatar>
      <div className="flex-1 min-w-0">
        <div className="flex items-center space-x-2 mb-1">
          <span className="font-medium text-sm">{message.user}</span>
          <span className="text-xs text-muted-foreground">{message.timestamp}</span>
        </div>
        <p className="text-sm text-foreground mb-2">{message.message}</p>
        
        {message.attachments && (
          <div className="flex items-center space-x-2 mb-2">
            {message.attachments.map((attachment, index) => (
              <div key={index} className="flex items-center space-x-2 p-2 bg-muted rounded border">
                <Paperclip className="h-3 w-3" />
                <span className="text-xs">{attachment.name}</span>
                <span className="text-xs text-muted-foreground">({attachment.size})</span>
              </div>
            ))}
          </div>
        )}
        
        <div className="flex items-center space-x-4">
          {message.reactions.length > 0 && (
            <div className="flex items-center space-x-1">
              {message.reactions.map((reaction, index) => (
                <Button key={index} variant="ghost" size="sm" className="h-6 px-2 text-xs">
                  {reaction.emoji} {reaction.count}
                </Button>
              ))}
            </div>
          )}
          
          <Button variant="ghost" size="sm" className="h-6 px-2 text-xs">
            <Smile className="h-3 w-3 mr-1" />
            React
          </Button>
          
          {message.replies > 0 && (
            <Button variant="ghost" size="sm" className="h-6 px-2 text-xs">
              <MessageSquare className="h-3 w-3 mr-1" />
              {message.replies} replies
            </Button>
          )}
        </div>
      </div>
      
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="sm" className="p-1">
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem>
            <Edit className="h-4 w-4 mr-2" />
            Edit
          </DropdownMenuItem>
          <DropdownMenuItem>
            <MessageSquare className="h-4 w-4 mr-2" />
            Reply
          </DropdownMenuItem>
          <DropdownMenuItem>
            <Star className="h-4 w-4 mr-2" />
            Save
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem className="text-red-600">
            <Trash2 className="h-4 w-4 mr-2" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Team Collaboration</h1>
          <p className="text-muted-foreground">Communicate and collaborate with your team</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Video className="h-4 w-4 mr-2" />
            Start Meeting
          </Button>
          <Button size="sm">
            <Plus className="h-4 w-4 mr-2" />
            New Channel
          </Button>
        </div>
      </div>

      {/* Main Collaboration Interface */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-200px)]">
        {/* Sidebar - Channels and Team */}
        <div className="lg:col-span-1 space-y-4">
          {/* Channels */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center space-x-2">
                <Hash className="h-4 w-4" />
                <span>Channels</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {channels.map((channel) => (
                <div
                  key={channel.id}
                  className={`flex items-center justify-between p-2 rounded cursor-pointer hover:bg-accent ${
                    selectedChannel === channel.id ? 'bg-accent' : ''
                  }`}
                  onClick={() => setSelectedChannel(channel.id)}
                >
                  <div className="flex items-center space-x-2">
                    <Hash className="h-3 w-3 text-muted-foreground" />
                    <span className="text-sm">{channel.name}</span>
                  </div>
                  {channel.unread > 0 && (
                    <Badge variant="destructive" className="h-4 w-4 p-0 text-xs">
                      {channel.unread}
                    </Badge>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Online Team Members */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center space-x-2">
                <Users className="h-4 w-4" />
                <span>Team ({teamMembers.length})</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {teamMembers.slice(0, 6).map((member) => (
                <div key={member.id} className="flex items-center space-x-2 p-1">
                  <div className="relative">
                    <Avatar className="h-6 w-6">
                      <AvatarImage src={member.avatar} alt={member.name} />
                      <AvatarFallback className="text-xs">{member.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                    </Avatar>
                    <div className="absolute -bottom-0.5 -right-0.5 w-2 h-2 bg-green-500 rounded-full border border-background"></div>
                  </div>
                  <span className="text-sm">{member.name}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-3">
          <Tabs defaultValue="chat" className="h-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="chat">Chat</TabsTrigger>
              <TabsTrigger value="documents">Documents</TabsTrigger>
              <TabsTrigger value="meetings">Meetings</TabsTrigger>
              <TabsTrigger value="activity">Activity</TabsTrigger>
            </TabsList>

            {/* Chat Tab */}
            <TabsContent value="chat" className="h-full">
              <Card className="h-full flex flex-col">
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center space-x-2">
                      <Hash className="h-4 w-4" />
                      <span>{channels.find(c => c.id === selectedChannel)?.name}</span>
                      <Badge variant="secondary" className="text-xs">
                        {channels.find(c => c.id === selectedChannel)?.members} members
                      </Badge>
                    </CardTitle>
                    <div className="flex items-center space-x-2">
                      <Button variant="ghost" size="sm">
                        <Phone className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Video className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Settings className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="flex-1 flex flex-col">
                  {/* Messages */}
                  <ScrollArea className="flex-1 mb-4">
                    <div className="space-y-1">
                      {messages.map((message) => (
                        <MessageItem key={message.id} message={message} />
                      ))}
                    </div>
                  </ScrollArea>
                  
                  {/* Message Input */}
                  <div className="flex items-center space-x-2">
                    <div className="flex-1 relative">
                      <Textarea
                        placeholder={`Message #${channels.find(c => c.id === selectedChannel)?.name}`}
                        value={messageInput}
                        onChange={(e) => setMessageInput(e.target.value)}
                        className="min-h-[40px] max-h-32 resize-none pr-20"
                      />
                      <div className="absolute right-2 bottom-2 flex items-center space-x-1">
                        <Button variant="ghost" size="sm" className="p-1">
                          <Paperclip className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="sm" className="p-1">
                          <Smile className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    <Button size="sm">
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Documents Tab */}
            <TabsContent value="documents">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center space-x-2">
                      <FileText className="h-5 w-5" />
                      <span>Shared Documents</span>
                    </CardTitle>
                    <Button size="sm">
                      <Plus className="h-4 w-4 mr-2" />
                      Upload Document
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {documents.map((doc) => (
                      <div key={doc.id} className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent">
                        <div className="flex items-center space-x-3">
                          {getFileIcon(doc.type)}
                          <div>
                            <h4 className="font-medium text-sm">{doc.name}</h4>
                            <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                              <span>by {doc.author}</span>
                              <span>•</span>
                              <span>{doc.lastModified}</span>
                              <span>•</span>
                              <span>{doc.size}</span>
                            </div>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full ${getStatusColor(doc.status)}`}></div>
                          <Badge variant="secondary" className="text-xs">{doc.status}</Badge>
                          
                          <div className="flex -space-x-1">
                            {doc.collaborators.slice(0, 3).map((collaborator, index) => (
                              <Avatar key={index} className="h-6 w-6 border-2 border-background">
                                <AvatarFallback className="text-xs">{collaborator.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                              </Avatar>
                            ))}
                          </div>
                          
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="sm" className="p-1">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem>
                                <Eye className="h-4 w-4 mr-2" />
                                View
                              </DropdownMenuItem>
                              <DropdownMenuItem>
                                <Edit className="h-4 w-4 mr-2" />
                                Edit
                              </DropdownMenuItem>
                              <DropdownMenuItem>
                                <Download className="h-4 w-4 mr-2" />
                                Download
                              </DropdownMenuItem>
                              <DropdownMenuItem>
                                <Share2 className="h-4 w-4 mr-2" />
                                Share
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Meetings Tab */}
            <TabsContent value="meetings">
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center space-x-2">
                        <Calendar className="h-5 w-5" />
                        <span>Today's Meetings</span>
                      </CardTitle>
                      <Button size="sm">
                        <Plus className="h-4 w-4 mr-2" />
                        Schedule Meeting
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {meetings.map((meeting) => (
                        <div key={meeting.id} className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex items-center space-x-3">
                            <div className="flex-shrink-0">
                              <Calendar className="h-4 w-4 text-blue-500" />
                            </div>
                            <div>
                              <h4 className="font-medium text-sm">{meeting.title}</h4>
                              <p className="text-xs text-muted-foreground">{meeting.time}</p>
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            <div className="flex -space-x-1">
                              {meeting.participants.slice(0, 4).map((participant, index) => (
                                <Avatar key={index} className="h-6 w-6 border-2 border-background">
                                  <AvatarFallback className="text-xs">{participant.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                                </Avatar>
                              ))}
                            </div>
                            
                            <Button size="sm" variant="outline">
                              <Video className="h-4 w-4 mr-2" />
                              Join
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Quick Actions */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardContent className="p-4 text-center">
                      <Video className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                      <h3 className="font-medium text-sm">Start Video Call</h3>
                      <p className="text-xs text-muted-foreground">Instant video meeting</p>
                    </CardContent>
                  </Card>
                  
                  <Card className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardContent className="p-4 text-center">
                      <Screen className="h-8 w-8 mx-auto mb-2 text-green-500" />
                      <h3 className="font-medium text-sm">Screen Share</h3>
                      <p className="text-xs text-muted-foreground">Share your screen</p>
                    </CardContent>
                  </Card>
                  
                  <Card className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardContent className="p-4 text-center">
                      <Calendar className="h-8 w-8 mx-auto mb-2 text-purple-500" />
                      <h3 className="font-medium text-sm">Schedule Meeting</h3>
                      <p className="text-xs text-muted-foreground">Plan future meetings</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>

            {/* Activity Tab */}
            <TabsContent value="activity">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Bell className="h-5 w-5" />
                    <span>Team Activity</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {communicationData.map((activity) => (
                      <div key={activity.id} className="flex items-start space-x-3 p-3 border rounded-lg">
                        <div className="flex-shrink-0">
                          {activity.type === 'meeting' && <Calendar className="h-4 w-4 text-blue-500" />}
                          {activity.type === 'message' && <MessageSquare className="h-4 w-4 text-green-500" />}
                          {activity.type === 'document' && <FileText className="h-4 w-4 text-purple-500" />}
                        </div>
                        <div className="flex-1">
                          <h4 className="font-medium text-sm">{activity.title}</h4>
                          <p className="text-xs text-muted-foreground mb-2">
                            {activity.type === 'meeting' && `Meeting with ${activity.participants.join(', ')}`}
                            {activity.type === 'message' && `Discussion between ${activity.participants.join(', ')}`}
                            {activity.type === 'document' && `Document by ${activity.author} • ${activity.collaborators.join(', ')}`}
                          </p>
                          <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                            <span>{activity.date}</span>
                            {activity.time && <span>• {activity.time}</span>}
                            <Badge variant="secondary" className="text-xs">{activity.status}</Badge>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}

export default TeamCollaboration

