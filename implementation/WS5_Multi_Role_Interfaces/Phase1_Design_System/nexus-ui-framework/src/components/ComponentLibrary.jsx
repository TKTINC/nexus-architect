import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './molecules/Card'
import { Button } from './atoms/Button'
import { Input, SearchInput, InputGroup } from './atoms/Input'
import { H1, H2, H3, P, Code } from './atoms/Typography'
import { Form, FormField, SubmitButton, validators } from './molecules/Form'
import { Modal, ModalContent, ModalHeader, ModalTitle, ModalBody, ModalFooter } from './molecules/Modal'
import { 
  Search, 
  Filter, 
  Grid, 
  List, 
  Eye, 
  Code2, 
  Copy,
  Check,
  ExternalLink,
  Zap,
  Palette,
  Type,
  Layout,
  MousePointer
} from 'lucide-react'

export const ComponentLibrary = () => {
  const [searchQuery, setSearchQuery] = React.useState('')
  const [selectedCategory, setSelectedCategory] = React.useState('all')
  const [viewMode, setViewMode] = React.useState('grid')
  const [selectedComponent, setSelectedComponent] = React.useState(null)
  const [copiedCode, setCopiedCode] = React.useState('')

  const categories = [
    { id: 'all', label: 'All Components', count: 40 },
    { id: 'atoms', label: 'Atoms', count: 15 },
    { id: 'molecules', label: 'Molecules', count: 12 },
    { id: 'organisms', label: 'Organisms', count: 8 },
    { id: 'layout', label: 'Layout', count: 5 }
  ]

  const components = [
    // Atoms
    {
      id: 'button',
      name: 'Button',
      category: 'atoms',
      description: 'Interactive button component with multiple variants and states',
      tags: ['interactive', 'form', 'action'],
      status: 'stable',
      usage: 'high',
      code: `<Button variant="default" size="md">
  Click me
</Button>`,
      preview: () => <Button>Click me</Button>
    },
    {
      id: 'input',
      name: 'Input',
      category: 'atoms',
      description: 'Text input field with validation and accessibility features',
      tags: ['form', 'text', 'validation'],
      status: 'stable',
      usage: 'high',
      code: `<Input 
  placeholder="Enter text"
  type="text"
/>`,
      preview: () => <Input placeholder="Enter text" />
    },
    {
      id: 'typography',
      name: 'Typography',
      category: 'atoms',
      description: 'Semantic text components with consistent styling',
      tags: ['text', 'heading', 'content'],
      status: 'stable',
      usage: 'high',
      code: `<H2>Heading Text</H2>
<P>Body paragraph text</P>`,
      preview: () => (
        <div>
          <H2 className="text-lg">Heading Text</H2>
          <P className="text-sm">Body paragraph text</P>
        </div>
      )
    },

    // Molecules
    {
      id: 'card',
      name: 'Card',
      category: 'molecules',
      description: 'Flexible container component for grouping related content',
      tags: ['container', 'content', 'layout'],
      status: 'stable',
      usage: 'high',
      code: `<Card>
  <CardHeader>
    <CardTitle>Card Title</CardTitle>
    <CardDescription>Description</CardDescription>
  </CardHeader>
  <CardContent>
    Card content goes here
  </CardContent>
</Card>`,
      preview: () => (
        <Card className="w-full max-w-sm">
          <CardHeader>
            <CardTitle className="text-base">Card Title</CardTitle>
            <CardDescription className="text-sm">Description</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm">Card content goes here</p>
          </CardContent>
        </Card>
      )
    },
    {
      id: 'form',
      name: 'Form',
      category: 'molecules',
      description: 'Form component with validation and accessibility',
      tags: ['form', 'validation', 'input'],
      status: 'stable',
      usage: 'medium',
      code: `<Form onSubmit={handleSubmit}>
  <FormField name="email" label="Email">
    <Input type="email" />
  </FormField>
  <SubmitButton>Submit</SubmitButton>
</Form>`,
      preview: () => (
        <Form className="w-full max-w-sm space-y-3">
          <FormField name="email" label="Email">
            <Input type="email" placeholder="Enter email" />
          </FormField>
          <SubmitButton size="sm">Submit</SubmitButton>
        </Form>
      )
    },
    {
      id: 'modal',
      name: 'Modal',
      category: 'molecules',
      description: 'Overlay dialog component with focus management',
      tags: ['overlay', 'dialog', 'popup'],
      status: 'stable',
      usage: 'medium',
      code: `<Modal open={isOpen} onOpenChange={setIsOpen}>
  <ModalContent>
    <ModalHeader>
      <ModalTitle>Modal Title</ModalTitle>
    </ModalHeader>
    <ModalBody>
      Modal content
    </ModalBody>
  </ModalContent>
</Modal>`,
      preview: () => (
        <Button 
          size="sm" 
          onClick={() => setSelectedComponent('modal-demo')}
        >
          Open Modal
        </Button>
      )
    },

    // Organisms
    {
      id: 'header',
      name: 'Header',
      category: 'organisms',
      description: 'Application header with navigation and user controls',
      tags: ['navigation', 'layout', 'header'],
      status: 'stable',
      usage: 'high',
      code: `<Header 
  onMenuToggle={toggleMenu}
  user={currentUser}
  showSearch={true}
/>`,
      preview: () => (
        <div className="w-full h-16 bg-background border rounded-lg flex items-center px-4">
          <div className="flex items-center space-x-2">
            <div className="h-6 w-6 bg-primary rounded"></div>
            <span className="font-semibold text-sm">Header</span>
          </div>
        </div>
      )
    },
    {
      id: 'sidebar',
      name: 'Sidebar',
      category: 'organisms',
      description: 'Navigation sidebar with role-based menu items',
      tags: ['navigation', 'layout', 'sidebar'],
      status: 'stable',
      usage: 'high',
      code: `<Sidebar 
  isOpen={sidebarOpen}
  userRole="admin"
  onClose={closeSidebar}
/>`,
      preview: () => (
        <div className="w-full h-32 bg-muted border rounded-lg flex">
          <div className="w-16 bg-background border-r flex flex-col items-center py-2 space-y-2">
            <div className="h-2 w-8 bg-primary rounded"></div>
            <div className="h-2 w-6 bg-muted-foreground rounded"></div>
            <div className="h-2 w-6 bg-muted-foreground rounded"></div>
          </div>
          <div className="flex-1 p-2">
            <div className="text-xs text-muted-foreground">Sidebar Content</div>
          </div>
        </div>
      )
    },

    // Layout
    {
      id: 'footer',
      name: 'Footer',
      category: 'layout',
      description: 'Application footer with links and system status',
      tags: ['layout', 'footer', 'links'],
      status: 'stable',
      usage: 'medium',
      code: `<Footer 
  variant="default"
  showSystemStatus={true}
  showSocialLinks={true}
/>`,
      preview: () => (
        <div className="w-full h-16 bg-muted border rounded-lg flex items-center justify-between px-4">
          <span className="text-xs text-muted-foreground">© 2024 Company</span>
          <div className="flex space-x-2">
            <div className="h-2 w-2 bg-green-500 rounded-full"></div>
            <span className="text-xs text-muted-foreground">Online</span>
          </div>
        </div>
      )
    }
  ]

  const filteredComponents = components.filter(component => {
    const matchesSearch = component.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         component.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         component.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    const matchesCategory = selectedCategory === 'all' || component.category === selectedCategory
    
    return matchesSearch && matchesCategory
  })

  const copyCode = (code, componentId) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(componentId)
    setTimeout(() => setCopiedCode(''), 2000)
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'stable': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'beta': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'alpha': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getUsageColor = (usage) => {
    switch (usage) {
      case 'high': return 'text-green-600'
      case 'medium': return 'text-yellow-600'
      case 'low': return 'text-gray-600'
      default: return 'text-gray-600'
    }
  }

  return (
    <div className="space-y-8 p-6">
      {/* Header */}
      <div className="space-y-4">
        <H1>Component Library</H1>
        <P className="text-lg text-muted-foreground">
          Explore and interact with all available components in the design system
        </P>
      </div>

      {/* Filters and Search */}
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
            {/* Search */}
            <div className="flex-1 max-w-md">
              <SearchInput
                placeholder="Search components..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            {/* Category Filter */}
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-muted-foreground" />
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              >
                {categories.map(category => (
                  <option key={category.id} value={category.id}>
                    {category.label} ({category.count})
                  </option>
                ))}
              </select>
            </div>

            {/* View Mode */}
            <div className="flex items-center space-x-1 border rounded-lg p-1">
              <Button
                variant={viewMode === 'grid' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('grid')}
                className="h-8 w-8 p-0"
              >
                <Grid className="h-4 w-4" />
              </Button>
              <Button
                variant={viewMode === 'list' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setViewMode('list')}
                className="h-8 w-8 p-0"
              >
                <List className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results Count */}
      <div className="flex items-center justify-between">
        <P className="text-sm text-muted-foreground">
          Showing {filteredComponents.length} of {components.length} components
        </P>
        
        {searchQuery && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSearchQuery('')}
          >
            Clear search
          </Button>
        )}
      </div>

      {/* Components Grid/List */}
      {viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredComponents.map(component => (
            <Card key={component.id} className="group hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-lg flex items-center gap-2">
                      {component.name}
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(component.status)}`}>
                        {component.status}
                      </span>
                    </CardTitle>
                    <CardDescription className="text-sm">
                      {component.description}
                    </CardDescription>
                  </div>
                  <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => setSelectedComponent(component)}
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8"
                      onClick={() => copyCode(component.code, component.id)}
                    >
                      {copiedCode === component.id ? (
                        <Check className="h-4 w-4 text-green-500" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                {/* Preview */}
                <div className="p-4 border rounded-lg bg-muted/30 min-h-[100px] flex items-center justify-center">
                  {component.preview()}
                </div>

                {/* Metadata */}
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2">
                    <span className="text-muted-foreground">Usage:</span>
                    <span className={`font-medium ${getUsageColor(component.usage)}`}>
                      {component.usage}
                    </span>
                  </div>
                  <div className="flex items-center space-x-1">
                    {component.tags.slice(0, 2).map(tag => (
                      <span key={tag} className="px-2 py-1 bg-secondary text-secondary-foreground rounded text-xs">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredComponents.map(component => (
            <Card key={component.id} className="group hover:shadow-md transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="w-16 h-16 border rounded-lg bg-muted/30 flex items-center justify-center">
                      {component.preview()}
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2">
                        <H3 className="text-lg">{component.name}</H3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(component.status)}`}>
                          {component.status}
                        </span>
                        <span className={`text-sm font-medium ${getUsageColor(component.usage)}`}>
                          {component.usage} usage
                        </span>
                      </div>
                      <P className="text-sm text-muted-foreground">
                        {component.description}
                      </P>
                      <div className="flex items-center space-x-1">
                        {component.tags.map(tag => (
                          <span key={tag} className="px-2 py-1 bg-secondary text-secondary-foreground rounded text-xs">
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setSelectedComponent(component)}
                    >
                      <Eye className="h-4 w-4 mr-2" />
                      View
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => copyCode(component.code, component.id)}
                    >
                      {copiedCode === component.id ? (
                        <Check className="h-4 w-4 mr-2 text-green-500" />
                      ) : (
                        <Copy className="h-4 w-4 mr-2" />
                      )}
                      Copy
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Empty State */}
      {filteredComponents.length === 0 && (
        <Card>
          <CardContent className="p-12 text-center">
            <Search className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <H3>No components found</H3>
            <P className="text-muted-foreground mt-2">
              Try adjusting your search or filter criteria
            </P>
            <Button
              variant="outline"
              className="mt-4"
              onClick={() => {
                setSearchQuery('')
                setSelectedCategory('all')
              }}
            >
              Clear filters
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Component Detail Modal */}
      {selectedComponent && selectedComponent !== 'modal-demo' && (
        <Modal open={!!selectedComponent} onOpenChange={() => setSelectedComponent(null)}>
          <ModalContent size="lg">
            <ModalHeader>
              <ModalTitle className="flex items-center gap-2">
                {selectedComponent.name}
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedComponent.status)}`}>
                  {selectedComponent.status}
                </span>
              </ModalTitle>
            </ModalHeader>
            <ModalBody className="space-y-6">
              <P>{selectedComponent.description}</P>
              
              {/* Preview */}
              <div className="space-y-2">
                <H3 className="text-base">Preview</H3>
                <div className="p-6 border rounded-lg bg-muted/30 flex items-center justify-center min-h-[120px]">
                  {selectedComponent.preview()}
                </div>
              </div>

              {/* Code */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <H3 className="text-base">Code</H3>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => copyCode(selectedComponent.code, selectedComponent.id)}
                  >
                    {copiedCode === selectedComponent.id ? (
                      <Check className="h-4 w-4 mr-2 text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4 mr-2" />
                    )}
                    Copy
                  </Button>
                </div>
                <Code className="text-sm bg-muted p-4 rounded-lg overflow-x-auto">
                  {selectedComponent.code}
                </Code>
              </div>

              {/* Metadata */}
              <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                <div>
                  <div className="text-sm font-medium">Category</div>
                  <div className="text-sm text-muted-foreground capitalize">{selectedComponent.category}</div>
                </div>
                <div>
                  <div className="text-sm font-medium">Usage</div>
                  <div className={`text-sm font-medium ${getUsageColor(selectedComponent.usage)}`}>
                    {selectedComponent.usage}
                  </div>
                </div>
              </div>
            </ModalBody>
            <ModalFooter>
              <Button variant="outline" onClick={() => setSelectedComponent(null)}>
                Close
              </Button>
              <Button>
                <ExternalLink className="h-4 w-4 mr-2" />
                View Documentation
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      )}

      {/* Demo Modal */}
      {selectedComponent === 'modal-demo' && (
        <Modal open={true} onOpenChange={() => setSelectedComponent(null)}>
          <ModalContent>
            <ModalHeader>
              <ModalTitle>Demo Modal</ModalTitle>
            </ModalHeader>
            <ModalBody>
              <P>This is a demonstration of the Modal component with focus management and accessibility features.</P>
            </ModalBody>
            <ModalFooter>
              <Button variant="outline" onClick={() => setSelectedComponent(null)}>
                Cancel
              </Button>
              <Button onClick={() => setSelectedComponent(null)}>
                Confirm
              </Button>
            </ModalFooter>
          </ModalContent>
        </Modal>
      )}
    </div>
  )
}

