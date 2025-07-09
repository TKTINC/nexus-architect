import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './molecules/Card'
import { Button } from './atoms/Button'
import { Input } from './atoms/Input'
import { H1, H2, H3, P, Code, Lead } from './atoms/Typography'
import { 
  Palette, 
  Type, 
  Layout, 
  Accessibility, 
  Smartphone, 
  Monitor,
  Tablet,
  Zap,
  Shield,
  Globe,
  CheckCircle,
  Star,
  Heart,
  Download,
  ExternalLink
} from 'lucide-react'
import { colors } from './design-tokens/colors'
import { typography } from './design-tokens/typography'
import { spacing } from './design-tokens/spacing'

export const DesignSystemShowcase = () => {
  const [selectedTheme, setSelectedTheme] = React.useState('light')

  const features = [
    {
      icon: Palette,
      title: 'Comprehensive Color System',
      description: 'WCAG 2.1 AA compliant color palette with semantic tokens and theme support',
      stats: '50+ colors, 3 themes'
    },
    {
      icon: Type,
      title: 'Typography Scale',
      description: 'Harmonious type system with responsive scaling and accessibility features',
      stats: '15+ text styles'
    },
    {
      icon: Layout,
      title: 'Flexible Components',
      description: 'Atomic design methodology with 40+ reusable components',
      stats: '40+ components'
    },
    {
      icon: Accessibility,
      title: 'Accessibility First',
      description: 'Built-in ARIA support, keyboard navigation, and screen reader compatibility',
      stats: '100% WCAG AA'
    },
    {
      icon: Smartphone,
      title: 'Mobile Responsive',
      description: 'Mobile-first design with responsive breakpoints and touch-friendly interactions',
      stats: '5 breakpoints'
    },
    {
      icon: Zap,
      title: 'Performance Optimized',
      description: 'Lightweight components with optimized bundle size and fast rendering',
      stats: '<50kb gzipped'
    }
  ]

  const colorPalettes = [
    { name: 'Primary', colors: colors.primary, description: 'Brand colors for primary actions' },
    { name: 'Secondary', colors: colors.secondary, description: 'Neutral colors for secondary elements' },
    { name: 'Success', colors: colors.success, description: 'Positive feedback and success states' },
    { name: 'Warning', colors: colors.warning, description: 'Caution and warning messages' },
    { name: 'Error', colors: colors.error, description: 'Error states and destructive actions' }
  ]

  const typographyExamples = [
    { variant: 'display-lg', text: 'Display Large', description: 'Hero headings and major titles' },
    { variant: 'heading-xl', text: 'Heading XL', description: 'Section headings' },
    { variant: 'heading-lg', text: 'Heading Large', description: 'Subsection headings' },
    { variant: 'body-lg', text: 'Body Large', description: 'Large body text and introductions' },
    { variant: 'body-md', text: 'Body Medium', description: 'Default body text' },
    { variant: 'caption-md', text: 'Caption Medium', description: 'Labels and captions' }
  ]

  const componentCategories = [
    {
      name: 'Atoms',
      count: 15,
      description: 'Basic building blocks',
      examples: ['Button', 'Input', 'Typography', 'Icon']
    },
    {
      name: 'Molecules',
      count: 12,
      description: 'Simple component groups',
      examples: ['Card', 'Form Field', 'Modal', 'Navigation Item']
    },
    {
      name: 'Organisms',
      count: 8,
      description: 'Complex UI sections',
      examples: ['Header', 'Sidebar', 'Footer', 'Data Table']
    },
    {
      name: 'Templates',
      count: 5,
      description: 'Page-level layouts',
      examples: ['Dashboard', 'Settings', 'Profile', 'Landing']
    }
  ]

  return (
    <div className="space-y-12 p-6">
      {/* Hero Section */}
      <div className="text-center space-y-6">
        <div className="space-y-4">
          <H1 className="text-4xl lg:text-6xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            Nexus Architect Design System
          </H1>
          <Lead className="text-xl text-muted-foreground max-w-3xl mx-auto">
            A comprehensive, accessible, and scalable design system built for modern enterprise applications. 
            Featuring atomic design methodology, WCAG 2.1 AA compliance, and responsive components.
          </Lead>
        </div>
        
        <div className="flex flex-wrap justify-center gap-4">
          <Button size="lg" className="gap-2">
            <Download className="h-5 w-5" />
            Get Started
          </Button>
          <Button variant="outline" size="lg" className="gap-2">
            <ExternalLink className="h-5 w-5" />
            View Documentation
          </Button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-2xl mx-auto mt-12">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary">40+</div>
            <div className="text-sm text-muted-foreground">Components</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary">100%</div>
            <div className="text-sm text-muted-foreground">WCAG AA</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary">3</div>
            <div className="text-sm text-muted-foreground">Themes</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary">5</div>
            <div className="text-sm text-muted-foreground">Breakpoints</div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <H2>Key Features</H2>
          <P className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Built with modern development practices and enterprise requirements in mind
          </P>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <Card key={index} className="relative overflow-hidden group hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <div className="p-2 rounded-lg bg-primary/10 text-primary">
                      <Icon className="h-6 w-6" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{feature.title}</CardTitle>
                      <div className="text-sm text-primary font-medium">{feature.stats}</div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <P className="text-muted-foreground">{feature.description}</P>
                </CardContent>
                <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              </Card>
            )
          })}
        </div>
      </section>

      {/* Color System */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <H2>Color System</H2>
          <P className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Semantic color tokens with WCAG 2.1 AA compliance and theme support
          </P>
        </div>

        <div className="space-y-6">
          {colorPalettes.map((palette) => (
            <Card key={palette.name}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  {palette.name}
                  <Code inline className="text-xs">{Object.keys(palette.colors).length} shades</Code>
                </CardTitle>
                <CardDescription>{palette.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-5 md:grid-cols-10 gap-2">
                  {Object.entries(palette.colors).map(([shade, color]) => (
                    <div key={shade} className="space-y-2">
                      <div 
                        className="h-12 w-full rounded-lg border shadow-sm"
                        style={{ backgroundColor: color }}
                        title={`${palette.name} ${shade}: ${color}`}
                      />
                      <div className="text-center">
                        <div className="text-xs font-medium">{shade}</div>
                        <div className="text-xs text-muted-foreground font-mono">{color}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Typography */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <H2>Typography System</H2>
          <P className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Harmonious type scale with responsive sizing and accessibility features
          </P>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Type Scale</CardTitle>
            <CardDescription>
              Semantic typography styles with consistent spacing and hierarchy
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {typographyExamples.map((example) => (
              <div key={example.variant} className="flex items-center justify-between border-b pb-4 last:border-b-0">
                <div className="space-y-1">
                  <div className={`font-${example.variant.includes('display') ? 'bold' : example.variant.includes('heading') ? 'semibold' : 'normal'}`}>
                    <span className={`text-${example.variant.split('-')[1] || 'base'}`}>
                      {example.text}
                    </span>
                  </div>
                  <P className="text-sm text-muted-foreground">{example.description}</P>
                </div>
                <Code inline className="text-xs">{example.variant}</Code>
              </div>
            ))}
          </CardContent>
        </Card>
      </section>

      {/* Component Architecture */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <H2>Component Architecture</H2>
          <P className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Atomic design methodology for scalable and maintainable components
          </P>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {componentCategories.map((category, index) => (
            <Card key={category.name} className="text-center">
              <CardHeader>
                <div className="mx-auto w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold text-xl">
                  {category.count}
                </div>
                <CardTitle>{category.name}</CardTitle>
                <CardDescription>{category.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {category.examples.map((example) => (
                    <div key={example} className="text-sm text-muted-foreground">
                      {example}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Interactive Demo */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <H2>Interactive Components</H2>
          <P className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Try out the components with different variants and states
          </P>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Component Playground</CardTitle>
            <CardDescription>
              Explore different component variants and interactions
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-8">
            {/* Button Variants */}
            <div className="space-y-4">
              <H3>Buttons</H3>
              <div className="flex flex-wrap gap-3">
                <Button>Default</Button>
                <Button variant="secondary">Secondary</Button>
                <Button variant="outline">Outline</Button>
                <Button variant="ghost">Ghost</Button>
                <Button variant="destructive">Destructive</Button>
                <Button variant="success">Success</Button>
              </div>
            </div>

            {/* Input Variants */}
            <div className="space-y-4">
              <H3>Inputs</H3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
                <Input placeholder="Default input" />
                <Input placeholder="With icon" leftIcon={<Globe className="h-4 w-4" />} />
                <Input placeholder="Success state" success="Looks good!" />
                <Input placeholder="Error state" error="This field is required" />
              </div>
            </div>

            {/* Card Variants */}
            <div className="space-y-4">
              <H3>Cards</H3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      Success
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <P className="text-sm">Operation completed successfully.</P>
                  </CardContent>
                </Card>
                
                <Card variant="elevated">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Star className="h-5 w-5 text-yellow-500" />
                      Featured
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <P className="text-sm">This is an elevated card variant.</P>
                  </CardContent>
                </Card>
                
                <Card variant="outlined">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Heart className="h-5 w-5 text-red-500" />
                      Favorite
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <P className="text-sm">This is an outlined card variant.</P>
                  </CardContent>
                </Card>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Responsive Design */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <H2>Responsive Design</H2>
          <P className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Mobile-first approach with flexible breakpoints and adaptive layouts
          </P>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Breakpoint System</CardTitle>
            <CardDescription>
              Responsive breakpoints for different device sizes
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center space-y-3">
                <Smartphone className="h-12 w-12 mx-auto text-primary" />
                <div>
                  <div className="font-semibold">Mobile</div>
                  <div className="text-sm text-muted-foreground">0px - 640px</div>
                  <Code inline className="text-xs">xs, sm</Code>
                </div>
              </div>
              <div className="text-center space-y-3">
                <Tablet className="h-12 w-12 mx-auto text-primary" />
                <div>
                  <div className="font-semibold">Tablet</div>
                  <div className="text-sm text-muted-foreground">640px - 1024px</div>
                  <Code inline className="text-xs">md, lg</Code>
                </div>
              </div>
              <div className="text-center space-y-3">
                <Monitor className="h-12 w-12 mx-auto text-primary" />
                <div>
                  <div className="font-semibold">Desktop</div>
                  <div className="text-sm text-muted-foreground">1024px+</div>
                  <Code inline className="text-xs">xl, 2xl</Code>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Getting Started */}
      <section className="space-y-8">
        <Card className="bg-gradient-to-r from-primary/5 to-primary/10 border-primary/20">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">Ready to Get Started?</CardTitle>
            <CardDescription className="text-lg">
              Start building with the Nexus Architect Design System today
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center space-y-6">
            <div className="flex flex-wrap justify-center gap-4">
              <Button size="lg" className="gap-2">
                <Download className="h-5 w-5" />
                Install Package
              </Button>
              <Button variant="outline" size="lg" className="gap-2">
                <ExternalLink className="h-5 w-5" />
                View on GitHub
              </Button>
            </div>
            
            <div className="max-w-md mx-auto">
              <Code className="text-sm bg-background/50 p-4 rounded-lg">
                npm install @nexus-architect/design-system
              </Code>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  )
}

