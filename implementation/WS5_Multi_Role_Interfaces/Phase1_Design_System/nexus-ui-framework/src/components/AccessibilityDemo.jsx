import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './molecules/Card'
import { Button } from './atoms/Button'
import { Input, InputGroup } from './atoms/Input'
import { H1, H2, H3, P, Code } from './atoms/Typography'
import { Form, FormField, SubmitButton, validators } from './molecules/Form'
import { 
  useFocusTrap, 
  useKeyboardNavigation, 
  useScreenReader, 
  useReducedMotion,
  useHighContrast,
  useLiveRegion 
} from '../hooks/useAccessibility'
import { 
  meetsContrastRequirement, 
  KEYBOARD_KEYS,
  announceToScreenReader 
} from '../utils/accessibility'
import { 
  Eye, 
  EyeOff, 
  Volume2, 
  VolumeX, 
  Contrast, 
  Accessibility,
  Keyboard,
  MousePointer,
  Monitor
} from 'lucide-react'

export const AccessibilityDemo = () => {
  const [focusTrapActive, setFocusTrapActive] = React.useState(false)
  const [announcements, setAnnouncements] = React.useState([])
  const focusTrapRef = useFocusTrap(focusTrapActive)
  const { announce, AnnouncementRegion } = useScreenReader()
  const { announcePolite, announceAssertive, LiveRegions } = useLiveRegion()
  const prefersReducedMotion = useReducedMotion()
  const prefersHighContrast = useHighContrast()

  // Keyboard navigation demo
  const [selectedItem, setSelectedItem] = React.useState(0)
  const navigationItems = ['Home', 'About', 'Services', 'Contact']
  
  const keyboardNav = useKeyboardNavigation({
    onArrowDown: () => setSelectedItem(prev => (prev + 1) % navigationItems.length),
    onArrowUp: () => setSelectedItem(prev => (prev - 1 + navigationItems.length) % navigationItems.length),
    onEnter: () => announcePolite(`Selected ${navigationItems[selectedItem]}`),
    onEscape: () => setSelectedItem(0)
  })

  // Form validation demo
  const formValidation = {
    email: [validators.required, validators.email],
    password: [validators.required, validators.minLength(8)]
  }

  const handleFormSubmit = (values) => {
    announceAssertive('Form submitted successfully!')
    console.log('Form values:', values)
  }

  // Color contrast examples
  const contrastExamples = [
    { bg: '#ffffff', fg: '#000000', label: 'Black on White' },
    { bg: '#000000', fg: '#ffffff', label: 'White on Black' },
    { bg: '#0066cc', fg: '#ffffff', label: 'White on Blue' },
    { bg: '#ffff00', fg: '#000000', label: 'Black on Yellow' },
    { bg: '#ff0000', fg: '#ffffff', label: 'White on Red' },
    { bg: '#cccccc', fg: '#666666', label: 'Gray on Light Gray (Poor)' }
  ]

  return (
    <div className="space-y-8 p-6">
      <AnnouncementRegion />
      <LiveRegions />
      
      {/* Header */}
      <div>
        <H1>Accessibility Implementation & Testing</H1>
        <P className="text-lg text-muted-foreground mt-2">
          Comprehensive WCAG 2.1 AA compliance demonstration with interactive examples
        </P>
      </div>

      {/* Accessibility Preferences */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Monitor className="h-5 w-5" />
            User Preferences Detection
          </CardTitle>
          <CardDescription>
            Automatically detected accessibility preferences from system settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <span>Prefers Reduced Motion</span>
              <span className={`px-2 py-1 rounded text-sm ${
                prefersReducedMotion ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {prefersReducedMotion ? 'Enabled' : 'Disabled'}
              </span>
            </div>
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <span>Prefers High Contrast</span>
              <span className={`px-2 py-1 rounded text-sm ${
                prefersHighContrast ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {prefersHighContrast ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Focus Management */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Focus Management & Keyboard Navigation
          </CardTitle>
          <CardDescription>
            Demonstrates focus trapping, keyboard navigation, and ARIA support
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Focus Trap Demo */}
          <div className="space-y-4">
            <H3>Focus Trap Example</H3>
            <Button 
              onClick={() => setFocusTrapActive(true)}
              aria-describedby="focus-trap-description"
            >
              Activate Focus Trap
            </Button>
            <P id="focus-trap-description" className="text-sm text-muted-foreground">
              Click to activate focus trap. Use Tab/Shift+Tab to navigate, Escape to close.
            </P>
            
            {focusTrapActive && (
              <div 
                ref={focusTrapRef}
                className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
                role="dialog"
                aria-modal="true"
                aria-labelledby="focus-trap-title"
              >
                <div className="bg-white p-6 rounded-lg max-w-md w-full mx-4">
                  <H3 id="focus-trap-title">Focus Trapped Modal</H3>
                  <P className="mt-2 mb-4">
                    Focus is trapped within this modal. Try tabbing through the elements.
                  </P>
                  <div className="space-y-3">
                    <Input placeholder="First input" />
                    <Input placeholder="Second input" />
                    <div className="flex gap-2">
                      <Button variant="outline">Cancel</Button>
                      <Button onClick={() => setFocusTrapActive(false)}>
                        Close Modal
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Keyboard Navigation Demo */}
          <div className="space-y-4">
            <H3>Keyboard Navigation (Arrow Keys)</H3>
            <div 
              className="border rounded-lg p-4 focus-within:ring-2 focus-within:ring-ring"
              tabIndex={0}
              {...keyboardNav}
              role="listbox"
              aria-label="Navigation menu"
              aria-activedescendant={`nav-item-${selectedItem}`}
            >
              <P className="text-sm text-muted-foreground mb-3">
                Focus this area and use arrow keys to navigate. Press Enter to select.
              </P>
              <div className="space-y-1">
                {navigationItems.map((item, index) => (
                  <div
                    key={item}
                    id={`nav-item-${index}`}
                    className={`p-2 rounded transition-colors ${
                      index === selectedItem 
                        ? 'bg-primary text-primary-foreground' 
                        : 'hover:bg-muted'
                    }`}
                    role="option"
                    aria-selected={index === selectedItem}
                  >
                    {item}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Screen Reader Support */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Volume2 className="h-5 w-5" />
            Screen Reader Support
          </CardTitle>
          <CardDescription>
            Live regions, ARIA labels, and screen reader announcements
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2 flex-wrap">
            <Button 
              onClick={() => announcePolite('This is a polite announcement')}
              variant="outline"
            >
              Polite Announcement
            </Button>
            <Button 
              onClick={() => announceAssertive('This is an assertive announcement!')}
              variant="outline"
            >
              Assertive Announcement
            </Button>
            <Button 
              onClick={() => announceToScreenReader('Direct screen reader announcement')}
              variant="outline"
            >
              Direct Announcement
            </Button>
          </div>
          
          <div className="p-4 border rounded-lg bg-muted/50">
            <H3>ARIA Labels Example</H3>
            <div className="mt-3 space-y-2">
              <Button 
                aria-label="Save document to cloud storage"
                aria-describedby="save-description"
              >
                Save
              </Button>
              <P id="save-description" className="text-sm text-muted-foreground">
                This button has a descriptive ARIA label for screen readers
              </P>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Color Contrast Testing */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Contrast className="h-5 w-5" />
            Color Contrast Compliance
          </CardTitle>
          <CardDescription>
            WCAG 2.1 AA color contrast ratio testing and examples
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {contrastExamples.map(({ bg, fg, label }, index) => {
              const ratio = meetsContrastRequirement(fg, bg, 'AA', 'normal')
              const contrastRatio = ((Math.max(0.2126, 0.7152) + 0.05) / (Math.min(0.2126, 0.7152) + 0.05)).toFixed(2)
              
              return (
                <div
                  key={index}
                  className="p-4 rounded-lg border"
                  style={{ backgroundColor: bg, color: fg }}
                >
                  <div className="font-medium">{label}</div>
                  <div className="text-sm mt-1">
                    Ratio: {contrastRatio}:1
                  </div>
                  <div className={`text-xs mt-1 px-2 py-1 rounded ${
                    ratio ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'
                  }`}>
                    {ratio ? 'WCAG AA ✓' : 'WCAG AA ✗'}
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Form Accessibility */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Keyboard className="h-5 w-5" />
            Accessible Forms
          </CardTitle>
          <CardDescription>
            Form validation, error handling, and accessibility features
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Form 
            onSubmit={handleFormSubmit}
            validationSchema={formValidation}
            className="max-w-md"
          >
            <FormField
              name="email"
              label="Email Address"
              description="We'll never share your email with anyone else"
              required
            >
              <Input 
                type="email" 
                placeholder="Enter your email"
                autoComplete="email"
              />
            </FormField>

            <FormField
              name="password"
              label="Password"
              description="Must be at least 8 characters long"
              required
            >
              <Input 
                type="password" 
                placeholder="Enter your password"
                autoComplete="new-password"
              />
            </FormField>

            <div className="flex gap-2">
              <SubmitButton>Submit Form</SubmitButton>
              <Button type="button" variant="outline">
                Cancel
              </Button>
            </div>
          </Form>
        </CardContent>
      </Card>

      {/* Skip Links Demo */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MousePointer className="h-5 w-5" />
            Skip Links & Landmarks
          </CardTitle>
          <CardDescription>
            Navigation aids for keyboard and screen reader users
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-4 border rounded-lg">
            <H3>Skip Links (Focus to reveal)</H3>
            <P className="text-sm text-muted-foreground mb-3">
              Tab to the area below to see skip links appear
            </P>
            <div className="relative border-2 border-dashed border-muted p-4 rounded">
              <a 
                href="#main-content"
                className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:bg-primary focus:text-primary-foreground focus:px-3 focus:py-1 focus:rounded focus:z-10"
              >
                Skip to main content
              </a>
              <a 
                href="#navigation"
                className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-40 focus:bg-primary focus:text-primary-foreground focus:px-3 focus:py-1 focus:rounded focus:z-10"
              >
                Skip to navigation
              </a>
              <P>Focus this area to reveal skip links</P>
            </div>
          </div>

          <div className="space-y-2">
            <H3>Semantic Landmarks</H3>
            <Code inline>
              &lt;main role="main"&gt;, &lt;nav role="navigation"&gt;, &lt;aside role="complementary"&gt;
            </Code>
            <P className="text-sm text-muted-foreground">
              Proper landmark roles help screen reader users navigate page structure
            </P>
          </div>
        </CardContent>
      </Card>

      {/* Accessibility Testing Tools */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Accessibility className="h-5 w-5" />
            Testing & Validation
          </CardTitle>
          <CardDescription>
            Built-in accessibility testing and validation tools
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 border rounded-lg">
              <H3 className="text-base mb-2">Automated Checks</H3>
              <ul className="text-sm space-y-1 text-muted-foreground">
                <li>✓ Color contrast ratios</li>
                <li>✓ ARIA label presence</li>
                <li>✓ Keyboard accessibility</li>
                <li>✓ Heading structure</li>
                <li>✓ Focus management</li>
              </ul>
            </div>
            <div className="p-4 border rounded-lg">
              <H3 className="text-base mb-2">Manual Testing</H3>
              <ul className="text-sm space-y-1 text-muted-foreground">
                <li>• Screen reader testing</li>
                <li>• Keyboard-only navigation</li>
                <li>• High contrast mode</li>
                <li>• Zoom to 200% testing</li>
                <li>• Voice control testing</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

