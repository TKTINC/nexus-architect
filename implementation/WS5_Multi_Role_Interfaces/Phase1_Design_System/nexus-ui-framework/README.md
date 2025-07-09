# Nexus Architect Design System

A comprehensive, accessible, and scalable design system built for modern enterprise applications. Features atomic design methodology, WCAG 2.1 AA compliance, and responsive components.

## 🚀 Features

- **Comprehensive Component Library**: 40+ components following atomic design principles
- **Accessibility First**: WCAG 2.1 AA compliant with built-in ARIA support
- **Responsive Design**: Mobile-first approach with 5 breakpoints
- **Theme Support**: Light, dark, and high contrast themes
- **TypeScript Ready**: Full TypeScript support with type definitions
- **Performance Optimized**: Lightweight components with tree-shaking support

## 📦 Installation

```bash
npm install @nexus-architect/design-system
# or
yarn add @nexus-architect/design-system
```

## 🎨 Design Tokens

### Colors
- **Primary**: Brand colors for primary actions
- **Secondary**: Neutral colors for secondary elements  
- **Success**: Positive feedback and success states
- **Warning**: Caution and warning messages
- **Error**: Error states and destructive actions

### Typography
- **Display**: Hero headings and major titles
- **Heading**: Section and subsection headings
- **Body**: Default body text and content
- **Caption**: Labels, captions, and metadata

### Spacing
- Consistent 8px grid system
- Responsive spacing tokens
- Semantic spacing names

## 🧩 Component Architecture

### Atoms (15 components)
Basic building blocks that can't be broken down further:
- Button
- Input
- Typography (H1-H6, P, Code, etc.)
- Icon
- Badge
- Avatar
- Spinner

### Molecules (12 components)
Simple groups of atoms functioning together:
- Card
- Form Field
- Modal
- Navigation Item
- Search Input
- Input Group
- Alert
- Tooltip

### Organisms (8 components)
Complex UI sections made of molecules and atoms:
- Header
- Sidebar
- Footer
- Data Table
- Navigation Menu
- Form
- Dashboard Widget
- Content Layout

### Templates (5 components)
Page-level layouts combining organisms:
- Dashboard Layout
- Settings Layout
- Profile Layout
- Landing Layout
- Error Layout

## 🎯 Usage Examples

### Basic Components

```jsx
import { Button, Input, Card } from '@nexus-architect/design-system'

function MyComponent() {
  return (
    <Card>
      <Card.Header>
        <Card.Title>Welcome</Card.Title>
      </Card.Header>
      <Card.Content>
        <Input placeholder="Enter your name" />
        <Button>Submit</Button>
      </Card.Content>
    </Card>
  )
}
```

### Form with Validation

```jsx
import { Form, FormField, Input, Button } from '@nexus-architect/design-system'

function ContactForm() {
  const handleSubmit = (values) => {
    console.log(values)
  }

  return (
    <Form onSubmit={handleSubmit}>
      <FormField 
        name="email" 
        label="Email" 
        required
        validation={[validators.required, validators.email]}
      >
        <Input type="email" placeholder="Enter your email" />
      </FormField>
      
      <FormField 
        name="message" 
        label="Message" 
        required
      >
        <textarea placeholder="Your message" />
      </FormField>
      
      <Button type="submit">Send Message</Button>
    </Form>
  )
}
```

### Layout Components

```jsx
import { Header, Sidebar, Footer } from '@nexus-architect/design-system'

function AppLayout({ children }) {
  return (
    <div className="min-h-screen flex flex-col">
      <Header 
        user={currentUser}
        onMenuToggle={toggleSidebar}
      />
      
      <div className="flex flex-1">
        <Sidebar 
          isOpen={sidebarOpen}
          userRole="admin"
          onClose={closeSidebar}
        />
        
        <main className="flex-1 p-6">
          {children}
        </main>
      </div>
      
      <Footer variant="default" />
    </div>
  )
}
```

## ♿ Accessibility

### WCAG 2.1 AA Compliance
- Color contrast ratios meet AA standards
- All interactive elements are keyboard accessible
- Screen reader support with proper ARIA labels
- Focus management and visual indicators

### Keyboard Navigation
- Tab order follows logical flow
- Arrow key navigation for complex components
- Escape key closes modals and dropdowns
- Enter/Space activates buttons and links

### Screen Reader Support
- Semantic HTML structure
- ARIA labels and descriptions
- Live regions for dynamic content
- Skip links for main content areas

## 🎨 Theming

### Built-in Themes
- **Light**: Default light theme
- **Dark**: Dark mode with proper contrast
- **High Contrast**: Enhanced contrast for accessibility

### Custom Themes
```jsx
import { ThemeProvider } from '@nexus-architect/design-system'

const customTheme = {
  colors: {
    primary: {
      50: '#f0f9ff',
      500: '#3b82f6',
      900: '#1e3a8a'
    }
  }
}

function App() {
  return (
    <ThemeProvider theme={customTheme}>
      <YourApp />
    </ThemeProvider>
  )
}
```

## 📱 Responsive Design

### Breakpoints
- **xs**: 0px - 640px (Mobile)
- **sm**: 640px - 768px (Large Mobile)
- **md**: 768px - 1024px (Tablet)
- **lg**: 1024px - 1280px (Desktop)
- **xl**: 1280px+ (Large Desktop)

### Responsive Utilities
```jsx
// Responsive props
<Button size={{ xs: 'sm', md: 'md', lg: 'lg' }}>
  Responsive Button
</Button>

// Responsive classes
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
  Responsive Grid
</div>
```

## 🔧 Development

### Prerequisites
- Node.js 18+
- npm or yarn

### Setup
```bash
# Clone the repository
git clone https://github.com/TKTINC/nexus-architect.git

# Navigate to design system
cd nexus-architect/implementation/WS5_Multi_Role_Interfaces/Phase1_Design_System/nexus-ui-framework

# Install dependencies
npm install

# Start development server
npm run dev
```

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run test` - Run tests
- `npm run lint` - Lint code
- `npm run storybook` - Start Storybook

### Project Structure
```
src/
├── components/
│   ├── atoms/          # Basic components
│   ├── molecules/      # Composite components
│   ├── organisms/      # Complex sections
│   ├── layout/         # Layout components
│   └── design-tokens/  # Design tokens
├── hooks/              # Custom hooks
├── utils/              # Utility functions
└── types/              # TypeScript types
```

## 📚 Documentation

### Component Documentation
Each component includes:
- Props API documentation
- Usage examples
- Accessibility guidelines
- Design guidelines

### Storybook
Interactive component documentation available at:
```bash
npm run storybook
```

## 🧪 Testing

### Accessibility Testing
- Automated WCAG compliance checks
- Keyboard navigation testing
- Screen reader compatibility
- Color contrast validation

### Unit Testing
```bash
npm run test
```

### Visual Regression Testing
```bash
npm run test:visual
```

## 🤝 Contributing

### Guidelines
1. Follow atomic design principles
2. Ensure WCAG 2.1 AA compliance
3. Include comprehensive tests
4. Update documentation
5. Follow coding standards

### Pull Request Process
1. Create feature branch
2. Implement changes
3. Add/update tests
4. Update documentation
5. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Design System Docs](https://nexus-architect.dev/design-system)
- **Issues**: [GitHub Issues](https://github.com/TKTINC/nexus-architect/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TKTINC/nexus-architect/discussions)
- **Email**: design-system@nexusarchitect.com

## 🗺️ Roadmap

### Phase 1 ✅ (Current)
- Core component library
- Design tokens
- Accessibility implementation
- Basic theming

### Phase 2 🚧 (Next)
- Advanced animations
- Data visualization components
- Form builder
- Advanced theming

### Phase 3 📋 (Future)
- Component variants
- Advanced layouts
- Micro-interactions
- Performance optimizations

---

Built with ❤️ by the Nexus Architect team

