import React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '../atoms/Button'
import { 
  Heart, 
  Github, 
  Twitter, 
  Linkedin, 
  Mail,
  ExternalLink,
  Shield,
  Zap,
  Globe,
  Clock,
  CheckCircle,
  AlertCircle,
  XCircle
} from 'lucide-react'

const Footer = React.forwardRef(({
  className,
  variant = 'default', // default, minimal, detailed
  showSystemStatus = true,
  showSocialLinks = true,
  showLegalLinks = true,
  systemStatus = {
    overall: 'operational', // operational, degraded, outage
    services: [
      { name: 'API', status: 'operational' },
      { name: 'Database', status: 'operational' },
      { name: 'CDN', status: 'degraded' },
      { name: 'Authentication', status: 'operational' }
    ],
    lastUpdated: new Date().toISOString()
  },
  ...props
}, ref) => {
  const currentYear = new Date().getFullYear()

  const socialLinks = [
    { name: 'GitHub', icon: Github, href: 'https://github.com', external: true },
    { name: 'Twitter', icon: Twitter, href: 'https://twitter.com', external: true },
    { name: 'LinkedIn', icon: Linkedin, href: 'https://linkedin.com', external: true },
    { name: 'Email', icon: Mail, href: 'mailto:contact@nexusarchitect.com', external: false }
  ]

  const legalLinks = [
    { name: 'Privacy Policy', href: '/privacy' },
    { name: 'Terms of Service', href: '/terms' },
    { name: 'Cookie Policy', href: '/cookies' },
    { name: 'Accessibility', href: '/accessibility' }
  ]

  const quickLinks = [
    { name: 'Documentation', href: '/docs' },
    { name: 'API Reference', href: '/api' },
    { name: 'Support', href: '/support' },
    { name: 'Status Page', href: '/status' }
  ]

  const getStatusIcon = (status) => {
    switch (status) {
      case 'operational':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'degraded':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />
      case 'outage':
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return <CheckCircle className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusText = (status) => {
    switch (status) {
      case 'operational':
        return 'All systems operational'
      case 'degraded':
        return 'Some systems degraded'
      case 'outage':
        return 'System outage detected'
      default:
        return 'Status unknown'
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'operational':
        return 'text-green-600'
      case 'degraded':
        return 'text-yellow-600'
      case 'outage':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  if (variant === 'minimal') {
    return (
      <footer
        ref={ref}
        className={cn(
          "border-t bg-background py-4",
          className
        )}
        {...props}
      >
        <div className="container px-4">
          <div className="flex flex-col sm:flex-row items-center justify-between space-y-2 sm:space-y-0">
            <div className="flex items-center space-x-2 text-sm text-muted-foreground">
              <span>© {currentYear} Nexus Architect</span>
              <span>•</span>
              <span>Made with</span>
              <Heart className="h-4 w-4 text-red-500" />
            </div>
            
            {showSystemStatus && (
              <div className="flex items-center space-x-2 text-sm">
                {getStatusIcon(systemStatus.overall)}
                <span className={getStatusColor(systemStatus.overall)}>
                  {getStatusText(systemStatus.overall)}
                </span>
              </div>
            )}
          </div>
        </div>
      </footer>
    )
  }

  if (variant === 'detailed') {
    return (
      <footer
        ref={ref}
        className={cn(
          "border-t bg-background",
          className
        )}
        {...props}
      >
        {/* System Status Bar */}
        {showSystemStatus && (
          <div className="border-b bg-muted/30 py-3">
            <div className="container px-4">
              <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between space-y-3 lg:space-y-0">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(systemStatus.overall)}
                  <div>
                    <div className={cn("font-medium", getStatusColor(systemStatus.overall))}>
                      {getStatusText(systemStatus.overall)}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Last updated: {new Date(systemStatus.lastUpdated).toLocaleString()}
                    </div>
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-4">
                  {systemStatus.services.map((service) => (
                    <div key={service.name} className="flex items-center space-x-2">
                      {getStatusIcon(service.status)}
                      <span className="text-sm">{service.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Main Footer Content */}
        <div className="py-12">
          <div className="container px-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {/* Company Info */}
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
                    <span className="text-primary-foreground font-bold text-sm">N</span>
                  </div>
                  <span className="font-bold text-lg">Nexus Architect</span>
                </div>
                <p className="text-sm text-muted-foreground">
                  Building the future of autonomous software development with AI-powered 
                  architecture and intelligent automation.
                </p>
                <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                  <Globe className="h-4 w-4" />
                  <span>Global • 24/7 Support</span>
                </div>
              </div>

              {/* Quick Links */}
              <div className="space-y-4">
                <h3 className="font-semibold">Resources</h3>
                <ul className="space-y-2">
                  {quickLinks.map((link) => (
                    <li key={link.name}>
                      <a
                        href={link.href}
                        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {link.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Features */}
              <div className="space-y-4">
                <h3 className="font-semibold">Features</h3>
                <ul className="space-y-2">
                  <li className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <Zap className="h-4 w-4" />
                    <span>AI-Powered Development</span>
                  </li>
                  <li className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <Shield className="h-4 w-4" />
                    <span>Enterprise Security</span>
                  </li>
                  <li className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <CheckCircle className="h-4 w-4" />
                    <span>99.9% Uptime SLA</span>
                  </li>
                  <li className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    <span>Real-time Monitoring</span>
                  </li>
                </ul>
              </div>

              {/* Contact & Social */}
              <div className="space-y-4">
                <h3 className="font-semibold">Connect</h3>
                {showSocialLinks && (
                  <div className="flex space-x-2">
                    {socialLinks.map((social) => {
                      const Icon = social.icon
                      return (
                        <Button
                          key={social.name}
                          variant="ghost"
                          size="icon"
                          asChild
                          className="h-8 w-8"
                        >
                          <a
                            href={social.href}
                            aria-label={social.name}
                            {...(social.external && {
                              target: "_blank",
                              rel: "noopener noreferrer"
                            })}
                          >
                            <Icon className="h-4 w-4" />
                          </a>
                        </Button>
                      )
                    })}
                  </div>
                )}
                <div className="text-sm text-muted-foreground">
                  <p>Need help? Contact our support team</p>
                  <a 
                    href="mailto:support@nexusarchitect.com"
                    className="text-primary hover:underline"
                  >
                    support@nexusarchitect.com
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t py-6">
          <div className="container px-4">
            <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
              <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                <span>© {currentYear} Nexus Architect. All rights reserved.</span>
                <span className="hidden md:inline">•</span>
                <span className="hidden md:inline">Version 1.0.0</span>
              </div>
              
              {showLegalLinks && (
                <div className="flex flex-wrap items-center space-x-4 text-sm">
                  {legalLinks.map((link, index) => (
                    <React.Fragment key={link.name}>
                      <a
                        href={link.href}
                        className="text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {link.name}
                      </a>
                      {index < legalLinks.length - 1 && (
                        <span className="text-muted-foreground">•</span>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </footer>
    )
  }

  // Default variant
  return (
    <footer
      ref={ref}
      className={cn(
        "border-t bg-background py-8",
        className
      )}
      {...props}
    >
      <div className="container px-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Company Info */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="h-6 w-6 rounded bg-primary flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-xs">N</span>
              </div>
              <span className="font-semibold">Nexus Architect</span>
            </div>
            <p className="text-sm text-muted-foreground">
              AI-powered software development platform for the modern enterprise.
            </p>
          </div>

          {/* Links */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="font-medium text-sm">Product</h4>
              <ul className="space-y-1">
                {quickLinks.slice(0, 2).map((link) => (
                  <li key={link.name}>
                    <a
                      href={link.href}
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-sm">Support</h4>
              <ul className="space-y-1">
                {quickLinks.slice(2).map((link) => (
                  <li key={link.name}>
                    <a
                      href={link.href}
                      className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                    >
                      {link.name}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Status & Social */}
          <div className="space-y-4">
            {showSystemStatus && (
              <div className="flex items-center space-x-2">
                {getStatusIcon(systemStatus.overall)}
                <span className={cn("text-sm", getStatusColor(systemStatus.overall))}>
                  {getStatusText(systemStatus.overall)}
                </span>
              </div>
            )}
            
            {showSocialLinks && (
              <div className="flex space-x-2">
                {socialLinks.map((social) => {
                  const Icon = social.icon
                  return (
                    <Button
                      key={social.name}
                      variant="ghost"
                      size="icon"
                      asChild
                      className="h-8 w-8"
                    >
                      <a
                        href={social.href}
                        aria-label={social.name}
                        {...(social.external && {
                          target: "_blank",
                          rel: "noopener noreferrer"
                        })}
                      >
                        <Icon className="h-4 w-4" />
                      </a>
                    </Button>
                  )
                })}
              </div>
            )}
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-8 pt-6 border-t flex flex-col sm:flex-row items-center justify-between space-y-2 sm:space-y-0">
          <div className="text-sm text-muted-foreground">
            © {currentYear} Nexus Architect. All rights reserved.
          </div>
          
          {showLegalLinks && (
            <div className="flex space-x-4 text-sm">
              {legalLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  className="text-muted-foreground hover:text-foreground transition-colors"
                >
                  {link.name}
                </a>
              ))}
            </div>
          )}
        </div>
      </div>
    </footer>
  )
})

Footer.displayName = "Footer"

export { Footer }

