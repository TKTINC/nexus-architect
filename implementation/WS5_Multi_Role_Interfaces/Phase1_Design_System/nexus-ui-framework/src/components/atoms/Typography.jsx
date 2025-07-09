import React from 'react'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'

// Typography variants
const typographyVariants = cva("", {
  variants: {
    variant: {
      // Display styles
      'display-2xl': 'text-6xl font-bold tracking-tight lg:text-7xl',
      'display-xl': 'text-5xl font-bold tracking-tight lg:text-6xl',
      'display-lg': 'text-4xl font-bold tracking-tight lg:text-5xl',
      'display-md': 'text-3xl font-bold tracking-tight lg:text-4xl',
      'display-sm': 'text-2xl font-bold tracking-tight lg:text-3xl',
      
      // Heading styles
      'heading-xl': 'text-xl font-semibold tracking-tight lg:text-2xl',
      'heading-lg': 'text-lg font-semibold tracking-tight lg:text-xl',
      'heading-md': 'text-base font-semibold tracking-tight lg:text-lg',
      'heading-sm': 'text-sm font-semibold tracking-tight',
      'heading-xs': 'text-xs font-semibold tracking-tight',
      
      // Body text styles
      'body-xl': 'text-xl leading-relaxed',
      'body-lg': 'text-lg leading-relaxed',
      'body-md': 'text-base leading-relaxed',
      'body-sm': 'text-sm leading-normal',
      'body-xs': 'text-xs leading-normal',
      
      // Caption and label styles
      'caption-lg': 'text-sm font-medium leading-tight',
      'caption-md': 'text-xs font-medium leading-tight',
      'caption-sm': 'text-xs font-medium leading-tight',
      
      // Code styles
      'code-lg': 'text-sm font-mono leading-normal',
      'code-md': 'text-xs font-mono leading-normal',
      'code-sm': 'text-xs font-mono leading-normal',
      
      // Utility styles
      'lead': 'text-xl text-muted-foreground leading-relaxed',
      'large': 'text-lg font-semibold',
      'small': 'text-sm font-medium leading-none',
      'muted': 'text-sm text-muted-foreground',
      'subtle': 'text-xs text-muted-foreground'
    },
    color: {
      default: 'text-foreground',
      muted: 'text-muted-foreground',
      primary: 'text-primary',
      secondary: 'text-secondary-foreground',
      destructive: 'text-destructive',
      success: 'text-green-600',
      warning: 'text-yellow-600',
      info: 'text-blue-600'
    },
    align: {
      left: 'text-left',
      center: 'text-center',
      right: 'text-right',
      justify: 'text-justify'
    },
    weight: {
      thin: 'font-thin',
      light: 'font-light',
      normal: 'font-normal',
      medium: 'font-medium',
      semibold: 'font-semibold',
      bold: 'font-bold',
      extrabold: 'font-extrabold'
    }
  },
  defaultVariants: {
    variant: 'body-md',
    color: 'default',
    align: 'left'
  }
})

// Base Typography component
const Typography = React.forwardRef(({
  className,
  variant,
  color,
  align,
  weight,
  as: Component = 'p',
  children,
  ...props
}, ref) => {
  return (
    <Component
      ref={ref}
      className={cn(typographyVariants({ variant, color, align, weight }), className)}
      {...props}
    >
      {children}
    </Component>
  )
})

Typography.displayName = "Typography"

// Heading components
const H1 = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="h1"
    variant="display-lg"
    className={cn("scroll-m-20", className)}
    {...props}
  />
))
H1.displayName = "H1"

const H2 = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="h2"
    variant="display-md"
    className={cn("scroll-m-20 border-b pb-2 first:mt-0", className)}
    {...props}
  />
))
H2.displayName = "H2"

const H3 = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="h3"
    variant="display-sm"
    className={cn("scroll-m-20", className)}
    {...props}
  />
))
H3.displayName = "H3"

const H4 = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="h4"
    variant="heading-xl"
    className={cn("scroll-m-20", className)}
    {...props}
  />
))
H4.displayName = "H4"

const H5 = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="h5"
    variant="heading-lg"
    className={cn("scroll-m-20", className)}
    {...props}
  />
))
H5.displayName = "H5"

const H6 = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="h6"
    variant="heading-md"
    className={cn("scroll-m-20", className)}
    {...props}
  />
))
H6.displayName = "H6"

// Paragraph component
const P = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="p"
    variant="body-md"
    className={cn("leading-7 [&:not(:first-child)]:mt-6", className)}
    {...props}
  />
))
P.displayName = "P"

// Lead paragraph
const Lead = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="p"
    variant="lead"
    className={className}
    {...props}
  />
))
Lead.displayName = "Lead"

// Large text
const Large = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="div"
    variant="large"
    className={className}
    {...props}
  />
))
Large.displayName = "Large"

// Small text
const Small = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="small"
    variant="small"
    className={className}
    {...props}
  />
))
Small.displayName = "Small"

// Muted text
const Muted = React.forwardRef(({ className, ...props }, ref) => (
  <Typography
    ref={ref}
    as="p"
    variant="muted"
    className={className}
    {...props}
  />
))
Muted.displayName = "Muted"

// Code component
const Code = React.forwardRef(({ className, inline = false, ...props }, ref) => (
  <Typography
    ref={ref}
    as={inline ? "code" : "pre"}
    variant="code-md"
    className={cn(
      inline 
        ? "relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold"
        : "mb-4 mt-6 overflow-x-auto rounded-lg border bg-muted p-4",
      className
    )}
    {...props}
  />
))
Code.displayName = "Code"

// Blockquote component
const Blockquote = React.forwardRef(({ className, ...props }, ref) => (
  <blockquote
    ref={ref}
    className={cn("mt-6 border-l-2 pl-6 italic", className)}
    {...props}
  />
))
Blockquote.displayName = "Blockquote"

// List components
const List = React.forwardRef(({ className, ordered = false, ...props }, ref) => (
  <Typography
    ref={ref}
    as={ordered ? "ol" : "ul"}
    className={cn(
      "my-6 ml-6",
      ordered ? "list-decimal" : "list-disc",
      "[&>li]:mt-2",
      className
    )}
    {...props}
  />
))
List.displayName = "List"

const ListItem = React.forwardRef(({ className, ...props }, ref) => (
  <li ref={ref} className={className} {...props} />
))
ListItem.displayName = "ListItem"

// Link component
const Link = React.forwardRef(({ 
  className, 
  variant = "default",
  external = false,
  children,
  ...props 
}, ref) => {
  const linkVariants = cva(
    "font-medium underline underline-offset-4 transition-colors",
    {
      variants: {
        variant: {
          default: "text-primary hover:text-primary/80",
          muted: "text-muted-foreground hover:text-foreground",
          destructive: "text-destructive hover:text-destructive/80"
        }
      },
      defaultVariants: {
        variant: "default"
      }
    }
  )

  return (
    <a
      ref={ref}
      className={cn(linkVariants({ variant }), className)}
      {...(external && { target: "_blank", rel: "noopener noreferrer" })}
      {...props}
    >
      {children}
      {external && (
        <span className="ml-1 inline-block" aria-hidden="true">
          ↗
        </span>
      )}
    </a>
  )
})
Link.displayName = "Link"

// Text highlight component
const Highlight = React.forwardRef(({ className, variant = "default", ...props }, ref) => {
  const highlightVariants = cva("px-1 py-0.5 rounded text-sm font-medium", {
    variants: {
      variant: {
        default: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
        primary: "bg-primary/10 text-primary",
        secondary: "bg-secondary text-secondary-foreground",
        success: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
        warning: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200",
        destructive: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
      }
    },
    defaultVariants: {
      variant: "default"
    }
  })

  return (
    <mark
      ref={ref}
      className={cn(highlightVariants({ variant }), className)}
      {...props}
    />
  )
})
Highlight.displayName = "Highlight"

// Keyboard key component
const Kbd = React.forwardRef(({ className, ...props }, ref) => (
  <kbd
    ref={ref}
    className={cn(
      "pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground opacity-100",
      className
    )}
    {...props}
  />
))
Kbd.displayName = "Kbd"

export {
  Typography,
  H1,
  H2,
  H3,
  H4,
  H5,
  H6,
  P,
  Lead,
  Large,
  Small,
  Muted,
  Code,
  Blockquote,
  List,
  ListItem,
  Link,
  Highlight,
  Kbd,
  typographyVariants
}

