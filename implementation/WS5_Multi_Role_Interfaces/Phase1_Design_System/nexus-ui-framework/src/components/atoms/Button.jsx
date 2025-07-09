import React from 'react'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'
import { Loader2 } from 'lucide-react'

// Button variants using class-variance-authority
const buttonVariants = cva(
  // Base styles
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
        success: "bg-green-600 text-white hover:bg-green-700",
        warning: "bg-yellow-600 text-white hover:bg-yellow-700",
        info: "bg-blue-600 text-white hover:bg-blue-700"
      },
      size: {
        default: "h-10 px-4 py-2",
        xs: "h-7 px-2 text-xs",
        sm: "h-9 px-3",
        lg: "h-11 px-8",
        xl: "h-12 px-10 text-base",
        icon: "h-10 w-10"
      },
      fullWidth: {
        true: "w-full",
        false: "w-auto"
      }
    },
    defaultVariants: {
      variant: "default",
      size: "default",
      fullWidth: false
    }
  }
)

const Button = React.forwardRef(({
  className,
  variant,
  size,
  fullWidth,
  loading = false,
  loadingText = "Loading...",
  leftIcon,
  rightIcon,
  children,
  disabled,
  ...props
}, ref) => {
  const isDisabled = disabled || loading

  return (
    <button
      className={cn(buttonVariants({ variant, size, fullWidth, className }))}
      ref={ref}
      disabled={isDisabled}
      aria-disabled={isDisabled}
      {...props}
    >
      {loading && (
        <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
      )}
      {!loading && leftIcon && (
        <span className="mr-2" aria-hidden="true">
          {leftIcon}
        </span>
      )}
      <span>
        {loading ? loadingText : children}
      </span>
      {!loading && rightIcon && (
        <span className="ml-2" aria-hidden="true">
          {rightIcon}
        </span>
      )}
    </button>
  )
})

Button.displayName = "Button"

// Button group component for related actions
const ButtonGroup = React.forwardRef(({
  className,
  orientation = "horizontal",
  size = "default",
  variant = "default",
  children,
  ...props
}, ref) => {
  const isHorizontal = orientation === "horizontal"
  
  return (
    <div
      ref={ref}
      className={cn(
        "inline-flex",
        isHorizontal ? "flex-row" : "flex-col",
        "[&>button]:rounded-none",
        isHorizontal ? "[&>button:first-child]:rounded-l-md [&>button:last-child]:rounded-r-md" : "[&>button:first-child]:rounded-t-md [&>button:last-child]:rounded-b-md",
        isHorizontal ? "[&>button:not(:last-child)]:border-r-0" : "[&>button:not(:last-child)]:border-b-0",
        className
      )}
      role="group"
      {...props}
    >
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child) && child.type === Button) {
          return React.cloneElement(child, {
            size: child.props.size || size,
            variant: child.props.variant || variant
          })
        }
        return child
      })}
    </div>
  )
})

ButtonGroup.displayName = "ButtonGroup"

// Icon button component
const IconButton = React.forwardRef(({
  className,
  variant = "ghost",
  size = "icon",
  children,
  "aria-label": ariaLabel,
  ...props
}, ref) => {
  return (
    <Button
      ref={ref}
      className={cn("shrink-0", className)}
      variant={variant}
      size={size}
      aria-label={ariaLabel}
      {...props}
    >
      {children}
    </Button>
  )
})

IconButton.displayName = "IconButton"

// Floating Action Button component
const FloatingActionButton = React.forwardRef(({
  className,
  variant = "default",
  size = "default",
  position = "bottom-right",
  children,
  ...props
}, ref) => {
  const positionClasses = {
    "bottom-right": "fixed bottom-6 right-6",
    "bottom-left": "fixed bottom-6 left-6",
    "top-right": "fixed top-6 right-6",
    "top-left": "fixed top-6 left-6"
  }

  return (
    <Button
      ref={ref}
      className={cn(
        "rounded-full shadow-lg hover:shadow-xl transition-shadow z-50",
        positionClasses[position],
        className
      )}
      variant={variant}
      size={size}
      {...props}
    >
      {children}
    </Button>
  )
})

FloatingActionButton.displayName = "FloatingActionButton"

export { Button, ButtonGroup, IconButton, FloatingActionButton, buttonVariants }

