import React from 'react'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'
import { MoreHorizontal, X } from 'lucide-react'
import { Button } from '../atoms/Button'

// Card variants
const cardVariants = cva(
  "rounded-lg border bg-card text-card-foreground shadow-sm transition-all duration-200",
  {
    variants: {
      variant: {
        default: "border-border",
        elevated: "border-border shadow-md hover:shadow-lg",
        outlined: "border-2 border-border shadow-none",
        filled: "border-transparent bg-muted",
        ghost: "border-transparent shadow-none bg-transparent"
      },
      size: {
        sm: "p-3",
        default: "p-6",
        lg: "p-8",
        xl: "p-10"
      },
      interactive: {
        true: "cursor-pointer hover:bg-accent/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        false: ""
      }
    },
    defaultVariants: {
      variant: "default",
      size: "default",
      interactive: false
    }
  }
)

const Card = React.forwardRef(({
  className,
  variant,
  size,
  interactive,
  onClick,
  children,
  ...props
}, ref) => {
  const isInteractive = interactive || !!onClick

  return (
    <div
      ref={ref}
      className={cn(cardVariants({ variant, size, interactive: isInteractive }), className)}
      onClick={onClick}
      role={isInteractive ? "button" : undefined}
      tabIndex={isInteractive ? 0 : undefined}
      onKeyDown={isInteractive ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onClick?.(e)
        }
      } : undefined}
      {...props}
    >
      {children}
    </div>
  )
})
Card.displayName = "Card"

const CardHeader = React.forwardRef(({
  className,
  children,
  ...props
}, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 p-6", className)}
    {...props}
  >
    {children}
  </div>
))
CardHeader.displayName = "CardHeader"

const CardTitle = React.forwardRef(({
  className,
  as: Component = "h3",
  children,
  ...props
}, ref) => (
  <Component
    ref={ref}
    className={cn("text-2xl font-semibold leading-none tracking-tight", className)}
    {...props}
  >
    {children}
  </Component>
))
CardTitle.displayName = "CardTitle"

const CardDescription = React.forwardRef(({
  className,
  children,
  ...props
}, ref) => (
  <p
    ref={ref}
    className={cn("text-sm text-muted-foreground", className)}
    {...props}
  >
    {children}
  </p>
))
CardDescription.displayName = "CardDescription"

const CardContent = React.forwardRef(({
  className,
  children,
  ...props
}, ref) => (
  <div
    ref={ref}
    className={cn("p-6 pt-0", className)}
    {...props}
  >
    {children}
  </div>
))
CardContent.displayName = "CardContent"

const CardFooter = React.forwardRef(({
  className,
  children,
  ...props
}, ref) => (
  <div
    ref={ref}
    className={cn("flex items-center p-6 pt-0", className)}
    {...props}
  >
    {children}
  </div>
))
CardFooter.displayName = "CardFooter"

// Enhanced Card with actions
const ActionCard = React.forwardRef(({
  className,
  title,
  description,
  actions,
  onClose,
  children,
  ...props
}, ref) => (
  <Card ref={ref} className={className} {...props}>
    <CardHeader className="flex flex-row items-start justify-between space-y-0 pb-2">
      <div className="space-y-1">
        {title && <CardTitle className="text-lg">{title}</CardTitle>}
        {description && <CardDescription>{description}</CardDescription>}
      </div>
      <div className="flex items-center space-x-1">
        {actions}
        {onClose && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={onClose}
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>
    </CardHeader>
    {children && <CardContent>{children}</CardContent>}
  </Card>
))
ActionCard.displayName = "ActionCard"

// Stats Card component
const StatsCard = React.forwardRef(({
  className,
  title,
  value,
  description,
  trend,
  trendDirection = "up",
  icon,
  ...props
}, ref) => {
  const trendColor = trendDirection === "up" ? "text-green-600" : 
                    trendDirection === "down" ? "text-red-600" : 
                    "text-muted-foreground"

  return (
    <Card ref={ref} className={className} {...props}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <p className="text-sm font-medium text-muted-foreground">{title}</p>
            <div className="flex items-baseline space-x-2">
              <p className="text-2xl font-bold">{value}</p>
              {trend && (
                <span className={cn("text-sm font-medium", trendColor)}>
                  {trendDirection === "up" && "↗"}
                  {trendDirection === "down" && "↘"}
                  {trendDirection === "neutral" && "→"}
                  {trend}
                </span>
              )}
            </div>
            {description && (
              <p className="text-xs text-muted-foreground">{description}</p>
            )}
          </div>
          {icon && (
            <div className="text-muted-foreground">
              {icon}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
})
StatsCard.displayName = "StatsCard"

// Feature Card component
const FeatureCard = React.forwardRef(({
  className,
  icon,
  title,
  description,
  href,
  onClick,
  ...props
}, ref) => {
  const isInteractive = !!href || !!onClick
  const Component = href ? "a" : "div"

  return (
    <Card
      ref={ref}
      className={cn(
        "group relative overflow-hidden",
        isInteractive && "cursor-pointer hover:shadow-md transition-shadow",
        className
      )}
      interactive={isInteractive}
      onClick={onClick}
      {...(href && { as: "a", href })}
      {...props}
    >
      <CardContent className="p-6">
        <div className="space-y-4">
          {icon && (
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 text-primary">
              {icon}
            </div>
          )}
          <div className="space-y-2">
            <h3 className="font-semibold leading-none tracking-tight">{title}</h3>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
        </div>
      </CardContent>
      {isInteractive && (
        <div className="absolute inset-0 bg-gradient-to-r from-transparent to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity" />
      )}
    </Card>
  )
})
FeatureCard.displayName = "FeatureCard"

// Product Card component
const ProductCard = React.forwardRef(({
  className,
  image,
  title,
  description,
  price,
  originalPrice,
  badge,
  actions,
  ...props
}, ref) => (
  <Card ref={ref} className={cn("overflow-hidden", className)} {...props}>
    {image && (
      <div className="relative aspect-video overflow-hidden">
        <img
          src={image}
          alt={title}
          className="h-full w-full object-cover transition-transform group-hover:scale-105"
        />
        {badge && (
          <div className="absolute top-2 left-2 rounded-full bg-primary px-2 py-1 text-xs font-medium text-primary-foreground">
            {badge}
          </div>
        )}
      </div>
    )}
    <CardContent className="p-4">
      <div className="space-y-2">
        <h3 className="font-semibold leading-none tracking-tight">{title}</h3>
        {description && (
          <p className="text-sm text-muted-foreground line-clamp-2">{description}</p>
        )}
        {(price || originalPrice) && (
          <div className="flex items-center space-x-2">
            {price && (
              <span className="text-lg font-bold">{price}</span>
            )}
            {originalPrice && (
              <span className="text-sm text-muted-foreground line-through">{originalPrice}</span>
            )}
          </div>
        )}
      </div>
    </CardContent>
    {actions && (
      <CardFooter className="p-4 pt-0">
        {actions}
      </CardFooter>
    )}
  </Card>
))
ProductCard.displayName = "ProductCard"

export {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  ActionCard,
  StatsCard,
  FeatureCard,
  ProductCard,
  cardVariants
}

