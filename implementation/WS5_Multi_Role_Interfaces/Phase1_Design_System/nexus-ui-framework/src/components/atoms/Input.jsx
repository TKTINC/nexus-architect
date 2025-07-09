import React from 'react'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'
import { Eye, EyeOff, AlertCircle, CheckCircle, Search, X } from 'lucide-react'

// Input variants
const inputVariants = cva(
  "flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-colors",
  {
    variants: {
      variant: {
        default: "border-input",
        error: "border-destructive focus-visible:ring-destructive",
        success: "border-green-500 focus-visible:ring-green-500",
        warning: "border-yellow-500 focus-visible:ring-yellow-500"
      },
      size: {
        sm: "h-8 px-2 text-xs",
        default: "h-10 px-3",
        lg: "h-12 px-4 text-base"
      }
    },
    defaultVariants: {
      variant: "default",
      size: "default"
    }
  }
)

const Input = React.forwardRef(({
  className,
  type = "text",
  variant,
  size,
  error,
  success,
  disabled,
  leftIcon,
  rightIcon,
  clearable = false,
  onClear,
  ...props
}, ref) => {
  const [showPassword, setShowPassword] = React.useState(false)
  const [value, setValue] = React.useState(props.value || props.defaultValue || '')
  
  // Determine variant based on state
  const currentVariant = error ? 'error' : success ? 'success' : variant

  const handleClear = () => {
    setValue('')
    if (onClear) {
      onClear()
    }
    if (props.onChange) {
      props.onChange({ target: { value: '' } })
    }
  }

  const handleChange = (e) => {
    setValue(e.target.value)
    if (props.onChange) {
      props.onChange(e)
    }
  }

  const isPassword = type === 'password'
  const inputType = isPassword && showPassword ? 'text' : type
  const hasValue = value && value.length > 0

  return (
    <div className="relative">
      {leftIcon && (
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
          {leftIcon}
        </div>
      )}
      
      <input
        type={inputType}
        className={cn(
          inputVariants({ variant: currentVariant, size }),
          leftIcon && "pl-10",
          (rightIcon || isPassword || (clearable && hasValue)) && "pr-10",
          className
        )}
        ref={ref}
        disabled={disabled}
        value={value}
        onChange={handleChange}
        {...props}
      />
      
      <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center space-x-1">
        {clearable && hasValue && !disabled && (
          <button
            type="button"
            onClick={handleClear}
            className="text-muted-foreground hover:text-foreground transition-colors"
            aria-label="Clear input"
          >
            <X className="h-4 w-4" />
          </button>
        )}
        
        {isPassword && (
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="text-muted-foreground hover:text-foreground transition-colors"
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        )}
        
        {rightIcon && !isPassword && !(clearable && hasValue) && (
          <div className="text-muted-foreground">
            {rightIcon}
          </div>
        )}
      </div>
    </div>
  )
})

Input.displayName = "Input"

// Search Input component
const SearchInput = React.forwardRef(({
  className,
  placeholder = "Search...",
  onSearch,
  ...props
}, ref) => {
  return (
    <Input
      ref={ref}
      type="search"
      placeholder={placeholder}
      leftIcon={<Search className="h-4 w-4" />}
      clearable
      className={className}
      {...props}
    />
  )
})

SearchInput.displayName = "SearchInput"

// Textarea component
const Textarea = React.forwardRef(({
  className,
  variant,
  error,
  success,
  resize = true,
  ...props
}, ref) => {
  const currentVariant = error ? 'error' : success ? 'success' : variant

  return (
    <textarea
      className={cn(
        "flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-colors",
        !resize && "resize-none",
        currentVariant === 'error' && "border-destructive focus-visible:ring-destructive",
        currentVariant === 'success' && "border-green-500 focus-visible:ring-green-500",
        currentVariant === 'warning' && "border-yellow-500 focus-visible:ring-yellow-500",
        className
      )}
      ref={ref}
      {...props}
    />
  )
})

Textarea.displayName = "Textarea"

// Input Group component for labels and help text
const InputGroup = React.forwardRef(({
  className,
  label,
  description,
  error,
  success,
  required = false,
  children,
  ...props
}, ref) => {
  const inputId = React.useId()
  const descriptionId = React.useId()
  const errorId = React.useId()

  return (
    <div ref={ref} className={cn("space-y-2", className)} {...props}>
      {label && (
        <label
          htmlFor={inputId}
          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          {label}
          {required && <span className="text-destructive ml-1">*</span>}
        </label>
      )}
      
      {description && (
        <p
          id={descriptionId}
          className="text-sm text-muted-foreground"
        >
          {description}
        </p>
      )}
      
      <div className="relative">
        {React.cloneElement(children, {
          id: inputId,
          'aria-describedby': cn(
            description && descriptionId,
            error && errorId
          ),
          'aria-invalid': error ? 'true' : 'false',
          error: !!error,
          success: !!success
        })}
      </div>
      
      {error && (
        <div
          id={errorId}
          className="flex items-center space-x-1 text-sm text-destructive"
          role="alert"
        >
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}
      
      {success && !error && (
        <div className="flex items-center space-x-1 text-sm text-green-600">
          <CheckCircle className="h-4 w-4" />
          <span>{success}</span>
        </div>
      )}
    </div>
  )
})

InputGroup.displayName = "InputGroup"

// Number Input component
const NumberInput = React.forwardRef(({
  className,
  min,
  max,
  step = 1,
  value,
  onChange,
  ...props
}, ref) => {
  const [inputValue, setInputValue] = React.useState(value || 0)

  const handleIncrement = () => {
    const newValue = Number(inputValue) + Number(step)
    if (max === undefined || newValue <= max) {
      setInputValue(newValue)
      if (onChange) {
        onChange({ target: { value: newValue } })
      }
    }
  }

  const handleDecrement = () => {
    const newValue = Number(inputValue) - Number(step)
    if (min === undefined || newValue >= min) {
      setInputValue(newValue)
      if (onChange) {
        onChange({ target: { value: newValue } })
      }
    }
  }

  const handleChange = (e) => {
    const newValue = e.target.value
    setInputValue(newValue)
    if (onChange) {
      onChange(e)
    }
  }

  return (
    <div className="relative">
      <Input
        ref={ref}
        type="number"
        min={min}
        max={max}
        step={step}
        value={inputValue}
        onChange={handleChange}
        className={cn("pr-16", className)}
        {...props}
      />
      <div className="absolute right-1 top-1/2 -translate-y-1/2 flex flex-col">
        <button
          type="button"
          onClick={handleIncrement}
          className="px-2 py-0.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Increment"
        >
          ▲
        </button>
        <button
          type="button"
          onClick={handleDecrement}
          className="px-2 py-0.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Decrement"
        >
          ▼
        </button>
      </div>
    </div>
  )
})

NumberInput.displayName = "NumberInput"

export { Input, SearchInput, Textarea, InputGroup, NumberInput, inputVariants }

