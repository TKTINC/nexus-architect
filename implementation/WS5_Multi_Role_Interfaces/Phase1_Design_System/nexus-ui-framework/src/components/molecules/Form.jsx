import React from 'react'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'
import { AlertCircle, CheckCircle, Eye, EyeOff } from 'lucide-react'
import { Button } from '../atoms/Button'
import { Input, InputGroup } from '../atoms/Input'

// Form context for managing form state
const FormContext = React.createContext({})

const useForm = () => {
  const context = React.useContext(FormContext)
  if (!context) {
    throw new Error('useForm must be used within a Form component')
  }
  return context
}

// Form validation utilities
const validators = {
  required: (value) => {
    if (!value || (typeof value === 'string' && value.trim() === '')) {
      return 'This field is required'
    }
    return null
  },
  
  email: (value) => {
    if (value && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
      return 'Please enter a valid email address'
    }
    return null
  },
  
  minLength: (min) => (value) => {
    if (value && value.length < min) {
      return `Must be at least ${min} characters`
    }
    return null
  },
  
  maxLength: (max) => (value) => {
    if (value && value.length > max) {
      return `Must be no more than ${max} characters`
    }
    return null
  },
  
  pattern: (regex, message) => (value) => {
    if (value && !regex.test(value)) {
      return message || 'Invalid format'
    }
    return null
  },
  
  custom: (validatorFn) => validatorFn
}

// Main Form component
const Form = React.forwardRef(({
  className,
  onSubmit,
  children,
  initialValues = {},
  validationSchema = {},
  ...props
}, ref) => {
  const [values, setValues] = React.useState(initialValues)
  const [errors, setErrors] = React.useState({})
  const [touched, setTouched] = React.useState({})
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  // Validate a single field
  const validateField = React.useCallback((name, value) => {
    const fieldValidators = validationSchema[name] || []
    
    for (const validator of fieldValidators) {
      const error = validator(value)
      if (error) {
        return error
      }
    }
    return null
  }, [validationSchema])

  // Validate all fields
  const validateForm = React.useCallback(() => {
    const newErrors = {}
    
    Object.keys(validationSchema).forEach(name => {
      const error = validateField(name, values[name])
      if (error) {
        newErrors[name] = error
      }
    })
    
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }, [values, validateField, validationSchema])

  // Handle field change
  const handleFieldChange = React.useCallback((name, value) => {
    setValues(prev => ({ ...prev, [name]: value }))
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }))
    }
  }, [errors])

  // Handle field blur
  const handleFieldBlur = React.useCallback((name) => {
    setTouched(prev => ({ ...prev, [name]: true }))
    
    // Validate field on blur
    const error = validateField(name, values[name])
    setErrors(prev => ({ ...prev, [name]: error }))
  }, [validateField, values])

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault()
    
    // Mark all fields as touched
    const allTouched = Object.keys(validationSchema).reduce((acc, key) => {
      acc[key] = true
      return acc
    }, {})
    setTouched(allTouched)
    
    // Validate form
    if (!validateForm()) {
      return
    }
    
    setIsSubmitting(true)
    
    try {
      await onSubmit?.(values)
    } catch (error) {
      console.error('Form submission error:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const contextValue = React.useMemo(() => ({
    values,
    errors,
    touched,
    isSubmitting,
    handleFieldChange,
    handleFieldBlur,
    validateField,
    setFieldValue: (name, value) => setValues(prev => ({ ...prev, [name]: value })),
    setFieldError: (name, error) => setErrors(prev => ({ ...prev, [name]: error })),
    resetForm: () => {
      setValues(initialValues)
      setErrors({})
      setTouched({})
      setIsSubmitting(false)
    }
  }), [values, errors, touched, isSubmitting, handleFieldChange, handleFieldBlur, validateField, initialValues])

  return (
    <FormContext.Provider value={contextValue}>
      <form
        ref={ref}
        className={cn("space-y-6", className)}
        onSubmit={handleSubmit}
        noValidate
        {...props}
      >
        {children}
      </form>
    </FormContext.Provider>
  )
})
Form.displayName = "Form"

// Form Field component
const FormField = React.forwardRef(({
  name,
  label,
  description,
  required = false,
  children,
  className,
  ...props
}, ref) => {
  const { values, errors, touched, handleFieldChange, handleFieldBlur } = useForm()
  const fieldId = React.useId()
  const descriptionId = React.useId()
  const errorId = React.useId()
  
  const fieldError = touched[name] ? errors[name] : null
  const fieldValue = values[name] || ''

  return (
    <div ref={ref} className={cn("space-y-2", className)} {...props}>
      {label && (
        <label
          htmlFor={fieldId}
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
          id: fieldId,
          name,
          value: fieldValue,
          onChange: (e) => {
            const value = e.target.type === 'checkbox' ? e.target.checked : e.target.value
            handleFieldChange(name, value)
            children.props.onChange?.(e)
          },
          onBlur: (e) => {
            handleFieldBlur(name)
            children.props.onBlur?.(e)
          },
          'aria-describedby': cn(
            description && descriptionId,
            fieldError && errorId
          ),
          'aria-invalid': fieldError ? 'true' : 'false',
          error: !!fieldError
        })}
      </div>
      
      {fieldError && (
        <div
          id={errorId}
          className="flex items-center space-x-1 text-sm text-destructive"
          role="alert"
        >
          <AlertCircle className="h-4 w-4" />
          <span>{fieldError}</span>
        </div>
      )}
    </div>
  )
})
FormField.displayName = "FormField"

// Form Section component
const FormSection = React.forwardRef(({
  title,
  description,
  children,
  className,
  ...props
}, ref) => (
  <div ref={ref} className={cn("space-y-4", className)} {...props}>
    {(title || description) && (
      <div className="space-y-1">
        {title && (
          <h3 className="text-lg font-medium leading-none">{title}</h3>
        )}
        {description && (
          <p className="text-sm text-muted-foreground">{description}</p>
        )}
      </div>
    )}
    <div className="space-y-4">
      {children}
    </div>
  </div>
))
FormSection.displayName = "FormSection"

// Form Actions component
const FormActions = React.forwardRef(({
  children,
  className,
  align = "right",
  ...props
}, ref) => {
  const alignClasses = {
    left: "justify-start",
    center: "justify-center",
    right: "justify-end",
    between: "justify-between"
  }

  return (
    <div
      ref={ref}
      className={cn(
        "flex items-center space-x-2",
        alignClasses[align],
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
})
FormActions.displayName = "FormActions"

// Submit Button component
const SubmitButton = React.forwardRef(({
  children = "Submit",
  loadingText = "Submitting...",
  ...props
}, ref) => {
  const { isSubmitting } = useForm()

  return (
    <Button
      ref={ref}
      type="submit"
      loading={isSubmitting}
      loadingText={loadingText}
      {...props}
    >
      {children}
    </Button>
  )
})
SubmitButton.displayName = "SubmitButton"

// Reset Button component
const ResetButton = React.forwardRef(({
  children = "Reset",
  ...props
}, ref) => {
  const { resetForm } = useForm()

  return (
    <Button
      ref={ref}
      type="button"
      variant="outline"
      onClick={resetForm}
      {...props}
    >
      {children}
    </Button>
  )
})
ResetButton.displayName = "ResetButton"

// Checkbox Field component
const CheckboxField = React.forwardRef(({
  name,
  label,
  description,
  required = false,
  className,
  ...props
}, ref) => {
  const { values, errors, touched, handleFieldChange, handleFieldBlur } = useForm()
  const fieldId = React.useId()
  const fieldError = touched[name] ? errors[name] : null
  const fieldValue = values[name] || false

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-start space-x-2">
        <input
          ref={ref}
          id={fieldId}
          name={name}
          type="checkbox"
          checked={fieldValue}
          onChange={(e) => handleFieldChange(name, e.target.checked)}
          onBlur={() => handleFieldBlur(name)}
          className="mt-1 h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
          aria-invalid={fieldError ? 'true' : 'false'}
          {...props}
        />
        <div className="space-y-1">
          <label
            htmlFor={fieldId}
            className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
          >
            {label}
            {required && <span className="text-destructive ml-1">*</span>}
          </label>
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </div>
      </div>
      
      {fieldError && (
        <div className="flex items-center space-x-1 text-sm text-destructive ml-6">
          <AlertCircle className="h-4 w-4" />
          <span>{fieldError}</span>
        </div>
      )}
    </div>
  )
})
CheckboxField.displayName = "CheckboxField"

// Select Field component
const SelectField = React.forwardRef(({
  name,
  label,
  description,
  options = [],
  placeholder = "Select an option",
  required = false,
  className,
  ...props
}, ref) => {
  const { values, errors, touched, handleFieldChange, handleFieldBlur } = useForm()
  const fieldId = React.useId()
  const fieldError = touched[name] ? errors[name] : null
  const fieldValue = values[name] || ''

  return (
    <div className={cn("space-y-2", className)}>
      {label && (
        <label
          htmlFor={fieldId}
          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          {label}
          {required && <span className="text-destructive ml-1">*</span>}
        </label>
      )}
      
      {description && (
        <p className="text-sm text-muted-foreground">{description}</p>
      )}
      
      <select
        ref={ref}
        id={fieldId}
        name={name}
        value={fieldValue}
        onChange={(e) => handleFieldChange(name, e.target.value)}
        onBlur={() => handleFieldBlur(name)}
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          fieldError && "border-destructive focus-visible:ring-destructive"
        )}
        aria-invalid={fieldError ? 'true' : 'false'}
        {...props}
      >
        <option value="" disabled>
          {placeholder}
        </option>
        {options.map((option) => (
          <option
            key={typeof option === 'string' ? option : option.value}
            value={typeof option === 'string' ? option : option.value}
          >
            {typeof option === 'string' ? option : option.label}
          </option>
        ))}
      </select>
      
      {fieldError && (
        <div className="flex items-center space-x-1 text-sm text-destructive">
          <AlertCircle className="h-4 w-4" />
          <span>{fieldError}</span>
        </div>
      )}
    </div>
  )
})
SelectField.displayName = "SelectField"

export {
  Form,
  FormField,
  FormSection,
  FormActions,
  SubmitButton,
  ResetButton,
  CheckboxField,
  SelectField,
  useForm,
  validators
}

