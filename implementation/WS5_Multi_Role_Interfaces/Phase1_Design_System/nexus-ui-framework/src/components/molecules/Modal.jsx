import React from 'react'
import { cva } from 'class-variance-authority'
import { cn } from '@/lib/utils'
import { X, AlertTriangle, CheckCircle, Info, AlertCircle } from 'lucide-react'
import { Button } from '../atoms/Button'
import { createPortal } from 'react-dom'

// Modal variants
const modalVariants = cva(
  "fixed inset-0 z-50 flex items-center justify-center p-4",
  {
    variants: {
      size: {
        sm: "max-w-sm",
        default: "max-w-md",
        lg: "max-w-lg",
        xl: "max-w-xl",
        "2xl": "max-w-2xl",
        "3xl": "max-w-3xl",
        "4xl": "max-w-4xl",
        "5xl": "max-w-5xl",
        "6xl": "max-w-6xl",
        "7xl": "max-w-7xl",
        full: "max-w-full"
      }
    },
    defaultVariants: {
      size: "default"
    }
  }
)

const modalContentVariants = cva(
  "relative w-full rounded-lg border bg-background p-6 shadow-lg animate-in fade-in-0 zoom-in-95 duration-200",
  {
    variants: {
      variant: {
        default: "border-border",
        destructive: "border-destructive",
        warning: "border-yellow-500",
        success: "border-green-500",
        info: "border-blue-500"
      }
    },
    defaultVariants: {
      variant: "default"
    }
  }
)

// Modal context for managing state
const ModalContext = React.createContext({})

const useModal = () => {
  const context = React.useContext(ModalContext)
  if (!context) {
    throw new Error('useModal must be used within a Modal component')
  }
  return context
}

// Main Modal component
const Modal = React.forwardRef(({
  children,
  open,
  onOpenChange,
  defaultOpen = false,
  modal = true,
  ...props
}, ref) => {
  const [isOpen, setIsOpen] = React.useState(defaultOpen)
  const isControlled = open !== undefined
  const modalOpen = isControlled ? open : isOpen

  const setModalOpen = React.useCallback((open) => {
    if (!isControlled) {
      setIsOpen(open)
    }
    onOpenChange?.(open)
  }, [isControlled, onOpenChange])

  // Handle escape key
  React.useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && modalOpen) {
        setModalOpen(false)
      }
    }

    if (modalOpen) {
      document.addEventListener('keydown', handleEscape)
      // Prevent body scroll
      document.body.style.overflow = 'hidden'
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = 'unset'
    }
  }, [modalOpen, setModalOpen])

  const contextValue = React.useMemo(() => ({
    open: modalOpen,
    onOpenChange: setModalOpen
  }), [modalOpen, setModalOpen])

  if (!modalOpen) return null

  return (
    <ModalContext.Provider value={contextValue}>
      {modal ? createPortal(children, document.body) : children}
    </ModalContext.Provider>
  )
})
Modal.displayName = "Modal"

// Modal trigger
const ModalTrigger = React.forwardRef(({
  children,
  asChild = false,
  ...props
}, ref) => {
  const { onOpenChange } = useModal()

  const handleClick = () => {
    onOpenChange(true)
  }

  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children, {
      ...props,
      ref,
      onClick: handleClick
    })
  }

  return (
    <button ref={ref} onClick={handleClick} {...props}>
      {children}
    </button>
  )
})
ModalTrigger.displayName = "ModalTrigger"

// Modal overlay
const ModalOverlay = React.forwardRef(({
  className,
  ...props
}, ref) => {
  const { onOpenChange } = useModal()

  return (
    <div
      ref={ref}
      className={cn(
        "fixed inset-0 z-50 bg-background/80 backdrop-blur-sm animate-in fade-in-0 duration-200",
        className
      )}
      onClick={() => onOpenChange(false)}
      {...props}
    />
  )
})
ModalOverlay.displayName = "ModalOverlay"

// Modal content
const ModalContent = React.forwardRef(({
  className,
  size,
  variant,
  children,
  ...props
}, ref) => {
  const { onOpenChange } = useModal()

  return (
    <>
      <ModalOverlay />
      <div className={cn(modalVariants({ size }))}>
        <div
          ref={ref}
          className={cn(modalContentVariants({ variant }), className)}
          onClick={(e) => e.stopPropagation()}
          role="dialog"
          aria-modal="true"
          {...props}
        >
          {children}
        </div>
      </div>
    </>
  )
})
ModalContent.displayName = "ModalContent"

// Modal header
const ModalHeader = React.forwardRef(({
  className,
  children,
  ...props
}, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col space-y-1.5 text-center sm:text-left", className)}
    {...props}
  >
    {children}
  </div>
))
ModalHeader.displayName = "ModalHeader"

// Modal title
const ModalTitle = React.forwardRef(({
  className,
  children,
  ...props
}, ref) => (
  <h2
    ref={ref}
    className={cn("text-lg font-semibold leading-none tracking-tight", className)}
    {...props}
  >
    {children}
  </h2>
))
ModalTitle.displayName = "ModalTitle"

// Modal description
const ModalDescription = React.forwardRef(({
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
ModalDescription.displayName = "ModalDescription"

// Modal body
const ModalBody = React.forwardRef(({
  className,
  children,
  ...props
}, ref) => (
  <div
    ref={ref}
    className={cn("py-4", className)}
    {...props}
  >
    {children}
  </div>
))
ModalBody.displayName = "ModalBody"

// Modal footer
const ModalFooter = React.forwardRef(({
  className,
  children,
  ...props
}, ref) => (
  <div
    ref={ref}
    className={cn("flex flex-col-reverse sm:flex-row sm:justify-end sm:space-x-2", className)}
    {...props}
  >
    {children}
  </div>
))
ModalFooter.displayName = "ModalFooter"

// Modal close button
const ModalClose = React.forwardRef(({
  className,
  children,
  asChild = false,
  ...props
}, ref) => {
  const { onOpenChange } = useModal()

  const handleClick = () => {
    onOpenChange(false)
  }

  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children, {
      ...props,
      ref,
      onClick: handleClick
    })
  }

  return (
    <button
      ref={ref}
      className={cn(
        "absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none",
        className
      )}
      onClick={handleClick}
      {...props}
    >
      {children || <X className="h-4 w-4" />}
      <span className="sr-only">Close</span>
    </button>
  )
})
ModalClose.displayName = "ModalClose"

// Confirmation Modal
const ConfirmationModal = React.forwardRef(({
  open,
  onOpenChange,
  title,
  description,
  confirmText = "Confirm",
  cancelText = "Cancel",
  variant = "default",
  onConfirm,
  onCancel,
  loading = false,
  ...props
}, ref) => {
  const handleConfirm = () => {
    onConfirm?.()
  }

  const handleCancel = () => {
    onCancel?.()
    onOpenChange(false)
  }

  const icons = {
    default: null,
    destructive: <AlertTriangle className="h-6 w-6 text-destructive" />,
    warning: <AlertCircle className="h-6 w-6 text-yellow-500" />,
    success: <CheckCircle className="h-6 w-6 text-green-500" />,
    info: <Info className="h-6 w-6 text-blue-500" />
  }

  return (
    <Modal open={open} onOpenChange={onOpenChange}>
      <ModalContent ref={ref} variant={variant} size="sm" {...props}>
        <ModalHeader>
          <div className="flex items-center space-x-3">
            {icons[variant]}
            <div>
              <ModalTitle>{title}</ModalTitle>
              {description && (
                <ModalDescription className="mt-2">{description}</ModalDescription>
              )}
            </div>
          </div>
        </ModalHeader>
        <ModalFooter className="mt-6">
          <Button
            variant="outline"
            onClick={handleCancel}
            disabled={loading}
          >
            {cancelText}
          </Button>
          <Button
            variant={variant === "destructive" ? "destructive" : "default"}
            onClick={handleConfirm}
            loading={loading}
          >
            {confirmText}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
})
ConfirmationModal.displayName = "ConfirmationModal"

// Alert Modal
const AlertModal = React.forwardRef(({
  open,
  onOpenChange,
  title,
  description,
  buttonText = "OK",
  variant = "info",
  onClose,
  ...props
}, ref) => {
  const handleClose = () => {
    onClose?.()
    onOpenChange(false)
  }

  const icons = {
    destructive: <AlertTriangle className="h-6 w-6 text-destructive" />,
    warning: <AlertCircle className="h-6 w-6 text-yellow-500" />,
    success: <CheckCircle className="h-6 w-6 text-green-500" />,
    info: <Info className="h-6 w-6 text-blue-500" />
  }

  return (
    <Modal open={open} onOpenChange={onOpenChange}>
      <ModalContent ref={ref} variant={variant} size="sm" {...props}>
        <ModalHeader>
          <div className="flex items-center space-x-3">
            {icons[variant]}
            <div>
              <ModalTitle>{title}</ModalTitle>
              {description && (
                <ModalDescription className="mt-2">{description}</ModalDescription>
              )}
            </div>
          </div>
        </ModalHeader>
        <ModalFooter className="mt-6">
          <Button onClick={handleClose}>
            {buttonText}
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  )
})
AlertModal.displayName = "AlertModal"

export {
  Modal,
  ModalTrigger,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalTitle,
  ModalDescription,
  ModalBody,
  ModalFooter,
  ModalClose,
  ConfirmationModal,
  AlertModal,
  useModal
}

