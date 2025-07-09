import { useState, useEffect, useCallback } from 'react'

// Mobile Features Hook for device-specific capabilities
export const useMobileFeatures = () => {
  const [biometricSupported, setBiometricSupported] = useState(false)
  const [pushNotificationsEnabled, setPushNotificationsEnabled] = useState(false)
  const [deviceInfo, setDeviceInfo] = useState({})
  const [cameraSupported, setCameraSupported] = useState(false)
  const [geolocationSupported, setGeolocationSupported] = useState(false)
  const [vibrationSupported, setVibrationSupported] = useState(false)

  // Initialize mobile features detection
  useEffect(() => {
    detectMobileFeatures()
  }, [])

  // Detect available mobile features
  const detectMobileFeatures = async () => {
    // Device Information
    const info = {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      cookieEnabled: navigator.cookieEnabled,
      onLine: navigator.onLine,
      screenWidth: window.screen.width,
      screenHeight: window.screen.height,
      pixelRatio: window.devicePixelRatio || 1,
      touchSupported: 'ontouchstart' in window,
      orientation: window.screen.orientation?.type || 'unknown'
    }

    // Detect mobile OS
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
    const isAndroid = /Android/.test(navigator.userAgent)

    info.isMobile = isMobile
    info.isIOS = isIOS
    info.isAndroid = isAndroid

    setDeviceInfo(info)

    // Check biometric authentication support
    if ('credentials' in navigator && 'create' in navigator.credentials) {
      try {
        // Check if WebAuthn is supported
        const isSupported = await PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable()
        setBiometricSupported(isSupported)
      } catch (error) {
        console.log('Biometric authentication not supported:', error)
        setBiometricSupported(false)
      }
    }

    // Check push notification support
    if ('Notification' in window && 'serviceWorker' in navigator) {
      setPushNotificationsEnabled(Notification.permission === 'granted')
    }

    // Check camera support
    if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices()
        const hasCamera = devices.some(device => device.kind === 'videoinput')
        setCameraSupported(hasCamera)
      } catch (error) {
        console.log('Camera detection failed:', error)
        setCameraSupported(false)
      }
    }

    // Check geolocation support
    if ('geolocation' in navigator) {
      setGeolocationSupported(true)
    }

    // Check vibration support
    if ('vibrate' in navigator) {
      setVibrationSupported(true)
    }
  }

  // Request biometric authentication
  const requestBiometric = useCallback(async () => {
    if (!biometricSupported) {
      throw new Error('Biometric authentication not supported')
    }

    try {
      // Create a new credential for biometric authentication
      const credential = await navigator.credentials.create({
        publicKey: {
          challenge: new Uint8Array(32),
          rp: {
            name: 'Nexus Architect',
            id: window.location.hostname
          },
          user: {
            id: new Uint8Array(16),
            name: 'user@nexusarchitect.com',
            displayName: 'Nexus User'
          },
          pubKeyCredParams: [
            {
              type: 'public-key',
              alg: -7 // ES256
            }
          ],
          authenticatorSelection: {
            authenticatorAttachment: 'platform',
            userVerification: 'required'
          },
          timeout: 60000,
          attestation: 'direct'
        }
      })

      return {
        success: true,
        credential: credential
      }
    } catch (error) {
      console.error('Biometric authentication failed:', error)
      return {
        success: false,
        error: error.message
      }
    }
  }, [biometricSupported])

  // Verify biometric authentication
  const verifyBiometric = useCallback(async (credentialId) => {
    if (!biometricSupported) {
      throw new Error('Biometric authentication not supported')
    }

    try {
      const assertion = await navigator.credentials.get({
        publicKey: {
          challenge: new Uint8Array(32),
          allowCredentials: [{
            id: credentialId,
            type: 'public-key'
          }],
          userVerification: 'required',
          timeout: 60000
        }
      })

      return {
        success: true,
        assertion: assertion
      }
    } catch (error) {
      console.error('Biometric verification failed:', error)
      return {
        success: false,
        error: error.message
      }
    }
  }, [biometricSupported])

  // Request notification permission
  const requestNotificationPermission = useCallback(async () => {
    if (!('Notification' in window)) {
      throw new Error('Notifications not supported')
    }

    try {
      const permission = await Notification.requestPermission()
      setPushNotificationsEnabled(permission === 'granted')
      return permission
    } catch (error) {
      console.error('Notification permission request failed:', error)
      return 'denied'
    }
  }, [])

  // Send local notification
  const sendNotification = useCallback((title, options = {}) => {
    if (!pushNotificationsEnabled) {
      console.warn('Notifications not enabled')
      return
    }

    const notification = new Notification(title, {
      icon: '/icon-192x192.png',
      badge: '/badge-72x72.png',
      ...options
    })

    // Auto-close after 5 seconds
    setTimeout(() => {
      notification.close()
    }, 5000)

    return notification
  }, [pushNotificationsEnabled])

  // Access camera
  const accessCamera = useCallback(async (constraints = {}) => {
    if (!cameraSupported) {
      throw new Error('Camera not supported')
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment', // Use back camera by default
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          ...constraints.video
        },
        audio: false,
        ...constraints
      })

      return stream
    } catch (error) {
      console.error('Camera access failed:', error)
      throw error
    }
  }, [cameraSupported])

  // Capture photo from camera
  const capturePhoto = useCallback(async (stream) => {
    return new Promise((resolve, reject) => {
      try {
        const video = document.createElement('video')
        const canvas = document.createElement('canvas')
        const context = canvas.getContext('2d')

        video.srcObject = stream
        video.play()

        video.onloadedmetadata = () => {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
          context.drawImage(video, 0, 0)

          canvas.toBlob((blob) => {
            resolve(blob)
          }, 'image/jpeg', 0.8)
        }
      } catch (error) {
        reject(error)
      }
    })
  }, [])

  // Get current location
  const getCurrentLocation = useCallback(async (options = {}) => {
    if (!geolocationSupported) {
      throw new Error('Geolocation not supported')
    }

    return new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            accuracy: position.coords.accuracy,
            timestamp: position.timestamp
          })
        },
        (error) => {
          reject(error)
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 60000,
          ...options
        }
      )
    })
  }, [geolocationSupported])

  // Vibrate device
  const vibrate = useCallback((pattern = 200) => {
    if (!vibrationSupported) {
      console.warn('Vibration not supported')
      return false
    }

    try {
      navigator.vibrate(pattern)
      return true
    } catch (error) {
      console.error('Vibration failed:', error)
      return false
    }
  }, [vibrationSupported])

  // Haptic feedback patterns
  const hapticFeedback = useCallback((type = 'light') => {
    const patterns = {
      light: 50,
      medium: 100,
      heavy: 200,
      success: [100, 50, 100],
      error: [200, 100, 200, 100, 200],
      warning: [100, 50, 100, 50, 100]
    }

    return vibrate(patterns[type] || patterns.light)
  }, [vibrate])

  // Check if device is in landscape mode
  const isLandscape = useCallback(() => {
    return window.innerWidth > window.innerHeight
  }, [])

  // Lock screen orientation (if supported)
  const lockOrientation = useCallback(async (orientation = 'portrait') => {
    if ('orientation' in screen && 'lock' in screen.orientation) {
      try {
        await screen.orientation.lock(orientation)
        return true
      } catch (error) {
        console.error('Orientation lock failed:', error)
        return false
      }
    }
    return false
  }, [])

  // Wake lock to prevent screen from sleeping
  const requestWakeLock = useCallback(async () => {
    if ('wakeLock' in navigator) {
      try {
        const wakeLock = await navigator.wakeLock.request('screen')
        return wakeLock
      } catch (error) {
        console.error('Wake lock failed:', error)
        return null
      }
    }
    return null
  }, [])

  // Share content using Web Share API
  const shareContent = useCallback(async (data) => {
    if ('share' in navigator) {
      try {
        await navigator.share(data)
        return true
      } catch (error) {
        console.error('Share failed:', error)
        return false
      }
    }
    return false
  }, [])

  // Install PWA prompt
  const [installPrompt, setInstallPrompt] = useState(null)

  useEffect(() => {
    const handleBeforeInstallPrompt = (e) => {
      e.preventDefault()
      setInstallPrompt(e)
    }

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt)

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
    }
  }, [])

  const installPWA = useCallback(async () => {
    if (!installPrompt) {
      return false
    }

    try {
      const result = await installPrompt.prompt()
      setInstallPrompt(null)
      return result.outcome === 'accepted'
    } catch (error) {
      console.error('PWA installation failed:', error)
      return false
    }
  }, [installPrompt])

  return {
    // Feature Support
    biometricSupported,
    pushNotificationsEnabled,
    cameraSupported,
    geolocationSupported,
    vibrationSupported,
    deviceInfo,

    // Authentication
    requestBiometric,
    verifyBiometric,

    // Notifications
    requestNotificationPermission,
    sendNotification,

    // Camera
    accessCamera,
    capturePhoto,

    // Location
    getCurrentLocation,

    // Haptics
    vibrate,
    hapticFeedback,

    // Device Features
    isLandscape,
    lockOrientation,
    requestWakeLock,
    shareContent,

    // PWA
    installPrompt: !!installPrompt,
    installPWA
  }
}

