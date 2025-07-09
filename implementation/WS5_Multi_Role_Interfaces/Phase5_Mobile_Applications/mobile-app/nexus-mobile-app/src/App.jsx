import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import './App.css'

// Layout Components
import MobileLayout from './components/layout/MobileLayout'
import BottomNavigation from './components/navigation/BottomNavigation'

// Screen Components
import HomeScreen from './components/screens/HomeScreen'
import ProjectsScreen from './components/screens/ProjectsScreen'
import TasksScreen from './components/screens/TasksScreen'
import TeamScreen from './components/screens/TeamScreen'
import ProfileScreen from './components/screens/ProfileScreen'
import NotificationsScreen from './components/screens/NotificationsScreen'

// Mobile Features
import OfflineIndicator from './components/mobile/OfflineIndicator'
import PushNotificationHandler from './components/mobile/PushNotificationHandler'

// Hooks
import { useOfflineSync } from './hooks/useOfflineSync'
import { useMobileFeatures } from './hooks/useMobileFeatures'

function App() {
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [currentUser, setCurrentUser] = useState(null)
  const [notifications, setNotifications] = useState([])

  // Mobile-specific hooks
  const { syncStatus, lastSync, pendingChanges } = useOfflineSync()
  const { 
    biometricSupported, 
    pushNotificationsEnabled, 
    deviceInfo,
    requestBiometric,
    requestNotificationPermission
  } = useMobileFeatures()

  // Network status monitoring
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  // Initialize mobile features
  useEffect(() => {
    const initializeMobileFeatures = async () => {
      // Request notification permissions
      if ('Notification' in window && Notification.permission === 'default') {
        await requestNotificationPermission()
      }

      // Initialize biometric authentication if supported
      if (biometricSupported) {
        console.log('Biometric authentication available')
      }

      // Set up service worker for offline functionality
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js')
          .then(registration => {
            console.log('Service Worker registered:', registration)
          })
          .catch(error => {
            console.log('Service Worker registration failed:', error)
          })
      }
    }

    initializeMobileFeatures()
  }, [biometricSupported, requestNotificationPermission])

  // Mock user authentication
  useEffect(() => {
    // Simulate user login
    setCurrentUser({
      id: 1,
      name: 'John Doe',
      email: 'john.doe@company.com',
      role: 'Project Manager',
      avatar: '/api/placeholder/40/40',
      preferences: {
        theme: 'light',
        notifications: true,
        biometric: biometricSupported
      }
    })
  }, [biometricSupported])

  return (
    <Router>
      <div className="min-h-screen bg-background text-foreground">
        {/* Offline Indicator */}
        <OfflineIndicator 
          isOnline={isOnline} 
          syncStatus={syncStatus}
          lastSync={lastSync}
          pendingChanges={pendingChanges}
        />

        {/* Push Notification Handler */}
        <PushNotificationHandler 
          notifications={notifications}
          onNotificationReceived={setNotifications}
        />

        {/* Main Application */}
        <MobileLayout currentUser={currentUser}>
          <Routes>
            <Route 
              path="/" 
              element={
                <HomeScreen 
                  currentUser={currentUser}
                  isOnline={isOnline}
                  deviceInfo={deviceInfo}
                />
              } 
            />
            <Route 
              path="/projects" 
              element={
                <ProjectsScreen 
                  isOnline={isOnline}
                  syncStatus={syncStatus}
                />
              } 
            />
            <Route 
              path="/tasks" 
              element={
                <TasksScreen 
                  isOnline={isOnline}
                  pendingChanges={pendingChanges}
                />
              } 
            />
            <Route 
              path="/team" 
              element={
                <TeamScreen 
                  currentUser={currentUser}
                  isOnline={isOnline}
                />
              } 
            />
            <Route 
              path="/notifications" 
              element={
                <NotificationsScreen 
                  notifications={notifications}
                  onClearNotifications={() => setNotifications([])}
                />
              } 
            />
            <Route 
              path="/profile" 
              element={
                <ProfileScreen 
                  currentUser={currentUser}
                  onUpdateUser={setCurrentUser}
                  biometricSupported={biometricSupported}
                  onRequestBiometric={requestBiometric}
                />
              } 
            />
          </Routes>
        </MobileLayout>

        {/* Bottom Navigation */}
        <BottomNavigation />
      </div>
    </Router>
  )
}

export default App

