import { useState, useEffect, useCallback } from 'react'

// Offline Sync Hook for managing data synchronization
export const useOfflineSync = () => {
  const [syncStatus, setSyncStatus] = useState('idle') // idle, syncing, success, error
  const [lastSync, setLastSync] = useState(null)
  const [pendingChanges, setPendingChanges] = useState([])
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [syncProgress, setSyncProgress] = useState(0)
  const [conflictResolutions, setConflictResolutions] = useState([])

  // Initialize offline storage
  useEffect(() => {
    initializeOfflineStorage()
    loadPendingChanges()
    loadLastSyncTime()
  }, [])

  // Monitor network status
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true)
      // Auto-sync when coming back online
      if (pendingChanges.length > 0) {
        syncData()
      }
    }

    const handleOffline = () => {
      setIsOnline(false)
      setSyncStatus('offline')
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)

    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [pendingChanges])

  // Initialize IndexedDB for offline storage
  const initializeOfflineStorage = async () => {
    try {
      if (!window.indexedDB) {
        console.warn('IndexedDB not supported')
        return
      }

      const request = indexedDB.open('NexusArchitectDB', 1)
      
      request.onerror = () => {
        console.error('Failed to open IndexedDB')
      }

      request.onupgradeneeded = (event) => {
        const db = event.target.result

        // Create object stores for different data types
        if (!db.objectStoreNames.contains('projects')) {
          const projectStore = db.createObjectStore('projects', { keyPath: 'id' })
          projectStore.createIndex('lastModified', 'lastModified', { unique: false })
        }

        if (!db.objectStoreNames.contains('tasks')) {
          const taskStore = db.createObjectStore('tasks', { keyPath: 'id' })
          taskStore.createIndex('projectId', 'projectId', { unique: false })
          taskStore.createIndex('lastModified', 'lastModified', { unique: false })
        }

        if (!db.objectStoreNames.contains('pendingChanges')) {
          const changesStore = db.createObjectStore('pendingChanges', { keyPath: 'id', autoIncrement: true })
          changesStore.createIndex('timestamp', 'timestamp', { unique: false })
          changesStore.createIndex('type', 'type', { unique: false })
        }

        if (!db.objectStoreNames.contains('syncMetadata')) {
          db.createObjectStore('syncMetadata', { keyPath: 'key' })
        }
      }

      request.onsuccess = () => {
        console.log('IndexedDB initialized successfully')
      }
    } catch (error) {
      console.error('Error initializing offline storage:', error)
    }
  }

  // Load pending changes from storage
  const loadPendingChanges = async () => {
    try {
      const changes = await getFromIndexedDB('pendingChanges')
      setPendingChanges(changes || [])
    } catch (error) {
      console.error('Error loading pending changes:', error)
    }
  }

  // Load last sync time
  const loadLastSyncTime = async () => {
    try {
      const metadata = await getFromIndexedDB('syncMetadata', 'lastSync')
      if (metadata) {
        setLastSync(new Date(metadata.value))
      }
    } catch (error) {
      console.error('Error loading last sync time:', error)
    }
  }

  // Generic IndexedDB operations
  const getFromIndexedDB = (storeName, key = null) => {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('NexusArchitectDB', 1)
      
      request.onsuccess = () => {
        const db = request.result
        const transaction = db.transaction([storeName], 'readonly')
        const store = transaction.objectStore(storeName)
        
        if (key) {
          const getRequest = store.get(key)
          getRequest.onsuccess = () => resolve(getRequest.result)
          getRequest.onerror = () => reject(getRequest.error)
        } else {
          const getAllRequest = store.getAll()
          getAllRequest.onsuccess = () => resolve(getAllRequest.result)
          getAllRequest.onerror = () => reject(getAllRequest.error)
        }
      }
      
      request.onerror = () => reject(request.error)
    })
  }

  const saveToIndexedDB = (storeName, data) => {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('NexusArchitectDB', 1)
      
      request.onsuccess = () => {
        const db = request.result
        const transaction = db.transaction([storeName], 'readwrite')
        const store = transaction.objectStore(storeName)
        
        const saveRequest = store.put(data)
        saveRequest.onsuccess = () => resolve(saveRequest.result)
        saveRequest.onerror = () => reject(saveRequest.error)
      }
      
      request.onerror = () => reject(request.error)
    })
  }

  // Add change to pending queue
  const addPendingChange = useCallback(async (change) => {
    const newChange = {
      ...change,
      id: Date.now() + Math.random(),
      timestamp: new Date().toISOString(),
      status: 'pending'
    }

    try {
      await saveToIndexedDB('pendingChanges', newChange)
      setPendingChanges(prev => [...prev, newChange])
    } catch (error) {
      console.error('Error adding pending change:', error)
    }
  }, [])

  // Sync data with server
  const syncData = useCallback(async () => {
    if (!isOnline || syncStatus === 'syncing') {
      return
    }

    setSyncStatus('syncing')
    setSyncProgress(0)

    try {
      const totalChanges = pendingChanges.length
      let processedChanges = 0

      // Process each pending change
      for (const change of pendingChanges) {
        try {
          await processPendingChange(change)
          processedChanges++
          setSyncProgress((processedChanges / totalChanges) * 100)
        } catch (error) {
          console.error('Error processing change:', error)
          // Handle conflict or retry logic here
          await handleSyncConflict(change, error)
        }
      }

      // Update last sync time
      const now = new Date()
      await saveToIndexedDB('syncMetadata', { key: 'lastSync', value: now.toISOString() })
      setLastSync(now)

      // Clear processed changes
      setPendingChanges([])
      setSyncStatus('success')
      
      // Auto-hide success status after 3 seconds
      setTimeout(() => {
        if (syncStatus === 'success') {
          setSyncStatus('idle')
        }
      }, 3000)

    } catch (error) {
      console.error('Sync failed:', error)
      setSyncStatus('error')
    }
  }, [isOnline, syncStatus, pendingChanges])

  // Process individual pending change
  const processPendingChange = async (change) => {
    const { type, data, operation } = change

    // Simulate API calls (replace with actual API endpoints)
    const apiEndpoint = getApiEndpoint(type, operation)
    const response = await fetch(apiEndpoint, {
      method: getHttpMethod(operation),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + localStorage.getItem('authToken')
      },
      body: operation !== 'delete' ? JSON.stringify(data) : undefined
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`)
    }

    return response.json()
  }

  // Handle sync conflicts
  const handleSyncConflict = async (change, error) => {
    const conflict = {
      id: Date.now(),
      change,
      error: error.message,
      timestamp: new Date().toISOString(),
      resolved: false
    }

    setConflictResolutions(prev => [...prev, conflict])

    // Implement conflict resolution strategies
    if (error.status === 409) { // Conflict
      // Strategy 1: Server wins (default)
      console.log('Conflict detected, server version takes precedence')
      
      // Strategy 2: Last write wins
      // Strategy 3: Manual resolution required
      // Strategy 4: Merge changes
    }
  }

  // Utility functions
  const getApiEndpoint = (type, operation) => {
    const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000/api'
    const endpoints = {
      projects: `${baseUrl}/projects`,
      tasks: `${baseUrl}/tasks`,
      users: `${baseUrl}/users`
    }
    return endpoints[type] || endpoints.projects
  }

  const getHttpMethod = (operation) => {
    switch (operation) {
      case 'create': return 'POST'
      case 'update': return 'PUT'
      case 'delete': return 'DELETE'
      default: return 'GET'
    }
  }

  // Manual sync trigger
  const forcSync = useCallback(() => {
    if (isOnline) {
      syncData()
    }
  }, [isOnline, syncData])

  // Clear all offline data
  const clearOfflineData = useCallback(async () => {
    try {
      const request = indexedDB.deleteDatabase('NexusArchitectDB')
      request.onsuccess = () => {
        setPendingChanges([])
        setLastSync(null)
        setConflictResolutions([])
        console.log('Offline data cleared')
      }
    } catch (error) {
      console.error('Error clearing offline data:', error)
    }
  }, [])

  // Get offline data statistics
  const getOfflineStats = useCallback(async () => {
    try {
      const projects = await getFromIndexedDB('projects')
      const tasks = await getFromIndexedDB('tasks')
      
      return {
        projectsCount: projects?.length || 0,
        tasksCount: tasks?.length || 0,
        pendingChangesCount: pendingChanges.length,
        lastSync,
        storageUsed: await getStorageUsage()
      }
    } catch (error) {
      console.error('Error getting offline stats:', error)
      return null
    }
  }, [pendingChanges, lastSync])

  // Estimate storage usage
  const getStorageUsage = async () => {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate()
      return {
        used: estimate.usage,
        quota: estimate.quota,
        percentage: (estimate.usage / estimate.quota) * 100
      }
    }
    return null
  }

  return {
    // State
    syncStatus,
    lastSync,
    pendingChanges,
    isOnline,
    syncProgress,
    conflictResolutions,

    // Actions
    addPendingChange,
    syncData,
    forcSync,
    clearOfflineData,
    getOfflineStats,

    // Utilities
    saveToIndexedDB,
    getFromIndexedDB
  }
}

