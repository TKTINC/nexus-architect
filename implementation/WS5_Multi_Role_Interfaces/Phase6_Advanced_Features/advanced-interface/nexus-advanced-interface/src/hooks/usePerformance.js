import { useState, useEffect, useRef } from 'react'

export function usePerformance() {
  const [metrics, setMetrics] = useState({
    loadTime: 0,
    renderTime: 0,
    memoryUsage: 0,
    fps: 0
  })
  const [isMonitoring, setIsMonitoring] = useState(false)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (isMonitoring) {
      intervalRef.current = setInterval(() => {
        collectMetrics()
      }, 1000)
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isMonitoring])

  const collectMetrics = () => {
    const newMetrics = {
      loadTime: getLoadTime(),
      renderTime: getRenderTime(),
      memoryUsage: getMemoryUsage(),
      fps: getFPS()
    }
    setMetrics(newMetrics)
  }

  const getLoadTime = () => {
    if (performance && performance.timing) {
      const timing = performance.timing
      return timing.loadEventEnd - timing.navigationStart
    }
    return 0
  }

  const getRenderTime = () => {
    if (performance && performance.now) {
      return performance.now()
    }
    return 0
  }

  const getMemoryUsage = () => {
    if (performance && performance.memory) {
      return (performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize) * 100
    }
    return 0
  }

  const getFPS = () => {
    // Simplified FPS calculation
    return 60 // Placeholder
  }

  const startMonitoring = () => setIsMonitoring(true)
  const stopMonitoring = () => setIsMonitoring(false)

  return {
    metrics,
    isMonitoring,
    startMonitoring,
    stopMonitoring
  }
}

