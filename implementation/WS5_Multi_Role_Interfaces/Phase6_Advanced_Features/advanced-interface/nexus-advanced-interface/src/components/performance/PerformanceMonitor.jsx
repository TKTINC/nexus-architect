import React, { useState, useEffect } from 'react';

export function PerformanceMonitor() {
  const [metrics, setMetrics] = useState({
    loadTime: 0,
    renderTime: 0,
    memoryUsage: 0,
    cacheHitRate: 0,
    networkLatency: 0,
    fps: 0
  });
  const [history, setHistory] = useState([]);
  const [isMonitoring, setIsMonitoring] = useState(true);

  useEffect(() => {
    let interval;
    
    if (isMonitoring) {
      // Start performance monitoring
      interval = setInterval(() => {
        collectMetrics();
      }, 1000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isMonitoring]);

  const collectMetrics = () => {
    // Collect various performance metrics
    const newMetrics = {
      loadTime: getLoadTime(),
      renderTime: getRenderTime(),
      memoryUsage: getMemoryUsage(),
      cacheHitRate: getCacheHitRate(),
      networkLatency: getNetworkLatency(),
      fps: getFPS(),
      timestamp: Date.now()
    };

    setMetrics(newMetrics);
    
    // Keep last 60 data points (1 minute of history)
    setHistory(prev => {
      const updated = [...prev, newMetrics];
      return updated.slice(-60);
    });
  };

  const getLoadTime = () => {
    if (performance && performance.timing) {
      const timing = performance.timing;
      return timing.loadEventEnd - timing.navigationStart;
    }
    return Math.random() * 2000 + 500; // Simulated
  };

  const getRenderTime = () => {
    if (performance && performance.now) {
      return performance.now();
    }
    return Math.random() * 100 + 10; // Simulated
  };

  const getMemoryUsage = () => {
    if (performance && performance.memory) {
      return (performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize) * 100;
    }
    return Math.random() * 60 + 20; // Simulated
  };

  const getCacheHitRate = () => {
    // Simulated cache hit rate
    return Math.random() * 20 + 75; // 75-95%
  };

  const getNetworkLatency = () => {
    // Simulated network latency
    return Math.random() * 100 + 50; // 50-150ms
  };

  const getFPS = () => {
    // Simulated FPS
    return Math.random() * 10 + 55; // 55-65 FPS
  };

  const getMetricStatus = (value, thresholds) => {
    if (value <= thresholds.good) return 'good';
    if (value <= thresholds.warning) return 'warning';
    return 'critical';
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const MetricCard = ({ title, value, unit, status, description }) => (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-muted-foreground">{title}</h4>
        <div className={`h-2 w-2 rounded-full ${
          status === 'good' ? 'bg-green-500' :
          status === 'warning' ? 'bg-yellow-500' :
          'bg-red-500'
        }`}></div>
      </div>
      <div className="mt-2">
        <div className="text-2xl font-bold">
          {typeof value === 'number' ? value.toFixed(1) : value}
          <span className="text-sm font-normal text-muted-foreground ml-1">{unit}</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Performance Monitor</h1>
          <p className="text-muted-foreground">
            Real-time application performance metrics and optimization insights
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsMonitoring(!isMonitoring)}
            className={`flex items-center space-x-2 rounded-lg px-3 py-1 text-sm font-medium ${
              isMonitoring
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
            }`}
          >
            <div className={`h-2 w-2 rounded-full ${
              isMonitoring ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
            }`}></div>
            <span>{isMonitoring ? 'Monitoring' : 'Paused'}</span>
          </button>
        </div>
      </div>

      {/* Core Web Vitals */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <MetricCard
          title="Load Time"
          value={metrics.loadTime}
          unit="ms"
          status={getMetricStatus(metrics.loadTime, { good: 1500, warning: 3000 })}
          description="Time to fully load the page"
        />
        <MetricCard
          title="Render Time"
          value={metrics.renderTime}
          unit="ms"
          status={getMetricStatus(metrics.renderTime, { good: 50, warning: 100 })}
          description="Time to render components"
        />
        <MetricCard
          title="Memory Usage"
          value={metrics.memoryUsage}
          unit="%"
          status={getMetricStatus(metrics.memoryUsage, { good: 50, warning: 80 })}
          description="JavaScript heap memory usage"
        />
        <MetricCard
          title="Cache Hit Rate"
          value={metrics.cacheHitRate}
          unit="%"
          status={getMetricStatus(100 - metrics.cacheHitRate, { good: 10, warning: 25 })}
          description="Percentage of cached requests"
        />
        <MetricCard
          title="Network Latency"
          value={metrics.networkLatency}
          unit="ms"
          status={getMetricStatus(metrics.networkLatency, { good: 100, warning: 200 })}
          description="Average network response time"
        />
        <MetricCard
          title="Frame Rate"
          value={metrics.fps}
          unit="FPS"
          status={getMetricStatus(60 - metrics.fps, { good: 5, warning: 15 })}
          description="Rendering frame rate"
        />
      </div>

      {/* Performance Chart */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4">Performance Trends</h3>
        <div className="h-64 flex items-end space-x-1">
          {history.slice(-30).map((point, index) => (
            <div key={index} className="flex-1 flex flex-col items-center">
              <div
                className="w-full bg-blue-500 rounded-t"
                style={{
                  height: `${(point.loadTime / 3000) * 100}%`,
                  minHeight: '2px'
                }}
              ></div>
              <div className="text-xs text-muted-foreground mt-1">
                {new Date(point.timestamp).toLocaleTimeString().slice(-8, -3)}
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center justify-center space-x-4 text-sm text-muted-foreground">
          <div className="flex items-center space-x-1">
            <div className="h-3 w-3 bg-blue-500 rounded"></div>
            <span>Load Time (ms)</span>
          </div>
        </div>
      </div>

      {/* Optimization Recommendations */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4">Optimization Recommendations</h3>
        <div className="space-y-4">
          {metrics.loadTime > 2000 && (
            <div className="flex items-start space-x-3 p-4 rounded-lg bg-yellow-50 border border-yellow-200">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-medium text-yellow-800">Slow Load Time Detected</h4>
                <p className="text-sm text-yellow-700 mt-1">
                  Consider implementing code splitting and lazy loading to improve initial load performance.
                </p>
              </div>
            </div>
          )}

          {metrics.memoryUsage > 70 && (
            <div className="flex items-start space-x-3 p-4 rounded-lg bg-red-50 border border-red-200">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-medium text-red-800">High Memory Usage</h4>
                <p className="text-sm text-red-700 mt-1">
                  Memory usage is high. Consider optimizing component re-renders and cleaning up event listeners.
                </p>
              </div>
            </div>
          )}

          {metrics.cacheHitRate < 80 && (
            <div className="flex items-start space-x-3 p-4 rounded-lg bg-blue-50 border border-blue-200">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-medium text-blue-800">Low Cache Hit Rate</h4>
                <p className="text-sm text-blue-700 mt-1">
                  Improve caching strategies to reduce network requests and improve performance.
                </p>
              </div>
            </div>
          )}

          {metrics.loadTime <= 1500 && metrics.memoryUsage <= 50 && metrics.cacheHitRate >= 80 && (
            <div className="flex items-start space-x-3 p-4 rounded-lg bg-green-50 border border-green-200">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h4 className="text-sm font-medium text-green-800">Excellent Performance</h4>
                <p className="text-sm text-green-700 mt-1">
                  All performance metrics are within optimal ranges. Great job!
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Performance Settings */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4">Performance Settings</h3>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium">Enable Performance Monitoring</label>
                <p className="text-xs text-muted-foreground">Collect real-time performance data</p>
              </div>
              <input 
                type="checkbox" 
                checked={isMonitoring}
                onChange={(e) => setIsMonitoring(e.target.checked)}
                className="rounded border-gray-300" 
              />
            </div>
            
            <div className="flex items-center justify-between">
              <div>
                <label className="text-sm font-medium">Auto-Optimization</label>
                <p className="text-xs text-muted-foreground">Automatically apply performance optimizations</p>
              </div>
              <input type="checkbox" className="rounded border-gray-300" />
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Monitoring Interval</label>
              <select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="1000">1 second</option>
                <option value="5000">5 seconds</option>
                <option value="10000">10 seconds</option>
                <option value="30000">30 seconds</option>
              </select>
            </div>
            
            <div className="space-y-2">
              <label className="text-sm font-medium">Data Retention</label>
              <select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                <option value="60">1 minute</option>
                <option value="300">5 minutes</option>
                <option value="900">15 minutes</option>
                <option value="3600">1 hour</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

