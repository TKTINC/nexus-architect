import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from 'next-themes'
import './App.css'

// Components
import DashboardLayout from './components/layout/DashboardLayout'
import ExecutiveDashboard from './components/dashboard/ExecutiveDashboard'
import BusinessImpact from './components/analytics/BusinessImpact'
import RealTimeMonitoring from './components/monitoring/RealTimeMonitoring'
import Reports from './components/reports/Reports'

function App() {
  const [user] = useState({
    name: 'Sarah Chen',
    role: 'Chief Technology Officer',
    avatar: '/api/placeholder/40/40',
    company: 'TechCorp Industries'
  })

  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <Router>
        <div className="min-h-screen bg-background">
          <DashboardLayout user={user}>
            <Routes>
              <Route path="/" element={<ExecutiveDashboard />} />
              <Route path="/dashboard" element={<ExecutiveDashboard />} />
              <Route path="/business-impact" element={<BusinessImpact />} />
              <Route path="/monitoring" element={<RealTimeMonitoring />} />
              <Route path="/reports" element={<Reports />} />
            </Routes>
          </DashboardLayout>
        </div>
      </Router>
    </ThemeProvider>
  )
}

export default App

