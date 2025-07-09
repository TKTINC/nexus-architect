import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import './App.css'

// Layout Components
import DeveloperLayout from './components/layout/DeveloperLayout'

// Page Components
import DeveloperDashboard from './components/dashboard/DeveloperDashboard'
import CodeQuality from './components/quality/CodeQuality'
import WorkflowOptimization from './components/workflow/WorkflowOptimization'
import LearningCenter from './components/learning/LearningCenter'
import IDEIntegration from './components/ide/IDEIntegration'

function App() {
  const [theme, setTheme] = useState('light')
  const [user, setUser] = useState({
    name: 'Alex Chen',
    role: 'Senior Developer',
    avatar: '/api/placeholder/32/32',
    team: 'Platform Engineering',
    experience: 'Senior',
    skills: ['React', 'Node.js', 'Python', 'AWS', 'Docker']
  })

  // Theme management
  useEffect(() => {
    const savedTheme = localStorage.getItem('developer-theme') || 'light'
    setTheme(savedTheme)
    document.documentElement.classList.toggle('dark', savedTheme === 'dark')
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('developer-theme', newTheme)
    document.documentElement.classList.toggle('dark', newTheme === 'dark')
  }

  return (
    <Router>
      <div className="min-h-screen bg-background">
        <DeveloperLayout 
          user={user} 
          theme={theme} 
          onThemeToggle={toggleTheme}
        >
          <Routes>
            <Route path="/" element={<DeveloperDashboard />} />
            <Route path="/dashboard" element={<DeveloperDashboard />} />
            <Route path="/code-quality" element={<CodeQuality />} />
            <Route path="/workflow" element={<WorkflowOptimization />} />
            <Route path="/learning" element={<LearningCenter />} />
            <Route path="/ide-integration" element={<IDEIntegration />} />
          </Routes>
        </DeveloperLayout>
      </div>
    </Router>
  )
}

export default App

