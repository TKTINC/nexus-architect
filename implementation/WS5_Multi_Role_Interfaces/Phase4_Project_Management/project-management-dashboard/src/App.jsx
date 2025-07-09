import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import './App.css'

// Layout Components
import ProjectLayout from './components/layout/ProjectLayout'

// Dashboard Components
import ProjectOverview from './components/dashboard/ProjectOverview'
import TaskManagement from './components/tasks/TaskManagement'
import TeamCollaboration from './components/collaboration/TeamCollaboration'
import ResourceManagement from './components/resources/ResourceManagement'

function App() {
  const [currentUser, setCurrentUser] = useState({
    id: 1,
    name: 'Sarah Chen',
    role: 'Project Manager',
    avatar: '/avatars/sarah-chen.jpg',
    permissions: ['project_management', 'team_collaboration', 'resource_allocation']
  })

  const [theme, setTheme] = useState('light')

  useEffect(() => {
    // Check for saved theme preference or default to 'light'
    const savedTheme = localStorage.getItem('theme') || 'light'
    setTheme(savedTheme)
    document.documentElement.classList.toggle('dark', savedTheme === 'dark')
  }, [])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light'
    setTheme(newTheme)
    localStorage.setItem('theme', newTheme)
    document.documentElement.classList.toggle('dark', newTheme === 'dark')
  }

  return (
    <Router>
      <div className="min-h-screen bg-background">
        <ProjectLayout 
          currentUser={currentUser}
          theme={theme}
          toggleTheme={toggleTheme}
        >
          <Routes>
            <Route path="/" element={<ProjectOverview />} />
            <Route path="/overview" element={<ProjectOverview />} />
            <Route path="/tasks" element={<TaskManagement />} />
            <Route path="/collaboration" element={<TeamCollaboration />} />
            <Route path="/resources" element={<ResourceManagement />} />
          </Routes>
        </ProjectLayout>
      </div>
    </Router>
  )
}

export default App

