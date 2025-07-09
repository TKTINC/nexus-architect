import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from 'next-themes'
import './App.css'

// Import components
import { DesignSystemShowcase } from './components/DesignSystemShowcase'
import { ComponentLibrary } from './components/ComponentLibrary'
import { AccessibilityDemo } from './components/AccessibilityDemo'
import { NavigationDemo } from './components/NavigationDemo'
import { ThemeDemo } from './components/ThemeDemo'

// Import layout components
import { Header } from './components/layout/Header'
import { Sidebar } from './components/layout/Sidebar'
import { Footer } from './components/layout/Footer'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
      <Router>
        <div className="min-h-screen bg-background text-foreground">
          <Header onMenuToggle={() => setSidebarOpen(!sidebarOpen)} />
          
          <div className="flex">
            <Sidebar isOpen={sidebarOpen} />
            
            <main className={`flex-1 transition-all duration-300 ${
              sidebarOpen ? 'ml-64' : 'ml-16'
            }`}>
              <div className="p-6">
                <Routes>
                  <Route path="/" element={<DesignSystemShowcase />} />
                  <Route path="/components" element={<ComponentLibrary />} />
                  <Route path="/accessibility" element={<AccessibilityDemo />} />
                  <Route path="/navigation" element={<NavigationDemo />} />
                  <Route path="/themes" element={<ThemeDemo />} />
                </Routes>
              </div>
            </main>
          </div>
          
          <Footer />
        </div>
      </Router>
    </ThemeProvider>
  )
}

export default App

