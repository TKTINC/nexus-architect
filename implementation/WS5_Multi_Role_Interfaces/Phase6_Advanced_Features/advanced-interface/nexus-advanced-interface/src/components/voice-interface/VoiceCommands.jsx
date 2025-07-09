import React, { useState, useEffect, useRef } from 'react';

export function VoiceCommands({ onCommand, isEnabled = true }) {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [supportedCommands, setSupportedCommands] = useState([]);
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Initialize supported commands
    setSupportedCommands([
      { command: 'open dashboard', description: 'Navigate to main dashboard' },
      { command: 'dark mode', description: 'Switch to dark theme' },
      { command: 'light mode', description: 'Switch to light theme' },
      { command: 'show reports', description: 'Open reports section' },
      { command: 'help', description: 'Show help information' },
      { command: 'search [query]', description: 'Search for specific content' },
      { command: 'create new', description: 'Create new item' },
      { command: 'save changes', description: 'Save current changes' },
      { command: 'export data', description: 'Export current data' },
      { command: 'refresh page', description: 'Refresh current view' }
    ]);

    // Initialize speech recognition if available
    if ('webkitSpeechRecognition' in window && isEnabled) {
      const recognition = new window.webkitSpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.onresult = (event) => {
        const result = event.results[event.results.length - 1];
        const transcript = result[0].transcript;
        const confidence = result[0].confidence;

        setTranscript(transcript);
        setConfidence(confidence);

        if (result.isFinal && confidence > 0.7) {
          processCommand(transcript.toLowerCase().trim());
        }
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [isEnabled]);

  const processCommand = (command) => {
    // Process the voice command
    let action = null;

    if (command.includes('dashboard')) {
      action = { type: 'navigate', target: 'dashboard' };
    } else if (command.includes('dark mode')) {
      action = { type: 'theme', value: 'dark' };
    } else if (command.includes('light mode')) {
      action = { type: 'theme', value: 'light' };
    } else if (command.includes('reports')) {
      action = { type: 'navigate', target: 'reports' };
    } else if (command.includes('help')) {
      action = { type: 'show', target: 'help' };
    } else if (command.includes('search')) {
      const query = command.replace('search', '').trim();
      action = { type: 'search', query };
    } else if (command.includes('create new')) {
      action = { type: 'create', target: 'new' };
    } else if (command.includes('save')) {
      action = { type: 'save' };
    } else if (command.includes('export')) {
      action = { type: 'export' };
    } else if (command.includes('refresh')) {
      action = { type: 'refresh' };
    }

    if (action && onCommand) {
      onCommand(action);
    }

    // Clear transcript after processing
    setTimeout(() => {
      setTranscript('');
      setConfidence(0);
    }, 2000);
  };

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.start();
    }
  };

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop();
    }
  };

  if (!isEnabled || !('webkitSpeechRecognition' in window)) {
    return (
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <div className="text-center">
          <div className="text-muted-foreground mb-2">
            <svg className="h-8 w-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-2">Voice Commands Unavailable</h3>
          <p className="text-sm text-muted-foreground">
            Voice recognition is not supported in this browser or has been disabled.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Voice Control Panel */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Voice Interface</h3>
          <div className={`flex items-center space-x-2 rounded-lg px-3 py-1 ${
            isListening ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
          }`}>
            <div className={`h-2 w-2 rounded-full ${
              isListening ? 'bg-red-500 animate-pulse' : 'bg-gray-400'
            }`}></div>
            <span className="text-sm font-medium">
              {isListening ? 'Listening...' : 'Ready'}
            </span>
          </div>
        </div>

        <div className="flex items-center space-x-4 mb-4">
          <button
            onClick={isListening ? stopListening : startListening}
            className={`flex items-center space-x-2 rounded-lg px-4 py-2 font-medium transition-colors ${
              isListening
                ? 'bg-red-500 hover:bg-red-600 text-white'
                : 'bg-primary hover:bg-primary/90 text-primary-foreground'
            }`}
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            <span>{isListening ? 'Stop Listening' : 'Start Listening'}</span>
          </button>
        </div>

        {/* Live Transcript */}
        {transcript && (
          <div className="rounded-lg bg-muted p-4 mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Live Transcript:</span>
              <span className="text-xs text-muted-foreground">
                Confidence: {Math.round(confidence * 100)}%
              </span>
            </div>
            <p className="text-sm">{transcript}</p>
          </div>
        )}
      </div>

      {/* Supported Commands */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4">Supported Voice Commands</h3>
        <div className="grid gap-3 md:grid-cols-2">
          {supportedCommands.map((cmd, index) => (
            <div key={index} className="flex items-start space-x-3 p-3 rounded-lg bg-muted/50">
              <div className="flex-shrink-0 mt-1">
                <div className="h-2 w-2 rounded-full bg-primary"></div>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground">"{cmd.command}"</p>
                <p className="text-xs text-muted-foreground">{cmd.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Voice Settings */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-4">Voice Settings</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium">Continuous Listening</label>
              <p className="text-xs text-muted-foreground">Keep voice recognition active</p>
            </div>
            <input type="checkbox" className="rounded border-gray-300" />
          </div>
          
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium">Voice Feedback</label>
              <p className="text-xs text-muted-foreground">Speak command confirmations</p>
            </div>
            <input type="checkbox" className="rounded border-gray-300" />
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium">Language</label>
            <select className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
              <option value="en-US">English (US)</option>
              <option value="en-GB">English (UK)</option>
              <option value="es-ES">Spanish</option>
              <option value="fr-FR">French</option>
              <option value="de-DE">German</option>
              <option value="zh-CN">Chinese</option>
              <option value="ja-JP">Japanese</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}

