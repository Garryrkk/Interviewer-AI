// routes/meetingRoutes.js
// Frontend routing configuration for meeting summarization feature

import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import MeetingSummarization from '../components/MeetingSummarization';
import { ProtectedRoute } from '../components/ProtectedRoute';
import { Summarization } from '../../services/aiService';

Summarization("This is the full transcript or meeting notes")
  .then(response => {
    console.log("API Response:", response);
  })
  .catch(err => {
    console.error("API Error:", err);
  });
  
// Main meeting routes component
export const MeetingRoutes = () => {
  return (
    <Routes>
      {/* Main meeting summarization dashboard */}
      <Route 
        path="/meetings" 
        element={
          <ProtectedRoute>
            <MeetingSummarization />
          </ProtectedRoute>
        } 
      />
      
      {/* Upload specific route */}
      <Route 
        path="/meetings/upload" 
        element={
          <ProtectedRoute>
            <MeetingSummarization defaultTab="upload" />
          </ProtectedRoute>
        } 
      />
      
      {/* Record specific route */}
      <Route 
        path="/meetings/record" 
        element={
          <ProtectedRoute>
            <MeetingSummarization defaultTab="record" />
          </ProtectedRoute>
        } 
      />
      
      {/* Analysis results route */}
      <Route 
        path="/meetings/results/:analysisId?" 
        element={
          <ProtectedRoute>
            <MeetingSummarization defaultTab="results" />
          </ProtectedRoute>
        } 
      />
      
      {/* User summaries route */}
      <Route 
        path="/meetings/summaries" 
        element={
          <ProtectedRoute>
            <MeetingSummarization defaultTab="summaries" />
          </ProtectedRoute>
        } 
      />
      
      {/* Individual summary view */}
      <Route 
        path="/meetings/summary/:summaryId" 
        element={
          <ProtectedRoute>
            <MeetingSummaryDetail />
          </ProtectedRoute>
        } 
      />
      
      {/* Real-time meeting route */}
      <Route 
        path="/meetings/live/:meetingId?" 
        element={
          <ProtectedRoute>
            <LiveMeetingAnalysis />
          </ProtectedRoute>
        } 
      />
      
      {/* Redirect root to main meetings page */}
      <Route path="/" element={<Navigate to="/meetings" replace />} />
      
      {/* 404 fallback */}
      <Route path="*" element={<Navigate to="/meetings" replace />} />
    </Routes>
  );
};

// App.js integration example
// import { BrowserRouter as Router } from 'react-router-dom';
// import { MeetingRoutes } from './routes/meetingRoutes';

export const AppWithRouting = () => {
  return (
    <Router>
      <div className="app">
        <MeetingRoutes />
      </div>
    </Router>
  );
};

// Navigation component for meeting features
export const MeetingNavigation = ({ currentPath }) => {
  const navItems = [
    { path: '/meetings', label: 'Dashboard', icon: 'home' },
    { path: '/meetings/upload', label: 'Upload Audio', icon: 'upload' },
    { path: '/meetings/record', label: 'Record Meeting', icon: 'mic' },
    { path: '/meetings/summaries', label: 'My Summaries', icon: 'file-text' },
    { path: '/meetings/live', label: 'Live Analysis', icon: 'radio' }
  ];

  return (
    <nav className="meeting-navigation">
      {navItems.map(item => (
        <a 
          key={item.path}
          href={item.path}
          className={`nav-item ${currentPath === item.path ? 'active' : ''}`}
        >
          <span className={`icon-${item.icon}`}></span>
          {item.label}
        </a>
      ))}
    </nav>
  );
};

// URL parameter hooks for React Router
export const useMeetingParams = () => {
  const { analysisId, summaryId, meetingId } = useParams();
  return { analysisId, summaryId, meetingId };
};

// Navigation utilities
export const meetingNavigation = {
  goToUpload: () => window.location.href = '/meetings/upload',
  goToRecord: () => window.location.href = '/meetings/record',
  goToResults: (analysisId) => window.location.href = `/meetings/results/${analysisId}`,
  goToSummaries: () => window.location.href = '/meetings/summaries',
  goToSummary: (summaryId) => window.location.href = `/meetings/summary/${summaryId}`,
  goToLive: (meetingId) => window.location.href = `/meetings/live/${meetingId}`
};

// Updated MeetingSummarization component with routing support
const MeetingSummarizationWithRouting = ({ defaultTab = 'upload' }) => {
  const { analysisId, summaryId } = useMeetingParams();
  const location = useLocation();
  const [activeTab, setActiveTab] = useState(defaultTab);

  // Set tab based on route
  useEffect(() => {
    const path = location.pathname;
    
    if (path.includes('/upload')) setActiveTab('upload');
    else if (path.includes('/record')) setActiveTab('record');
    else if (path.includes('/results')) setActiveTab('results');
    else if (path.includes('/summaries')) setActiveTab('summaries');
    else setActiveTab(defaultTab);
  }, [location.pathname, defaultTab]);

  // Load specific analysis or summary based on URL params
  useEffect(() => {
    if (analysisId) {
      loadAnalysis(analysisId);
    }
    if (summaryId) {
      loadSummary(summaryId);
    }
  }, [analysisId, summaryId]);

  const loadAnalysis = async (id) => {
    try {
      const analysis = await summarizationService.getAnalysisStatus(id);
      setAnalysisResult(analysis);
    } catch (error) {
      setError('Failed to load analysis: ' + error.message);
    }
  };

  const loadSummary = async (id) => {
    try {
      const summary = await summarizationService.getMeetingSummary(id);
      setSelectedSummary(summary);
    } catch (error) {
      setError('Failed to load summary: ' + error.message);
    }
  };

  // Rest of the MeetingSummarization component implementation...
  return <MeetingSummarization />;
};

// Individual summary detail component
const MeetingSummaryDetail = () => {
  const { summaryId } = useParams();
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadSummary = async () => {
      try {
        setLoading(true);
        const summaryData = await summarizationService.getMeetingSummary(summaryId);
        setSummary(summaryData);
      } catch (err) {
        setError('Failed to load summary: ' + err.message);
      } finally {
        setLoading(false);
      }
    };

    if (summaryId) {
      loadSummary();
    }
  }, [summaryId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading summary...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-400 mb-4">{error}</p>
          <button 
            onClick={() => window.history.back()}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 mb-4">Summary not found</p>
          <a 
            href="/meetings/summaries"
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
          >
            View All Summaries
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6" style={{ backgroundColor: '#1E1E2F' }}>
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <button 
              onClick={() => window.history.back()}
              className="text-gray-400 hover:text-gray-200"
            >
              ← Back
            </button>
            <h1 className="text-2xl font-bold text-gray-100">Meeting Summary</h1>
          </div>
          
          <div className="flex items-center space-x-4 text-sm text-gray-400">
            <span>Created: {new Date(summary.created_at).toLocaleDateString()}</span>
            <span>Type: {summary.summary_type}</span>
            <span>Words: {summary.word_count}</span>
          </div>
        </div>

        {/* Summary Content */}
        <div className="space-y-6">
          <div className="bg-gray-800 p-6 rounded-lg" style={{ backgroundColor: '#2A2A3E' }}>
            <h2 className="text-lg font-semibold mb-4 text-gray-100">Summary</h2>
            <p className="text-gray-300 leading-relaxed">{summary.summary_text}</p>
          </div>

          {summary.key_points?.length > 0 && (
            <div className="bg-gray-800 p-6 rounded-lg" style={{ backgroundColor: '#2A2A3E' }}>
              <h2 className="text-lg font-semibold mb-4 text-gray-100">Key Points</h2>
              <ul className="space-y-2">
                {summary.key_points.map((point, index) => (
                  <li key={index} className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-purple-400 rounded-full mt-2 flex-shrink-0"></div>
                    <span className="text-gray-300">{point}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {summary.action_items?.length > 0 && (
            <div className="bg-gray-800 p-6 rounded-lg" style={{ backgroundColor: '#2A2A3E' }}>
              <h2 className="text-lg font-semibold mb-4 text-gray-100">Action Items</h2>
              <div className="space-y-3">
                {summary.action_items.map((item, index) => (
                  <div key={index} className="p-3 bg-gray-700 rounded-lg">
                    <p className="font-medium text-gray-100">{item.task}</p>
                    {item.assignee && (
                      <p className="text-sm text-gray-400 mt-1">Assigned to: {item.assignee}</p>
                    )}
                    {item.deadline && (
                      <p className="text-sm text-gray-400">Due: {item.deadline}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="mt-8 flex space-x-4">
          <button
            onClick={() => summarizationService.exportSummary(summary.summary_id, 'pdf')}
            className="px-6 py-2 rounded-lg font-medium transition-all"
            style={{ backgroundColor: '#8F74D4', color: '#F8FAFC' }}
          >
            Export PDF
          </button>
          
          <button
            onClick={() => navigator.share({ 
              title: 'Meeting Summary', 
              text: summary.summary_text,
              url: window.location.href
            })}
            className="px-6 py-2 bg-gray-700 text-gray-200 rounded-lg font-medium hover:bg-gray-600 transition-all"
          >
            Share
          </button>
        </div>
      </div>
    </div>
  );
};

// Live meeting analysis component
const LiveMeetingAnalysis = () => {
  const { meetingId } = useParams();
  const [isLive, setIsLive] = useState(false);
  const [realTimeInsights, setRealTimeInsights] = useState([]);
  const [currentSentiment, setCurrentSentiment] = useState('neutral');
  const wsRef = useRef(null);

  useEffect(() => {
    if (meetingId && isLive) {
      // Connect to WebSocket for real-time updates
      wsRef.current = summarizationService.connectWebSocket(
        meetingId,
        handleRealTimeUpdate,
        handleWebSocketError
      );
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [meetingId, isLive]);

  const handleRealTimeUpdate = (data) => {
    if (data.type === 'insight') {
      setRealTimeInsights(prev => [data.insight, ...prev.slice(0, 9)]); // Keep last 10
    } else if (data.type === 'sentiment') {
      setCurrentSentiment(data.sentiment);
    }
  };

  const handleWebSocketError = (error) => {
    console.error('WebSocket error:', error);
    setIsLive(false);
  };

  const toggleLiveAnalysis = () => {
    setIsLive(!isLive);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-6" style={{ backgroundColor: '#1E1E2F' }}>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-4">Live Meeting Analysis</h1>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={toggleLiveAnalysis}
              className={`px-6 py-2 rounded-lg font-medium transition-all ${
                isLive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {isLive ? 'Stop Analysis' : 'Start Analysis'}
            </button>
            
            <div className={`flex items-center space-x-2 ${isLive ? 'text-green-400' : 'text-gray-400'}`}>
              <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`}></div>
              <span>{isLive ? 'Live' : 'Offline'}</span>
            </div>
          </div>
        </div>

        {/* Real-time insights */}
        <div className="space-y-6">
          <div className="bg-gray-800 p-6 rounded-lg" style={{ backgroundColor: '#2A2A3E' }}>
            <h2 className="text-lg font-semibold mb-4">Current Sentiment</h2>
            <div className={`inline-flex px-4 py-2 rounded-full text-sm font-medium ${
              currentSentiment === 'positive' ? 'bg-green-900 text-green-200' :
              currentSentiment === 'negative' ? 'bg-red-900 text-red-200' :
              'bg-yellow-900 text-yellow-200'
            }`}>
              {currentSentiment.charAt(0).toUpperCase() + currentSentiment.slice(1)}
            </div>
          </div>

          <div className="bg-gray-800 p-6 rounded-lg" style={{ backgroundColor: '#2A2A3E' }}>
            <h2 className="text-lg font-semibold mb-4">Real-time Insights</h2>
            {realTimeInsights.length === 0 ? (
              <p className="text-gray-400">No insights yet. Start live analysis to see real-time updates.</p>
            ) : (
              <div className="space-y-3">
                {realTimeInsights.map((insight, index) => (
                  <div key={index} className="p-3 bg-gray-700 rounded-lg">
                    <p className="text-gray-300">{insight}</p>
                    <span className="text-xs text-gray-500">
                      {new Date().toLocaleTimeString()}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Export all components and utilities
export {
  MeetingSummarizationWithRouting,
  MeetingSummaryDetail,
  LiveMeetingAnalysis,
  MeetingNavigation,
  useMeetingParams,
  meetingNavigation
};

// ProtectedRoute component (should be implemented based on your auth system)
const ProtectedRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
        if (!token) {
          setIsAuthenticated(false);
          setLoading(false);
          return;
        }

        // Verify token with backend
        const response = await fetch(`${process.env.REACT_APP_API_URL}/auth/verify`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });

        setIsAuthenticated(response.ok);
      } catch (error) {
        console.error('Auth check failed:', error);
        setIsAuthenticated(false);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-400 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return children;
};

// Main App component integration example
export const MeetingApp = () => {
  const [apiHealth, setApiHealth] = useState(true);

  useEffect(() => {
    const checkApiHealth = async () => {
      const health = await summarizationService.checkHealth();
      setApiHealth(health);
    };

    checkApiHealth();
    // Check API health every 5 minutes
    const interval = setInterval(checkApiHealth, 5 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, []);

  if (!apiHealth) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-400 mb-4">Service Unavailable</h1>
          <p className="text-gray-400 mb-6">
            The meeting summarization service is currently unavailable. Please try again later.
          </p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <div className="app">
        {/* Global error boundary */}
        <ErrorBoundary>
          <MeetingRoutes />
        </ErrorBoundary>
      </div>
    </Router>
  );
};

// Error boundary component
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Application error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-red-400 mb-4">Something went wrong</h1>
            <p className="text-gray-400 mb-6">
              An unexpected error occurred. Please refresh the page.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}