import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import QuickRespond from "./features/QuickRespond/QuickRespond";
import Summarization from "./features/Summarization/Summarization";
import VoiceRecognition from "./features/VoiceRecognition/VoiceRecognition";
import MainFeature from "./features/MainFeature/MainFeature";
import ImageRecognition from "./features/ImageRecognition/ImageRecognition";
import KeyInsights from "./features/KeyInsights/KeyInsights";
import HandsFree from "./features/HandsFree/HandsFree";
import Overlay from "./features/MainFeature/Overlay";
import HiddenTest from "./HiddenTest";

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        {/* Main Content */}
        <main className="flex-grow">
          <Routes>
            <Route path="/quick-respond" element={<QuickRespond />} />
            <Route path="/summarization" element={<Summarization />} />
            <Route path="/voice-recognition" element={<VoiceRecognition />} />
            <Route path="/main-feature" element={<MainFeature />} />
            <Route path="/image-recognition" element={<ImageRecognition />} />
            <Route path="/key-insights" element={<KeyInsights />} />
            <Route path="/hands-free" element={<HandsFree />} />
            <Route path="/overlay" element={<Overlay />} />
            <Route path="/hidden-test" element={<HiddenTest />} />

            <Route
              path="/"
              element={
                <div className="text-center mt-20">
                  <h1 className="text-3xl font-semibold text-gray-800">
                    Welcome to the Interviewer AI Dashboard
                  </h1>
                  <p className="mt-4 text-gray-600">
                    Select a feature from the sidebar to get started.
                  </p>
                </div>
              }
            />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="text-center py-4 text-gray-500 text-sm border-t">
          © {new Date().getFullYear()}
        </footer>
      </div>
    </Router>
  );
}

export default App;