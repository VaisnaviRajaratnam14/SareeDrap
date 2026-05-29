import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import { DrapingProvider } from './context/DrapingContext.jsx';

import Navbar from './components/Navbar.jsx';
import HomePage from './pages/HomePage.jsx';
import UploadPage from './pages/UploadPage.jsx';
import CustomizePage from './pages/CustomizePage.jsx';
import PreviewPage from './pages/PreviewPage.jsx';
import NotFoundPage from './pages/NotFoundPage.jsx';

export default function App() {
  return (
    <DrapingProvider>
      <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <div className="min-h-screen bg-gray-950">
          <Navbar />
          <Routes>
            <Route path="/"         element={<HomePage />} />
            <Route path="/upload"   element={<UploadPage />} />
            <Route path="/customize" element={<CustomizePage />} />
            <Route path="/preview"  element={<PreviewPage />} />
            <Route path="*"         element={<NotFoundPage />} />
          </Routes>
          <ToastContainer
            position="top-right"
            autoClose={4000}
            theme="dark"
            toastStyle={{ background: '#1a0a2e', borderLeft: '4px solid #a800e6' }}
          />
        </div>
      </Router>
    </DrapingProvider>
  );
}
