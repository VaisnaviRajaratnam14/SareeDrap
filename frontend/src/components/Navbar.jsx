import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Sparkles, Menu, X } from 'lucide-react';

export default function Navbar() {
  const location        = useLocation();
  const [open, setOpen] = useState(false);

  const isActive = path => location.pathname === path;

  const navLink = (to, label) => (
    <Link
      to={to}
      onClick={() => setOpen(false)}
      className={`text-sm font-medium transition-colors ${
        isActive(to) ? 'text-saree-300' : 'text-gray-300 hover:text-white'
      }`}
    >
      {label}
    </Link>
  );

  return (
    <nav className="sticky top-0 z-50 bg-gray-950/90 backdrop-blur-md border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-saree-600 to-saree-400 flex items-center justify-center shadow-lg shadow-saree-500/30">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="font-display font-bold text-white text-lg">
              Saree<span className="text-saree-400">Drape</span>
            </span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-6">
            {navLink('/', 'Home')}
            {navLink('/upload', 'Try Now')}
          </div>

          {/* Desktop CTA */}
          <div className="hidden md:flex items-center gap-3">
            <Link to="/upload" className="btn-primary !py-2 !px-5 text-sm flex items-center gap-2">
              <Sparkles className="w-4 h-4" /> Start Draping
            </Link>
          </div>

          {/* Mobile toggle */}
          <button
            className="md:hidden text-gray-300 hover:text-white"
            onClick={() => setOpen(!open)}
          >
            {open ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden border-t border-white/5 bg-gray-950/95 backdrop-blur-md px-4 py-4 space-y-3">
          {navLink('/', 'Home')}
          {navLink('/upload', 'Try Now')}
          <div className="pt-2 border-t border-white/10">
            <Link
              to="/upload"
              onClick={() => setOpen(false)}
              className="btn-primary w-full text-center block"
            >
              Start Draping
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}
