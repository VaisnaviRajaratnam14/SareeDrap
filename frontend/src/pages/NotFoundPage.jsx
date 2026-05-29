import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles } from 'lucide-react';

export default function NotFoundPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center gap-6 text-center px-4">
      <Sparkles className="w-16 h-16 text-saree-400/40" />
      <h1 className="font-display text-6xl font-bold text-white">404</h1>
      <p className="text-gray-400 text-lg">This page doesn't exist in our wardrobe.</p>
      <Link to="/" className="btn-primary">Back to Home</Link>
    </div>
  );
}
