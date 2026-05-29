import React from 'react';
import { Sparkles } from 'lucide-react';

export default function LoadingSpinner({ message = 'Loading...' }) {
  return (
    <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center gap-4">
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 rounded-full border-4 border-saree-500/20" />
        <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-saree-500 animate-spin" />
        <div className="absolute inset-0 flex items-center justify-center">
          <Sparkles className="w-6 h-6 text-saree-400 animate-pulse" />
        </div>
      </div>
      <p className="text-gray-400 text-sm">{message}</p>
    </div>
  );
}
