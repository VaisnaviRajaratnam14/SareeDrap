import React from 'react';
import { Check } from 'lucide-react';

const STEPS = [
  { n: 1, label: 'Upload'    },
  { n: 2, label: 'Customize' },
  { n: 3, label: 'Preview'   },
];

export default function StepIndicator({ current }) {
  return (
    <div className="flex items-center justify-center gap-0 mb-10">
      {STEPS.map((step, idx) => (
        <React.Fragment key={step.n}>
          {/* Step circle */}
          <div className="flex flex-col items-center gap-1.5">
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 ${
                step.n < current
                  ? 'bg-green-500 text-white shadow-lg shadow-green-500/30'
                  : step.n === current
                  ? 'bg-gradient-to-br from-saree-600 to-saree-400 text-white shadow-lg shadow-saree-500/30'
                  : 'bg-gray-800 text-gray-500 border border-white/10'
              }`}
            >
              {step.n < current ? <Check className="w-4 h-4" /> : step.n}
            </div>
            <span
              className={`text-xs font-medium ${
                step.n === current ? 'text-saree-300' : 'text-gray-500'
              }`}
            >
              {step.label}
            </span>
          </div>

          {/* Connector */}
          {idx < STEPS.length - 1 && (
            <div
              className={`h-0.5 w-20 mx-2 mb-4 transition-colors duration-300 ${
                step.n < current ? 'bg-green-500/50' : 'bg-gray-700'
              }`}
            />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}
