import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Download, Trash2, RotateCcw, CheckCircle2,
  ArrowLeft, Sparkles, Info, Cpu, Zap, AlertTriangle,
} from 'lucide-react';
import { toast } from 'react-toastify';
import StepIndicator from '../components/StepIndicator';
import { useDraping } from '../context/DrapingContext';
import { downloadOutput, cleanupOutput, getEngineStatus } from '../services/api';

export default function PreviewPage() {
  const navigate   = useNavigate();
  const draping    = useDraping();
  const [cleaning,    setCleaning]    = useState(false);
  const [downloaded,  setDownloaded]  = useState(false);
  const [engineInfo,  setEngineInfo]  = useState(null);

  // Prefer context values set during render; fall back to a fresh status call
  useEffect(() => {
    if (draping.engineUsed !== null && draping.engineUsed !== undefined) {
      setEngineInfo({
        engine:             draping.engineUsed,
        ml_model_available: draping.mlModelAvailable,
        fallback_reason:    draping.fallbackReason || '',
        weights_path:       '',   // not returned in render response, fetched below
      });
    }
    // Always fetch full status to get weights_path
    getEngineStatus()
      .then(res => {
        const s = res.data.data;
        setEngineInfo(prev => ({
          engine:             draping.engineUsed ?? s.engine,
          ml_model_available: draping.mlModelAvailable ?? s.ml_model_available,
          fallback_reason:    draping.fallbackReason   || s.fallback_reason || '',
          weights_path:       s.weights_path || '',
        }));
      })
      .catch(() => {});
  }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  const API_BASE = '';

  const handleDownload = async () => {
    if (!draping.outputId) return toast.error('No output to download');
    try {
      const res  = await downloadOutput(draping.outputId);
      const url  = URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href       = url;
      link.download   = `saree-draping-${draping.outputId}.jpg`;
      link.click();
      URL.revokeObjectURL(url);
      setDownloaded(true);
      toast.success('Image downloaded!');
    } catch {
      toast.error('Download failed');
    }
  };

  const handleCleanup = async () => {
    if (!draping.outputId) return;
    setCleaning(true);
    try {
      await cleanupOutput(draping.outputId);
      toast.success('Files deleted for your privacy.');
      draping.resetDraping();
      navigate('/upload');
    } catch {
      toast.error('Cleanup failed');
    } finally {
      setCleaning(false);
    }
  };

  const handleNewDraping = () => {
    draping.resetDraping();
    navigate('/upload');
  };

  // Build image URL from the output path served by Flask
  const imageUrl = draping.outputPath
    ? `${API_BASE}/outputs/${draping.outputPath.split(/[\\/]/).pop()}`
    : null;

  if (!draping.outputId) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <Sparkles className="w-12 h-12 text-saree-400 animate-pulse" />
        <p className="text-gray-400">No output found. Please complete the draping first.</p>
        <button onClick={() => navigate('/upload')} className="btn-primary">
          Start Draping
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-10 px-4">
      <div className="max-w-5xl mx-auto">
        <StepIndicator current={3} />

        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 bg-green-500/10 border border-green-500/20 text-green-400 px-4 py-2 rounded-full text-sm font-medium mb-4">
            <CheckCircle2 className="w-4 h-4" />
            Draping Complete!
          </div>
          <h1 className="font-display text-4xl font-bold text-white mb-3">Your Draped Saree</h1>
          <p className="text-gray-400">Review, download, or start a new draping session</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Output image */}
          <div className="lg:col-span-2 space-y-3">
            <div className="card overflow-hidden">
              {imageUrl ? (
                <img
                  src={imageUrl}
                  alt="Draped saree result"
                  className="w-full object-contain max-h-[760px]"
                  onError={e => { e.target.src = 'https://placehold.co/600x800/1a0a2e/a800e6?text=Preview+Available+After+Processing'; }}
                />
              ) : (
                <div className="flex flex-col items-center justify-center h-96 bg-gray-900/60">
                  <Sparkles className="w-12 h-12 text-saree-400/40 mb-3" />
                  <p className="text-gray-500">Preview rendering...</p>
                </div>
              )}
            </div>

            {/* Prominent download button below image */}
            {imageUrl && (
              <button
                onClick={handleDownload}
                className="w-full flex items-center justify-center gap-2.5 bg-gradient-to-r from-saree-600 to-purple-600 hover:from-saree-500 hover:to-purple-500 text-white font-semibold py-3.5 rounded-xl shadow-lg shadow-saree-500/20 transition-all active:scale-[0.98]"
              >
                <Download className="w-5 h-5" />
                {downloaded ? 'Download Again' : 'Download Result'}
              </button>
            )}
          </div>

          {/* Action panel */}
          <div className="space-y-4">

            {/* ── Engine Status panel ─────────────────────────────────── */}
            {engineInfo && (() => {
              const isML      = engineInfo.engine === 'ml_model';
              const hasWeights = engineInfo.ml_model_available;
              return (
                <div className={`card p-4 border ${
                  isML
                    ? 'border-green-500/25 bg-green-500/5'
                    : 'border-amber-500/25 bg-amber-500/5'
                }`}>
                  {/* Badge */}
                  <div className={`inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full mb-3 ${
                    isML
                      ? 'bg-green-500/15 text-green-300'
                      : 'bg-amber-500/15 text-amber-300'
                  }`}>
                    {isML
                      ? <Zap className="w-3 h-3" />
                      : <Cpu className="w-3 h-3" />}
                    {isML ? 'ML Model Active' : 'Geometric Fallback Active'}
                  </div>

                  {/* Summary message */}
                  <p className={`text-xs leading-relaxed mb-3 ${
                    isML ? 'text-green-400/80' : 'text-amber-400/80'
                  }`}>
                    {isML
                      ? 'Trained saree try-on model is being used for realistic results.'
                      : 'Train on Kaggle and place saree_tryon_model.pth in dataset/trained_models/ to enable ML mode.'}
                  </p>

                  {/* Detail rows */}
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between gap-2">
                      <span className="text-gray-500 flex-shrink-0">Engine used</span>
                      <span className={`font-mono font-medium truncate ${
                        isML ? 'text-green-300' : 'text-amber-300'
                      }`}>{engineInfo.engine}</span>
                    </div>
                    <div className="flex justify-between gap-2">
                      <span className="text-gray-500 flex-shrink-0">ML weights</span>
                      <span className={`font-medium ${hasWeights ? 'text-green-400' : 'text-red-400'}`}>
                        {hasWeights ? '✓ Found' : '✗ Not found'}
                      </span>
                    </div>
                    {engineInfo.fallback_reason && (
                      <div className="flex items-start gap-1.5 pt-1">
                        <AlertTriangle className="w-3 h-3 text-amber-400 mt-0.5 flex-shrink-0" />
                        <span className="text-amber-400/70 leading-relaxed break-all">
                          {engineInfo.fallback_reason}
                        </span>
                      </div>
                    )}
                    {engineInfo.weights_path && (
                      <div className="pt-1 border-t border-white/5">
                        <p className="text-gray-600 text-[10px] mb-0.5">Expected path</p>
                        <p className="text-gray-500 font-mono text-[10px] break-all leading-relaxed">
                          {engineInfo.weights_path}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}

            {/* Config recap */}
            <div className="card p-5">
              <h3 className="text-white font-semibold mb-4 text-sm flex items-center gap-2">
                <Info className="w-4 h-4 text-saree-400" /> Draping Details
              </h3>
              <div className="space-y-2 text-sm">
                {[
                  ['Style',   draping.drapingStyle],
                  ['Fabric',  draping.fabricType],
                  ['Neck',    draping.blouseConfig.neck_type],
                  ['Sleeve',  draping.blouseConfig.sleeve_type],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-gray-400">{k}</span>
                    <span className="text-white capitalize">{v}</span>
                  </div>
                ))}
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Colour</span>
                  <div className="flex items-center gap-1.5">
                    <div className="w-4 h-4 rounded" style={{ background: draping.blouseConfig.color }} />
                    <span className="text-white font-mono text-xs">{draping.blouseConfig.color}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Download (sidebar duplicate — smaller) */}
            <button
              onClick={handleDownload}
              className="btn-secondary w-full flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              {downloaded ? 'Downloaded ✓' : 'Save HD Image'}
            </button>

            {/* New draping */}
            <button
              onClick={handleNewDraping}
              className="btn-secondary w-full flex items-center justify-center gap-2"
            >
              <RotateCcw className="w-4 h-4" /> New Draping
            </button>

            {/* Go back */}
            <button
              onClick={() => navigate('/customize')}
              className="w-full flex items-center justify-center gap-2 text-gray-400 hover:text-white text-sm transition-colors"
            >
              <ArrowLeft className="w-4 h-4" /> Adjust Customization
            </button>

            {/* Privacy cleanup */}
            <div className="card p-4 border-red-500/20">
              <p className="text-xs text-gray-400 mb-3 leading-relaxed">
                For your privacy, delete all uploaded images and generated output from our servers.
              </p>
              <button
                onClick={handleCleanup}
                disabled={cleaning}
                className="w-full flex items-center justify-center gap-2 text-sm text-red-400 hover:text-red-300 border border-red-500/20 hover:border-red-400/40 rounded-xl py-2.5 transition-all disabled:opacity-50"
              >
                {cleaning ? (
                  <><div className="w-3.5 h-3.5 border-2 border-red-400/30 border-t-red-400 rounded-full animate-spin" />Deleting...</>
                ) : (
                  <><Trash2 className="w-4 h-4" /> Delete All Files</>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
