import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft, ArrowRight, Loader2, Scissors, Shirt, Cpu, Zap,
  Database, Sparkles, CheckCircle2, XCircle, RefreshCw,
  ChevronDown, Download, Wand2, AlertTriangle, ExternalLink,
  BookOpen, TrendingUp, Cloud, MonitorSmartphone, FlaskConical,
} from 'lucide-react';
import { toast } from 'react-toastify';
import StepIndicator from '../components/StepIndicator';
import { useDraping } from '../context/DrapingContext';
import {
  renderDraping, getEngineStatus, testInference,
  fetchDataset, cleanDataset, annotateDataset, startTraining,
  getDatasetStatus, getTrainingStatus, getLiveCounts,
} from '../services/api';

const DRAPING_STYLES = [
  { id: 'nivi',     label: 'Nivi Style',     desc: 'Classic South/North Indian',  color: 'from-purple-600 to-pink-500'   },
  { id: 'bridal',   label: 'Bridal Style',   desc: 'Grand ceremonial drape',      color: 'from-rose-600   to-orange-500' },
  { id: 'hanging',  label: 'Hanging Style',  desc: 'Loose flowing East Indian',   color: 'from-blue-600   to-cyan-500'   },
  { id: 'gujarati', label: 'Gujarati Style', desc: 'Front pallu West Indian',     color: 'from-amber-500  to-yellow-400' },
];

const NECK_TYPES   = [
  { id: 'round',      label: 'Round Neck'  },
  { id: 'boat',       label: 'Boat Neck'   },
  { id: 'v_neck',     label: 'V-Neck'      },
  { id: 'square',     label: 'Square Neck' },
  { id: 'sweetheart', label: 'Sweetheart'  },
];

const SLEEVE_TYPES = [
  { id: 'sleeveless', label: 'Sleeveless' },
  { id: 'short',      label: 'Short'      },
  { id: 'half',       label: 'Half'       },
  { id: 'full',       label: 'Full'       },
];

const CATEGORIES = [
  { id: 'body_images',     label: 'Body Images'     },
  { id: 'saree_images',    label: 'Saree Images'    },
  { id: 'blouse_materials',label: 'Blouse Materials'},
];

const FABRIC_TYPES = [
  { id: 'silk',    label: 'Silk'    },
  { id: 'cotton',  label: 'Cotton'  },
  { id: 'chiffon', label: 'Chiffon' },
  { id: 'georgette',label:'Georgette'},
];

const KAGGLE_DATASETS = [
  { label: 'Saree Images Collection',  url: 'https://www.kaggle.com/datasets/search?q=saree+images',    category: 'saree_images'    },
  { label: 'Human Body Pose Dataset',  url: 'https://www.kaggle.com/datasets/search?q=human+body+pose', category: 'body_images'     },
  { label: 'Fabric Texture Dataset',   url: 'https://www.kaggle.com/datasets/search?q=fabric+texture',  category: 'blouse_materials'},
];

const READINESS_TIERS = [
  { min: 500, label: 'Production-quality dataset', color: 'text-green-300',  bg: 'bg-green-500' },
  { min: 150, label: 'Good training dataset',      color: 'text-green-400',  bg: 'bg-green-500' },
  { min: 50,  label: 'Testing-ready dataset',      color: 'text-amber-400',  bg: 'bg-amber-500' },
  { min: 0,   label: 'Insufficient dataset',       color: 'text-red-400',    bg: 'bg-red-500'   },
];

function getTier(cnt) {
  return READINESS_TIERS.find(t => cnt >= t.min);
}

function fmtEta(sec) {
  if (sec == null || sec < 0) return '—';
  if (sec < 60)  return `${sec}s`;
  const m = Math.floor(sec / 60), s = sec % 60;
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

// ── Small helpers ─────────────────────────────────────────────────────────────
function StatusBadge({ ok, label }) {
  return ok ? (
    <span className="inline-flex items-center gap-1 text-xs text-green-400 font-medium">
      <CheckCircle2 className="w-3.5 h-3.5" /> {label}
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-xs text-gray-500">
      <XCircle className="w-3.5 h-3.5" /> {label}
    </span>
  );
}

function StepRow({ number, title, done, children }) {
  return (
    <div className="flex gap-4">
      <div className="flex flex-col items-center">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold flex-shrink-0 ${
          done ? 'bg-green-500/20 text-green-400 border border-green-500/30'
               : 'bg-saree-500/10 text-saree-400 border border-saree-500/20'
        }`}>
          {done ? <CheckCircle2 className="w-4 h-4" /> : number}
        </div>
        {number < 3 && <div className="w-px flex-1 bg-white/5 mt-1" />}
      </div>
      <div className="pb-6 flex-1 min-w-0">
        <p className={`text-sm font-semibold mb-3 ${done ? 'text-green-300' : 'text-white'}`}>{title}</p>
        {children}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function CustomizePage() {
  const navigate = useNavigate();
  const draping  = useDraping();

  // Engine
  const [engineStatus,   setEngineStatus]   = useState(null);

  // Dataset step
  const [datasetSlug,    setDatasetSlug]    = useState('');
  const [category,       setCategory]       = useState('body_images');
  const [maxImages,      setMaxImages]      = useState(150);
  const [fetchMode,      setFetchMode]      = useState('append');
  const [fetchLoading,   setFetchLoading]   = useState(false);
  const [fetchResult,    setFetchResult]    = useState(null);
  const [perCategory,    setPerCategory]    = useState({});
  const [rawCounts,      setRawCounts]      = useState({});
  const [datasetFetched, setDatasetFetched] = useState(false);

  // Clean step
  const [cleanLoading,   setCleanLoading]   = useState(false);
  const [cleanResult,    setCleanResult]    = useState(null);
  const [cleanDone,      setCleanDone]      = useState(false);
  const [cleanedCounts,  setCleanedCounts]  = useState({});

  // Annotate step
  const [annotateLoading,  setAnnotateLoading]  = useState(false);
  const [annotateResult,   setAnnotateResult]   = useState(null);
  const [annotateDone,     setAnnotateDone]     = useState(false);

  // Train step
  const [trainLoading,   setTrainLoading]   = useState(false);
  const [trainResult,    setTrainResult]    = useState(null);
  const [trainDone,      setTrainDone]      = useState(false);
  const [trainProgress,  setTrainProgress]  = useState(null); // live poll data
  const [epochsInput,    setEpochsInput]    = useState(10);
  const [trainingMode,   setTrainingMode]   = useState('prototype'); // 'prototype' | 'kaggle_full'
  const trainPollRef = useRef(null);

  // Test inference
  const [testLoading,    setTestLoading]    = useState(false);
  const [testResult,     setTestResult]     = useState(null);

  // Generate
  const [genLoading,     setGenLoading]     = useState(false);

  // ── Refresh live counts from /api/dataset/counts (never cached) ─────────────
  const refreshDatasetCounts = useCallback(() => {
    getLiveCounts()
      .then(res => {
        const d = res.data?.data || res.data;
        if (d?.raw)     { setRawCounts(d.raw);     setPerCategory(d.raw); }
        if (d?.cleaned) { setCleanedCounts(d.cleaned); }
        if (d?.total_raw > 0) setDatasetFetched(true);
      })
      .catch(() => {});
    // Also sync state flags from the cached status (fetched/cleaned/annotated booleans only)
    getDatasetStatus()
      .then(res => {
        const d = res.data?.data || res.data;
        if (d?.cleaned)  setCleanDone(true);
        if (d?.annotated) setAnnotateDone(true);
      })
      .catch(() => {});
  }, []);

  // Poll engine + dataset/train status on mount
  const refreshEngine = useCallback(() => {
    console.log('[Engine] Fetching engine status from /api/render/engine-status …');
    getEngineStatus()
      .then(res => {
        const d = res.data.data;
        setEngineStatus(d);
        const realML = d.ml_model_available && d.model_loaded && !d.is_stub;
        if (realML) {
          console.log(`[Engine] ✅ ML Model Active — ${d.param_count_m}M params on ${d.device}`);
          setTrainDone(true);
        } else {
          console.log(`[Engine] ⚠ Geometric fallback — ${d.fallback_reason || d.log}`);
          setTrainDone(false);
        }
      })
      .catch(err => {
        console.error('[Engine] engine-status request failed:', err.message);
        setEngineStatus(null);
      });
  }, []);

  const stopTrainPoll = useCallback(() => {
    if (trainPollRef.current) {
      clearInterval(trainPollRef.current);
      trainPollRef.current = null;
    }
  }, []);

  const startTrainPoll = useCallback(() => {
    stopTrainPoll();
    trainPollRef.current = setInterval(async () => {
      try {
        const res = await getTrainingStatus();
        const d   = res.data?.data || res.data;
          setTrainProgress(d);

        if (d?.status === 'completed') {
          stopTrainPoll();
          setTrainLoading(false);
          const isReal = d?.model_exists && !d?.is_stub;
          setTrainResult({
            message: d?.message,
            status:  d?.status,
            isReal,
            loss:    d?.loss,
            epochs:  d?.total_epochs,
          });
          if (isReal) {
            setTrainDone(true);
            toast.success('Model trained! Running inference test…');
            refreshEngine();
            setTimeout(async () => {
              setTestLoading(true);
              try {
                const tres = await testInference();
                const td   = tres.data?.data || tres.data;
                setTestResult(td);
                if (td?.passed) {
                  toast.success(`Inference verified — ${td.latency_ms}ms. ML Model Active!`);
                } else {
                  toast.warn('Training done but inference test failed — using geometric fallback');
                }
                refreshEngine();
              } catch { /* ignore */ } finally {
                setTestLoading(false);
              }
            }, 800);
          } else {
            toast.info(d?.message || 'Training complete (stub — PyTorch not installed)');
            refreshEngine();
          }
        } else if (d?.status === 'failed') {
          stopTrainPoll();
          setTrainLoading(false);
          setTrainResult({ error: d?.message || 'Training failed' });
          toast.error(d?.message || 'Training failed');
        }
      } catch { /* network hiccup — keep polling */ }
    }, 1500);
  }, [stopTrainPoll, refreshEngine]);

  useEffect(() => {
    refreshEngine();
    // Load dataset counts
    // Load live counts from filesystem (never cached)
    getLiveCounts()
      .then(res => {
        const d = res.data?.data || res.data;
        if (d?.raw)     { setRawCounts(d.raw);     setPerCategory(d.raw); }
        if (d?.cleaned) { setCleanedCounts(d.cleaned); }
        if (d?.total_raw > 0) setDatasetFetched(true);
      })
      .catch(() => {});
    // Load state flags only (not counts)
    getDatasetStatus()
      .then(res => {
        const d = res.data?.data || res.data;
        if (d?.cleaned)   setCleanDone(true);
        if (d?.annotated) setAnnotateDone(true);
      })
      .catch(() => {});
    // Load training status — do NOT use per_category from here (cleaned counts)
    getTrainingStatus()
      .then(res => {
        const d = res.data?.data || res.data;
        if (false && d?.per_category) setPerCategory(prev => ({ ...prev, ...d.per_category })); // intentionally disabled
        if (d?.status === 'completed' && d?.model_exists && !d?.is_stub) {
          console.log('[Engine] Train status: completed with real weights');
          setTrainDone(true);
        }
        if (d?.status === 'running') {
          setTrainLoading(true);
          setTrainProgress(d);
          startTrainPoll();
        }
      })
      .catch(() => {});
  }, [refreshEngine, startTrainPoll]);

  // ── Handlers ────────────────────────────────────────────────────────────────
  const handleFetch = async () => {
    setFetchLoading(true);
    setFetchResult(null);
    try {
      const res = await fetchDataset(datasetSlug || undefined, category, maxImages, fetchMode);
      const d   = res.data?.data || res.data;
      // Immediately apply counts from fetch response
      if (d?.per_category) { setRawCounts(d.per_category); setPerCategory(d.per_category); }
      const imported = d?.imported ?? d?.downloaded ?? 0;
      const total    = d?.total_in_dest ?? d?.total_images ?? '?';
      setFetchResult({
        imported,
        skipped:      d?.skipped  ?? 0,
        total,
        categoryPath: d?.category_path,
        message:      d?.message || `${imported} images imported`,
      });
      setDatasetFetched(true);
      toast.success(`Dataset fetched successfully — ${imported} images added to ${category.replace(/_/g,' ')}`);
      // Auto-refresh counts from server to ensure consistency
      setTimeout(refreshDatasetCounts, 300);
    } catch (err) {
      const msg = err.response?.data?.error || err.response?.data?.message || 'Fetch failed';
      setFetchResult({ error: msg });
      toast.error(msg);
    } finally {
      setFetchLoading(false);
    }
  };

  const handleClean = async () => {
    setCleanLoading(true);
    setCleanResult(null);
    try {
      const res = await cleanDataset();
      const d   = res.data?.data || res.data;
      if (d?.cleaned_per_category) setCleanedCounts(d.cleaned_per_category);
      const count   = d?.cleaned ?? d?.count ?? d?.images_cleaned ?? '?';
      const removed = d?.removed ?? 0;
      setCleanResult({ count, removed, message: d?.message || `${count} images cleaned` });
      setCleanDone(true);
      setAnnotateDone(false); // cleaning resets annotation
      toast.success(`Dataset cleaned — ${count} images saved to cleaned/`);
      setTimeout(refreshDatasetCounts, 300);
    } catch (err) {
      const msg = err.response?.data?.error || err.response?.data?.message || 'Cleaning failed';
      setCleanResult({ error: msg });
      toast.error(msg);
    } finally {
      setCleanLoading(false);
    }
  };

  const handleAnnotate = async () => {
    setAnnotateLoading(true);
    setAnnotateResult(null);
    try {
      const res = await annotateDataset();
      const d   = res.data?.data || res.data;
      setAnnotateResult({
        annotated:   d?.annotated   ?? 0,
        masks_saved: d?.masks_saved ?? 0,
        failed:      d?.failed      ?? 0,
        message:     d?.message || `${d?.annotated ?? 0} images annotated`,
      });
      setAnnotateDone(true);
      toast.success(`Annotation complete — ${d?.annotated ?? 0} keypoints, ${d?.masks_saved ?? 0} masks`);
      setTimeout(refreshDatasetCounts, 300);
    } catch (err) {
      const msg = err.response?.data?.error || err.response?.data?.message || 'Annotation failed';
      setAnnotateResult({ error: msg });
      toast.error(msg);
    } finally {
      setAnnotateLoading(false);
    }
  };

  const handleTestInference = async () => {
    console.log('[Engine] Running inference test via /api/render/test-inference …');
    setTestLoading(true);
    setTestResult(null);
    try {
      const res = await testInference();
      const d   = res.data?.data || res.data;
      setTestResult(d);
      if (d.passed) {
        console.log(`[Engine] ✅ Inference test PASSED — ${d.latency_ms}ms on ${d.engine}`);
        toast.success(`Inference test passed — ${d.latency_ms}ms`);
      } else {
        console.warn(`[Engine] ❌ Inference test FAILED — ${d.error}. Engine: ${d.engine}`);
        toast.error('Inference test failed — switched to geometric fallback');
      }
      refreshEngine();
    } catch (err) {
      const msg = err.response?.data?.error || 'Test inference request failed';
      console.error('[Engine] test-inference request error:', msg);
      setTestResult({ passed: false, error: msg, engine: 'geometric_fallback', latency_ms: 0 });
      toast.error(msg);
    } finally {
      setTestLoading(false);
    }
  };

  const handleTrain = async () => {
    setTrainLoading(true);
    setTrainResult(null);
    setTrainProgress(null);
    try {
      const res = await startTraining({
        draping_style:  draping.drapingStyle,
        epochs:         epochsInput,
        training_mode:  trainingMode,
      });
      const d = res.data?.data || res.data;
      toast.info('Training started — watching progress…');
      setTrainProgress(d);
      startTrainPoll();
    } catch (err) {
      const body = err.response?.data || {};
      const code = body.code || '';
      const msg  = body.error || body.message || 'Training failed';
      refreshDatasetCounts();
      if (code === 'KAGGLE_TRAINING_REQUIRED' || code === 'DATASET_TOO_LARGE_FOR_LOCAL') {
        setTrainResult({ kaggleRequired: true, error: msg });
        toast.warn('Use Kaggle GPU for full training. See instructions below.');
      } else if (code === 'INSUFFICIENT_DATASET') {
        setTrainResult({ error: msg, blocked: body.blocked });
        toast.error(msg);
      } else {
        setTrainResult({ error: msg });
        toast.error(msg);
      }
      setTrainLoading(false);
    }
  };

  // Clean up poll on unmount
  useEffect(() => () => stopTrainPoll(), [stopTrainPoll]);

  const handleGenerate = async () => {
    if (!draping.bodyImagePath)   return toast.error('Missing body image — go back to upload.');
    if (!draping.sareeImagePath)  return toast.error('Missing saree image — go back to upload.');
    if (!draping.blouseImagePath) return toast.error('Missing blouse image — go back to upload.');

    const engine = mlActive ? 'ml_model (verified)' : 'geometric_fallback';
    console.log(`[Generate] Starting draping — engine will be: ${engine}`);

    setGenLoading(true);
    try {
      const payload = {
        body_image_path:   draping.bodyImagePath,
        saree_image_path:  draping.sareeImagePath,
        blouse_image_path: draping.blouseImagePath,
        pose_data:         draping.poseData      || {},
        saree_features:    draping.sareeFeatures  || {},
        blouse_config:     draping.blouseConfig,
        draping_style:     draping.drapingStyle,
      };
      const res = await renderDraping(payload);
      const d   = res.data.data;
      console.log(`[Generate] ✅ Render complete — engine_used: ${d.engine_used}, ml_available: ${d.ml_model_available}`);
      draping.setOutputId(d.output_id);
      draping.setOutputPath(d.output_path);
      draping.setEngineUsed(d.engine_used);
      draping.setMlModelAvailable(d.ml_model_available);
      draping.setFallbackReason(d.fallback_reason || '');
      draping.setCurrentStep(3);
      toast.success('Saree draped successfully!');
      navigate('/preview');
    } catch (err) {
      toast.error(err.response?.data?.error || 'Rendering failed. Please try again.');
    } finally {
      setGenLoading(false);
    }
  };

  // ── Derived state ────────────────────────────────────────────────────────────
  const hasUploads      = !!(draping.bodyImagePath && draping.sareeImagePath && draping.blouseImagePath);
  const canGenerate     = hasUploads && !trainLoading;
  // Derived training readiness from CLEANED counts (not raw)
  const canTrain        = Object.keys(cleanedCounts).length === 3 &&
                          Object.values(cleanedCounts).every(c => c >= 50);
  // Real ML active = backend says available AND model loaded AND NOT a stub
  const mlActive        = !!(engineStatus?.ml_model_available && engineStatus?.model_loaded && !engineStatus?.is_stub);
  // Weights file exists but failed to load (corrupted / stub)
  const corruptedWeights = !!(engineStatus?.weights_exists && !engineStatus?.model_loaded && !mlActive);
  // Inference test passed (from testResult state)
  const inferenceVerified = !!(testResult?.passed);

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen py-10 px-4">
      <div className="max-w-6xl mx-auto">
        <StepIndicator current={2} />

        <div className="text-center mb-10">
          <h1 className="font-display text-4xl font-bold text-white mb-3">Customize Your Drape</h1>
          <p className="text-gray-400">Configure draping style and prepare the AI model</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* ── Left: Draping options ── */}
          <div className="lg:col-span-2 space-y-6">

            {/* Draping Style */}
            <div className="card p-6">
              <div className="flex items-center gap-2 mb-5">
                <Shirt className="w-5 h-5 text-saree-400" />
                <h2 className="text-white font-semibold">Draping Style</h2>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {DRAPING_STYLES.map(s => (
                  <button
                    key={s.id}
                    onClick={() => draping.setDrapingStyle(s.id)}
                    className={`p-4 rounded-xl border text-left transition-all ${
                      draping.drapingStyle === s.id
                        ? 'border-saree-500 bg-saree-500/10'
                        : 'border-white/10 bg-gray-800/40 hover:border-white/20'
                    }`}
                  >
                    <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${s.color} mb-2`} />
                    <p className="text-white font-medium text-sm">{s.label}</p>
                    <p className="text-gray-400 text-xs mt-0.5">{s.desc}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Neck Style */}
            <div className="card p-6">
              <div className="flex items-center gap-2 mb-5">
                <Scissors className="w-5 h-5 text-saree-400" />
                <h2 className="text-white font-semibold">Neck Style</h2>
              </div>
              <div className="flex flex-wrap gap-2">
                {NECK_TYPES.map(n => (
                  <button
                    key={n.id}
                    onClick={() => draping.updateBlouseConfig('neck_type', n.id)}
                    className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                      draping.blouseConfig.neck_type === n.id
                        ? 'bg-saree-600 text-white shadow-lg shadow-saree-500/20'
                        : 'bg-gray-800/60 text-gray-300 hover:bg-gray-700/60 border border-white/10'
                    }`}
                  >
                    {n.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Sleeve */}
            <div className="card p-6">
              <div className="flex items-center gap-2 mb-5">
                <Scissors className="w-5 h-5 text-saree-400" />
                <h2 className="text-white font-semibold">Sleeve Length</h2>
              </div>
              <div className="flex flex-wrap gap-2">
                {SLEEVE_TYPES.map(s => (
                  <button
                    key={s.id}
                    onClick={() => draping.updateBlouseConfig('sleeve_type', s.id)}
                    className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                      draping.blouseConfig.sleeve_type === s.id
                        ? 'bg-saree-600 text-white shadow-lg shadow-saree-500/20'
                        : 'bg-gray-800/60 text-gray-300 hover:bg-gray-700/60 border border-white/10'
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>

            {/* ── Model Preparation ── */}
            <div className="card p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                  <Database className="w-5 h-5 text-saree-400" />
                  <h2 className="text-white font-semibold">Model Preparation</h2>
                </div>
                <button
                  onClick={() => { refreshEngine(); refreshDatasetCounts(); }}
                  className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
                  title="Refresh engine status and live dataset counts"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-0">

                {/* Step 1 — Fetch Dataset */}
                <StepRow number={1} title="Fetch Dataset" done={datasetFetched}>
                  <div className="space-y-3">

                    {/* Kaggle slug */}
                    <input
                      type="text"
                      value={datasetSlug}
                      onChange={e => setDatasetSlug(e.target.value)}
                      placeholder="Kaggle slug — required (e.g. username/dataset-name)"
                      className={`w-full bg-gray-800/60 border rounded-xl px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none ${
                        datasetSlug.trim()
                          ? 'border-white/10 focus:border-saree-500/50'
                          : 'border-amber-500/30 focus:border-amber-500/60'
                      }`}
                    />
                    {!datasetSlug.trim() && (
                      <p className="text-xs text-amber-400/80 flex items-center gap-1.5">
                        <AlertTriangle className="w-3 h-3 flex-shrink-0" />
                        No real dataset downloaded yet. Enter a Kaggle slug to fetch real images.
                      </p>
                    )}

                    {/* Category + Max Images row */}
                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <select
                          value={category}
                          onChange={e => setCategory(e.target.value)}
                          className="w-full appearance-none bg-gray-800/60 border border-white/10 rounded-xl px-3 py-2 text-sm text-white focus:outline-none focus:border-saree-500/50 pr-8"
                        >
                          {CATEGORIES.map(c => (
                            <option key={c.id} value={c.id}>{c.label}</option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                      </div>
                      <input
                        type="number"
                        min={1} max={2000}
                        value={maxImages}
                        onChange={e => setMaxImages(Math.max(1, Math.min(2000, parseInt(e.target.value) || 150)))}
                        className="w-20 bg-gray-800/60 border border-white/10 rounded-xl px-3 py-2 text-sm text-white text-center focus:outline-none focus:border-saree-500/50"
                        title="Max images to import"
                      />
                    </div>

                    {/* Mode + Fetch button */}
                    <div className="flex gap-2 items-center">
                      <div className="flex bg-gray-800/60 border border-white/10 rounded-xl overflow-hidden text-xs">
                        {['append','replace'].map(m => (
                          <button
                            key={m}
                            onClick={() => setFetchMode(m)}
                            className={`px-3 py-2 capitalize transition-colors ${
                              fetchMode === m ? 'bg-saree-600 text-white' : 'text-gray-400 hover:text-white'
                            }`}
                          >{m}</button>
                        ))}
                      </div>
                      <span className="text-xs text-gray-500 flex-1">
                        {fetchMode === 'replace' ? 'Clears existing images first' : 'Adds to existing'}
                        {' · '}{maxImages} max
                      </span>
                      <button
                        onClick={handleFetch}
                        disabled={fetchLoading || !datasetSlug.trim()}
                        title={!datasetSlug.trim() ? 'Enter a Kaggle dataset slug first' : ''}
                        className="btn-secondary flex items-center gap-2 !py-2 !px-4 text-sm disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0"
                      >
                        {fetchLoading
                          ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Fetching...</>
                          : <><Download className="w-3.5 h-3.5" /> Fetch</>}
                      </button>
                    </div>

                    {/* Fetch result */}
                    {fetchResult && (
                      <div className={`text-xs px-3 py-2 rounded-lg ${
                        fetchResult.error
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                          : 'bg-green-500/10 text-green-400 border border-green-500/20'
                      }`}>
                        {fetchResult.error ? (
                          <span className="flex items-center gap-1">
                            <AlertTriangle className="w-3.5 h-3.5" /> {fetchResult.error}
                          </span>
                        ) : (
                          <div className="space-y-1">
                            <span className="flex items-center gap-1">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              <strong>{fetchResult.imported}</strong> imported
                              {fetchResult.skipped > 0 && <>, <span className="text-amber-400">{fetchResult.skipped} skipped</span></>}
                              {' · '} total in category: <strong>{fetchResult.total}</strong>
                            </span>
                            {fetchResult.categoryPath && (
                              <p className="text-gray-500 font-mono text-xs truncate" title={fetchResult.categoryPath}>
                                {fetchResult.categoryPath.split(/[\\/]/).slice(-3).join('/')}
                              </p>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {/* ── Card A: Local Prototype Dataset ── */}
                    <div className="rounded-xl border border-green-500/20 overflow-hidden">
                      <div className="bg-green-500/8 px-3 py-2.5 flex items-center justify-between">
                        <div className="flex items-center gap-1.5">
                          <MonitorSmartphone className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                          <span className="text-xs font-semibold text-green-300">Local Prototype Dataset</span>
                          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs bg-green-500/15 text-green-400 ml-1">active</span>
                        </div>
                        <span className="text-xs text-green-500/70">Used for local prototype demo</span>
                      </div>
                      <div className="bg-gray-900/50 px-3 py-3 space-y-2">
                        {[
                          ['Body Images',      'body_images'],
                          ['Saree Images',     'saree_images'],
                          ['Blouse Materials', 'blouse_materials'],
                        ].map(([label, key]) => {
                          const raw = rawCounts[key] ?? perCategory[key] ?? 0;
                          const cln = cleanedCounts[key] ?? 0;
                          const pct = Math.min(100, Math.round((raw / 150) * 100));
                          const ok  = raw >= 50;
                          return (
                            <div key={key}>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-400">{label}</span>
                                <span className={raw === 0 ? 'text-gray-600' : ok ? 'text-green-400' : 'text-amber-400'}>
                                  {raw === 0 ? (
                                    <span className="text-gray-600">no images yet</span>
                                  ) : (
                                    <span className="flex items-center gap-1.5">
                                      <span className="font-semibold tabular-nums">{raw}</span>
                                      <span className="text-gray-600">raw</span>
                                      {cln > 0 && (
                                        <span className="text-blue-400/80">· {cln} cleaned</span>
                                      )}
                                      {ok
                                        ? <CheckCircle2 className="w-3 h-3 text-green-500" />
                                        : <span className="text-amber-500">need 50</span>}
                                    </span>
                                  )}
                                </span>
                              </div>
                              <div className="h-1 bg-gray-700/60 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${
                                    raw === 0 ? 'bg-transparent' : ok ? 'bg-green-500' : 'bg-amber-500'
                                  }`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                        <div className="pt-1 flex items-center justify-between">
                          <span className="text-xs text-gray-600">
                            Total: {Object.values(rawCounts).length > 0
                              ? Object.values(rawCounts).reduce((a, b) => a + b, 0).toLocaleString()
                              : '—'} images
                          </span>
                          <span className="text-xs text-gray-600">min 50 · 150 recommended</span>
                        </div>
                      </div>
                    </div>

                    {/* ── Card B: Colab GPU Training Dataset ── */}
                    <div className="rounded-xl border border-blue-500/20 overflow-hidden">
                      <div className="bg-blue-500/8 px-3 py-2.5 flex items-center justify-between">
                        <div className="flex items-center gap-1.5">
                          <Cloud className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
                          <span className="text-xs font-semibold text-blue-300">Colab GPU Training Dataset</span>
                        </div>
                        <span className="text-xs text-blue-500/70">Prepared in Google Colab for GPU training</span>
                      </div>
                      <div className="bg-gray-900/50 px-3 py-3 space-y-2">
                        {[
                          ['Body Images',      50000, 50000],
                          ['Saree Images',     35000, 50000],
                          ['Blouse Materials', 50000, 50000],
                        ].map(([label, count, cap]) => (
                          <div key={label}>
                            <div className="flex justify-between text-xs mb-1">
                              <span className="text-gray-400">{label}</span>
                              <span className="text-blue-300 font-semibold tabular-nums">{count.toLocaleString()}</span>
                            </div>
                            <div className="h-1 bg-gray-700/60 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full bg-blue-500/70 transition-all"
                                style={{ width: `${Math.round((count / cap) * 100)}%` }}
                              />
                            </div>
                          </div>
                        ))}
                        <div className="pt-1 flex items-center justify-between">
                          <span className="text-xs text-gray-600">Total: 135,000 images</span>
                          <a
                            href="https://colab.research.google.com"
                            target="_blank" rel="noopener noreferrer"
                            className="text-xs text-blue-500 hover:text-blue-400 flex items-center gap-1 transition-colors"
                          >
                            <ExternalLink className="w-3 h-3" /> Open Colab
                          </a>
                        </div>
                      </div>
                    </div>

                    {/* Explanatory note */}
                    <div className="flex items-start gap-2 bg-gray-800/40 rounded-lg px-3 py-2.5">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-400/70 flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-gray-500 leading-relaxed">
                        Large-scale dataset is prepared in Colab; local app uses sample data for prototype demo.
                      </p>
                    </div>

                  </div>
                </StepRow>

                {/* Step 2 — Clean Dataset */}
                <StepRow number={2} title="Clean Dataset" done={cleanDone}>
                  <div className="space-y-3">
                    <p className="text-xs text-gray-400">
                      Validate, resize to 512×768, normalise brightness (CLAHE), convert to PNG.
                      Saves clean copies into <code className="text-gray-500">dataset/cleaned/</code>
                    </p>
                    {!datasetFetched && (
                      <p className="text-xs text-amber-400/80 flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3" /> Fetch real images first.
                      </p>
                    )}
                    <button
                      onClick={handleClean}
                      disabled={cleanLoading || !datasetFetched}
                      className="btn-secondary flex items-center gap-2 !py-2 !px-4 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {cleanLoading
                        ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Cleaning…</>
                        : <><Sparkles className="w-3.5 h-3.5" /> Clean Dataset</>}
                    </button>
                    {cleanResult && (
                      <div className={`text-xs px-3 py-2 rounded-lg ${
                        cleanResult.error
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                          : 'bg-green-500/10 text-green-400 border border-green-500/20'
                      }`}>
                        {cleanResult.error
                          ? <span className="flex items-center gap-1"><AlertTriangle className="w-3.5 h-3.5" /> {cleanResult.error}</span>
                          : <span className="flex items-center gap-1 flex-wrap gap-y-0.5">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              <strong>{cleanResult.count}</strong> saved to cleaned/
                              {cleanResult.removed > 0 && <span className="text-amber-400">, {cleanResult.removed} rejected</span>}
                            </span>
                        }
                      </div>
                    )}
                    {/* Cleaned counts per category */}
                    {Object.keys(cleanedCounts).length > 0 && (
                      <div className="space-y-1">
                        {Object.entries(cleanedCounts).map(([cat, cnt]) => (
                          <div key={cat} className="flex justify-between text-xs">
                            <span className="text-gray-500 capitalize">{cat.replace(/_/g,' ')}</span>
                            <span className={cnt >= 50 ? 'text-green-400' : 'text-amber-400'}>
                              {cnt} cleaned {cnt >= 50 ? '✓' : ''}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </StepRow>

                {/* Step 3 — Annotate Dataset */}
                <StepRow number={3} title="Annotate Dataset" done={annotateDone}>
                  <div className="space-y-3">
                    <p className="text-xs text-gray-400">
                      MediaPipe Pose keypoints on body images. Binary segmentation masks for saree
                      &amp; blouse. Saves to <code className="text-gray-500">dataset/annotations/</code>
                    </p>
                    {!cleanDone && (
                      <p className="text-xs text-amber-400/80 flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3" /> Clean dataset before annotating.
                      </p>
                    )}
                    <button
                      onClick={handleAnnotate}
                      disabled={annotateLoading || !cleanDone}
                      className="btn-secondary flex items-center gap-2 !py-2 !px-4 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      {annotateLoading
                        ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Annotating…</>
                        : <><Database className="w-3.5 h-3.5" /> Annotate Dataset</>}
                    </button>
                    {annotateResult && (
                      <div className={`text-xs px-3 py-2 rounded-lg ${
                        annotateResult.error
                          ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                          : 'bg-green-500/10 text-green-400 border border-green-500/20'
                      }`}>
                        {annotateResult.error
                          ? <span className="flex items-center gap-1"><AlertTriangle className="w-3.5 h-3.5" /> {annotateResult.error}</span>
                          : <span className="flex items-center gap-1">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              <strong>{annotateResult.annotated}</strong> keypoints
                              {annotateResult.masks_saved > 0 && <>, <strong>{annotateResult.masks_saved}</strong> masks</>}
                              {annotateResult.failed > 0 && <span className="text-amber-400">, {annotateResult.failed} failed</span>}
                            </span>
                        }
                      </div>
                    )}
                  </div>
                </StepRow>

                {/* Step 4 — Train Model */}
                <StepRow number={4} title="Train Model" done={trainDone}>
                  <div className="space-y-3">

                    {/* ── Training Mode Selector ── */}
                    <div className="bg-gray-800/50 border border-white/8 rounded-xl p-3 space-y-2">
                      <p className="text-xs text-gray-400 font-medium mb-2">Training Mode</p>
                      <div className="grid grid-cols-2 gap-2">
                        {/* Prototype */}
                        <button
                          onClick={() => setTrainingMode('prototype')}
                          className={`flex flex-col items-start gap-1 p-3 rounded-xl border text-left transition-all ${
                            trainingMode === 'prototype'
                              ? 'border-saree-500 bg-saree-500/10'
                              : 'border-white/10 bg-gray-800/40 hover:border-white/20'
                          }`}
                        >
                          <div className="flex items-center gap-1.5">
                            <FlaskConical className={`w-3.5 h-3.5 ${trainingMode === 'prototype' ? 'text-saree-400' : 'text-gray-500'}`} />
                            <span className={`text-xs font-semibold ${trainingMode === 'prototype' ? 'text-saree-300' : 'text-gray-300'}`}>
                              Prototype Local
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 leading-tight">
                            150–500 imgs · CPU · pipeline test
                          </p>
                        </button>
                        {/* Kaggle Full */}
                        <button
                          onClick={() => setTrainingMode('kaggle_full')}
                          className={`flex flex-col items-start gap-1 p-3 rounded-xl border text-left transition-all ${
                            trainingMode === 'kaggle_full'
                              ? 'border-blue-500 bg-blue-500/10'
                              : 'border-white/10 bg-gray-800/40 hover:border-white/20'
                          }`}
                        >
                          <div className="flex items-center gap-1.5">
                            <Cloud className={`w-3.5 h-3.5 ${trainingMode === 'kaggle_full' ? 'text-blue-400' : 'text-gray-500'}`} />
                            <span className={`text-xs font-semibold ${trainingMode === 'kaggle_full' ? 'text-blue-300' : 'text-gray-300'}`}>
                              Full Kaggle GPU
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 leading-tight">
                            150k imgs · T4/P100 · production
                          </p>
                        </button>
                      </div>
                    </div>

                    {/* ── Kaggle Full instructions panel ── */}
                    {trainingMode === 'kaggle_full' && (
                      <div className="bg-blue-500/8 border border-blue-500/25 rounded-xl p-4 space-y-3">
                        <div className="flex items-center gap-2">
                          <Cloud className="w-4 h-4 text-blue-400 flex-shrink-0" />
                          <p className="text-sm font-semibold text-blue-300">Kaggle GPU Training Required</p>
                        </div>
                        <p className="text-xs text-blue-300/80 leading-relaxed">
                          150k-image training requires a GPU (T4/P100). Run the provided notebook on Kaggle, then download
                          the weights and place them locally.
                        </p>
                        <ol className="text-xs text-gray-300 space-y-2 pl-1">
                          <li className="flex gap-2"><span className="text-blue-400 font-bold flex-shrink-0">1.</span> Open <code className="bg-black/30 px-1 rounded text-blue-300">notebooks/full_saree_training_150k.ipynb</code> in Kaggle</li>
                          <li className="flex gap-2"><span className="text-blue-400 font-bold flex-shrink-0">2.</span> Set runtime to <strong>GPU T4 × 2</strong> and enable Internet access</li>
                          <li className="flex gap-2"><span className="text-blue-400 font-bold flex-shrink-0">3.</span> Run all cells — training takes ~4–8 hours</li>
                          <li className="flex gap-2"><span className="text-blue-400 font-bold flex-shrink-0">4.</span> Download <code className="bg-black/30 px-1 rounded text-blue-300">saree_tryon_model.pth</code> and <code className="bg-black/30 px-1 rounded text-blue-300">model_config.json</code> from output</li>
                          <li className="flex gap-2"><span className="text-blue-400 font-bold flex-shrink-0">5.</span> Place both files in <code className="bg-black/30 px-1 rounded text-blue-300">backend/dataset/trained_models/</code></li>
                          <li className="flex gap-2"><span className="text-blue-400 font-bold flex-shrink-0">6.</span> Restart the Flask backend — engine will auto-detect the weights</li>
                        </ol>
                        <a
                          href="https://www.kaggle.com/code"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 transition-colors"
                        >
                          <ExternalLink className="w-3 h-3" /> Open Kaggle
                        </a>
                      </div>
                    )}

                    {/* Dataset readiness per-category with tier labels — only in prototype mode */}
                    {trainingMode === 'prototype' && Object.keys(perCategory).length > 0 && (
                      <div className="bg-gray-800/40 rounded-xl px-3 py-3 space-y-2.5">
                        <div className="flex items-center justify-between mb-1">
                          <p className="text-xs text-gray-400 font-medium">Dataset readiness</p>
                          <span className="text-xs text-gray-600">
                            {Object.values(perCategory).reduce((a,b)=>a+b,0)} total imgs
                          </span>
                        </div>
                        {Object.entries(perCategory).map(([cat, cnt]) => {
                          const tier = getTier(cnt);
                          const pct  = Math.min(100, Math.round((cnt / 500) * 100));
                          return (
                            <div key={cat}>
                              <div className="flex justify-between text-xs mb-1">
                                <span className="text-gray-400 capitalize">{cat.replace(/_/g,' ')}</span>
                                <span className={`font-medium ${tier.color}`}>
                                  {cnt} — {tier.label}
                                </span>
                              </div>
                              <div className="relative h-1.5 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${tier.bg}`}
                                  style={{ width: `${pct}%` }}
                                />
                                <div className="absolute top-0 left-[10%] w-px h-full bg-white/20" title="min 50" />
                                <div className="absolute top-0 left-[30%] w-px h-full bg-white/20" title="150" />
                              </div>
                            </div>
                          );
                        })}
                        <div className="flex gap-3 text-xs text-gray-600 pt-1 flex-wrap">
                          <span>▏ 50 = min</span>
                          <span>▏ 150 = prototype</span>
                          <span>▏ 500 = max local</span>
                          <span className="text-blue-500">▏ 5000+ = use Kaggle</span>
                        </div>
                      </div>
                    )}

                    {/* Kaggle dataset suggestions */}
                    <details className="group">
                      <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300 flex items-center gap-1.5 select-none">
                        <BookOpen className="w-3.5 h-3.5" />
                        Recommended Kaggle datasets
                        <ChevronDown className="w-3 h-3 group-open:rotate-180 transition-transform ml-auto" />
                      </summary>
                      <div className="mt-2 space-y-1.5">
                        {KAGGLE_DATASETS.map(ds => (
                          <a
                            key={ds.label}
                            href={ds.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center justify-between px-3 py-2 rounded-lg bg-gray-800/60 border border-white/5 hover:border-saree-500/30 text-xs text-gray-300 hover:text-white transition-colors group/link"
                          >
                            <span>
                              <span className="text-gray-500 capitalize mr-1.5">
                                [{ds.category.replace(/_/g,' ')}]
                              </span>
                              {ds.label}
                            </span>
                            <ExternalLink className="w-3 h-3 text-gray-500 group-hover/link:text-saree-400" />
                          </a>
                        ))}
                        <p className="text-xs text-gray-600 px-1">
                          Copy the slug (username/dataset-name) into Fetch Dataset above.
                        </p>
                      </div>
                    </details>

                    {/* Insufficient dataset warning — prototype mode only */}
                    {trainingMode === 'prototype' && Object.keys(perCategory).length > 0 &&
                      Object.values(perCategory).some(c => c < 50) && (
                      <div className="bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-2.5">
                        <p className="text-xs text-red-400 flex items-start gap-1.5 font-medium mb-1">
                          <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                          Minimum 50 images required per category
                        </p>
                        <p className="text-xs text-red-400/70 pl-5">
                          Fetch more datasets above before training.
                        </p>
                      </div>
                    )}

                    {/* Fetch datasets prompt if not ready — prototype only */}
                    {trainingMode === 'prototype' && !canTrain && !trainLoading && Object.keys(perCategory).length > 0 && (
                      <p className="text-xs text-amber-400/80 flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3" /> Fetch datasets before training.
                      </p>
                    )}

                    {/* Epoch count input + Train button */}
                    {!trainLoading && (
                      <div className="flex items-center gap-2">
                        {trainingMode === 'prototype' && (
                          <div className="flex items-center gap-2 flex-1">
                            <label className="text-xs text-gray-400 whitespace-nowrap">Epochs</label>
                            <input
                              type="number" min={1} max={200}
                              value={epochsInput}
                              onChange={e => setEpochsInput(Math.max(1, Math.min(200, parseInt(e.target.value)||10)))}
                              className="w-16 bg-gray-800/60 border border-white/10 rounded-lg px-2 py-1.5 text-sm text-white text-center focus:outline-none focus:border-saree-500/50"
                            />
                          </div>
                        )}
                        {trainingMode === 'kaggle_full' ? (
                          <div className="flex-1 text-xs text-blue-400/70 flex items-center gap-1.5">
                            <Cloud className="w-3.5 h-3.5" />
                            Local training disabled — use Kaggle notebook above
                          </div>
                        ) : (
                          <button
                            onClick={handleTrain}
                            disabled={trainLoading || !annotateDone || !canTrain}
                            className="btn-primary flex items-center gap-2 !py-2 !px-4 text-sm disabled:opacity-40 disabled:cursor-not-allowed"
                            title={
                              !cleanDone    ? 'Clean dataset first' :
                              !annotateDone ? 'Annotate dataset first' :
                              !canTrain     ? 'Need 50+ cleaned images per category' : ''
                            }
                          >
                            <Wand2 className="w-3.5 h-3.5" /> Train Prototype
                          </button>
                        )}
                      </div>
                    )}

                    {/* Live training progress */}
                    {trainLoading && trainProgress && (
                      <div className="space-y-2.5">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-blue-300 flex items-center gap-1.5">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            {trainProgress.message || 'Training…'}
                          </span>
                          <span className="text-gray-500">
                            {trainProgress.eta_seconds != null && `ETA ${fmtEta(trainProgress.eta_seconds)}`}
                          </span>
                        </div>
                        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-saree-600 to-purple-500 rounded-full transition-all duration-700"
                            style={{ width: `${trainProgress.progress || 0}%` }}
                          />
                        </div>
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span>
                            {trainProgress.epoch > 0
                              ? `Epoch ${trainProgress.epoch}/${trainProgress.total_epochs}`
                              : 'Initialising…'}
                          </span>
                          <div className="flex items-center gap-3">
                            {trainProgress.loss != null && (
                              <span className="flex items-center gap-1">
                                <TrendingUp className="w-3 h-3" />
                                loss={trainProgress.loss}
                              </span>
                            )}
                            {trainProgress.best_loss != null && (
                              <span className="text-green-500">best={trainProgress.best_loss}</span>
                            )}
                            <span>{trainProgress.progress || 0}%</span>
                          </div>
                        </div>
                        {trainProgress.latest_checkpoint && (
                          <p className="text-xs text-gray-600">
                            Last checkpoint: {trainProgress.latest_checkpoint}
                          </p>
                        )}
                      </div>
                    )}

                    {/* Train result */}
                    {trainResult && !trainLoading && (
                      <div className={`text-xs px-3 py-2.5 rounded-xl border space-y-1 ${
                        trainResult.kaggleRequired
                          ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                          : trainResult.error
                            ? 'bg-red-500/10 text-red-400 border-red-500/20'
                            : trainResult.isReal
                              ? 'bg-green-500/10 text-green-400 border-green-500/20'
                              : 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                      }`}>
                        {trainResult.kaggleRequired ? (
                          <span className="flex items-start gap-1.5">
                            <Cloud className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                            Use Kaggle notebook for full training. See instructions above.
                          </span>
                        ) : trainResult.error ? (
                          <span className="flex items-start gap-1.5">
                            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                            {trainResult.error}
                          </span>
                        ) : (
                          <>
                            <span className="flex items-center gap-1.5">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              {trainResult.isReal
                                ? (trainProgress?.is_prototype
                                    ? 'Prototype model trained — pipeline verified!'
                                    : 'Real model trained and saved!')
                                : 'Training complete'}
                            </span>
                            {trainResult.isReal && trainResult.loss != null && (
                              <p className="pl-5 text-gray-400">
                                {trainResult.epochs} epochs · best loss={trainResult.loss}
                                {trainProgress?.is_prototype && (
                                  <span className="ml-2 text-amber-400/80">[prototype]</span>
                                )}
                              </p>
                            )}
                          </>
                        )}
                      </div>
                    )}

                    {/* Training pending banner — shown when no weights file exists and not currently training */}
                    {!trainDone && !trainLoading && !trainResult && (
                      <div className="flex items-start gap-2 bg-amber-500/8 border border-amber-500/20 rounded-xl px-3 py-2.5">
                        <Cpu className="w-3.5 h-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="text-xs text-amber-300 font-medium">
                            Model training pending GPU / trained weights import
                          </p>
                          <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">
                            Run <code className="text-gray-400">notebooks/colab_saree_training_10k.ipynb</code> in Colab,
                            then copy <code className="text-gray-400">saree_tryon_model.pth</code> to{' '}
                            <code className="text-gray-400">backend/dataset/trained_models/</code> and restart Flask.
                          </p>
                        </div>
                      </div>
                    )}

                    {!cleanDone && !trainLoading && (
                      <p className="text-xs text-gray-600">
                        Complete steps 1 &amp; 2 (Fetch + Clean) before training.
                      </p>
                    )}

                  </div>
                </StepRow>

              </div>
            </div>
          </div>

          {/* ── Right: config summary + engine ── */}
          <div className="space-y-6">

            {/* Engine Status Card — real verified data from backend */}
            <div className={`card p-5 border ${
              mlActive
                ? 'border-green-500/20 bg-green-500/5'
                : corruptedWeights
                  ? 'border-red-500/20 bg-red-500/5'
                  : 'border-amber-500/20 bg-amber-500/5'
            }`}>
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  {mlActive
                    ? <Zap className="w-5 h-5 text-green-400" />
                    : corruptedWeights
                      ? <AlertTriangle className="w-5 h-5 text-red-400" />
                      : <Cpu className="w-5 h-5 text-amber-400" />}
                  <div>
                    <p className={`text-sm font-semibold ${
                      mlActive ? 'text-green-300'
                        : corruptedWeights ? 'text-red-300'
                        : 'text-amber-300'
                    }`}>
                      {mlActive
                        ? 'ML Model Active'
                        : corruptedWeights
                          ? 'Invalid weights — using fallback'
                          : 'Geometric Engine'}
                    </p>
                    <p className="text-xs text-gray-400 mt-0.5">
                      {mlActive
                        ? `Verified — ${engineStatus?.param_count_m ?? '?'}M params on ${engineStatus?.device ?? 'cpu'}`
                        : testResult && !testResult.passed
                          ? 'Invalid weights — using geometric fallback'
                          : corruptedWeights
                            ? 'Weights found but failed to load'
                            : 'No valid weights — OpenCV fallback'}
                    </p>
                  </div>
                </div>
                <button
                  onClick={refreshEngine}
                  className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
                  title="Re-check engine"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Status checklist */}
              <div className="space-y-1.5 mb-4">
                <StatusBadge ok={engineStatus?.weights_exists}     label="Weights file exists"   />
                <StatusBadge ok={engineStatus?.model_loaded}       label="torch.load() passed"   />
                <StatusBadge ok={!engineStatus?.is_stub}           label="Real trained weights"  />
                <StatusBadge ok={mlActive}                         label="ML Model Active"       />
                <StatusBadge ok={inferenceVerified && testResult?.passed} label="Inference test passed" />
              </div>

              {/* Inference result banner */}
              {testResult && (
                <div className={`rounded-lg px-3 py-2 mb-3 border text-xs ${
                  testResult.passed
                    ? 'bg-green-500/10 border-green-500/20 text-green-400'
                    : 'bg-red-500/10 border-red-500/20 text-red-400'
                }`}>
                  {testResult.passed
                    ? <span className="flex items-center gap-1"><CheckCircle2 className="w-3.5 h-3.5" /> ML Model Active — inference {testResult.latency_ms}ms on {testResult.engine}</span>
                    : <span className="flex items-center gap-1"><XCircle className="w-3.5 h-3.5" /> Invalid weights — using geometric fallback</span>}
                </div>
              )}

              {/* Fallback / error reason */}
              {!mlActive && engineStatus?.fallback_reason && (
                <div className={`rounded-lg px-3 py-2 mb-4 border ${
                  corruptedWeights
                    ? 'bg-red-500/10 border-red-500/20'
                    : 'bg-amber-500/10 border-amber-500/20'
                }`}>
                  <p className={`text-xs leading-relaxed ${
                    corruptedWeights ? 'text-red-400' : 'text-amber-400'
                  }`}>
                    {engineStatus.fallback_reason}
                  </p>
                  {engineStatus.fallback_reason?.toLowerCase().includes('pytorch') && (
                    <p className="text-xs text-gray-500 mt-1">
                      Install PyTorch: <code className="bg-black/30 px-1 rounded">pip install torch torchvision</code>
                    </p>
                  )}
                </div>
              )}

              {/* Test Inference button */}
              <button
                onClick={handleTestInference}
                disabled={testLoading}
                className="w-full btn-secondary flex items-center justify-center gap-2 !py-2 text-xs disabled:opacity-40 disabled:cursor-not-allowed mb-3"
              >
                {testLoading
                  ? <><Loader2 className="w-3 h-3 animate-spin" /> Testing...</>
                  : <><Zap className="w-3 h-3" /> Test Inference</>}
              </button>

              {/* Test result */}
              {testResult && (
                <div className={`text-xs px-3 py-2 rounded-lg mb-3 ${
                  testResult.passed
                    ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                    : 'bg-red-500/10 text-red-400 border border-red-500/20'
                }`}>
                  {testResult.passed ? (
                    <span className="flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Passed — {testResult.latency_ms}ms · engine: {testResult.engine}
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 flex-wrap">
                      <XCircle className="w-3.5 h-3.5 flex-shrink-0" />
                      {testResult.error || 'Failed'}
                    </span>
                  )}
                </div>
              )}

              {/* Debug Details */}
              <details className="group">
                <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-300 flex items-center gap-1 select-none">
                  <ChevronDown className="w-3 h-3 group-open:rotate-180 transition-transform" />
                  Debug Details
                </summary>
                <div className="mt-2 space-y-1.5 text-xs">
                  {[
                    ['weights_exists',        String(!!engineStatus?.weights_exists)],
                    ['weights_path',          engineStatus?.weights_path ? engineStatus.weights_path.split(/[\\/]/).slice(-3).join('/') : '—'],
                    ['weights_size_kb',       engineStatus?.weights_size_kb != null ? `${engineStatus.weights_size_kb} KB` : '—'],
                    ['model_loaded',          String(!!engineStatus?.model_loaded)],
                    ['ml_model_available',    String(!!engineStatus?.ml_model_available)],
                    ['is_stub',               String(!!engineStatus?.is_stub)],
                    ['model_type',            engineStatus?.model_type || 'SareeTryOnModel'],
                    ['device',                engineStatus?.device || '—'],
                    ['param_count_m',         engineStatus?.param_count_m ? `${engineStatus.param_count_m}M` : '—'],
                    ['engine_used',           engineStatus?.engine || 'geometric_fallback'],
                    ['inference_test_passed',  testResult ? String(testResult.passed) : 'not run'],
                    ['inference_latency_ms',  testResult?.latency_ms ? `${testResult.latency_ms} ms` : '—'],
                    ['fallback_reason',       engineStatus?.fallback_reason || '—'],
                  ].map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2 py-0.5 border-b border-white/5 last:border-0">
                      <span className="text-gray-500 font-mono shrink-0">{k}</span>
                      <span className={`font-mono text-right truncate max-w-[130px] ${
                        v === 'true'  ? 'text-green-400' :
                        v === 'false' ? 'text-red-400'   : 'text-gray-300'
                      }`} title={v}>{v}</span>
                    </div>
                  ))}
                  {engineStatus?.log && (
                    <div className="mt-2 bg-black/30 rounded px-2 py-1.5 text-gray-400 leading-relaxed break-all border border-white/5">
                      {engineStatus.log}
                    </div>
                  )}
                </div>
              </details>
            </div>

            {/* Configuration Summary */}
            <div className="card p-5">
              <h2 className="text-white font-semibold mb-4 text-sm">Configuration</h2>
              <div className="space-y-3 text-sm">
                {[
                  ['Draping Style', DRAPING_STYLES.find(s => s.id === draping.drapingStyle)?.label   || '—'],
                  ['Neck Style',    NECK_TYPES.find(n => n.id === draping.blouseConfig?.neck_type)?.label   || '—'],
                  ['Sleeve',        SLEEVE_TYPES.find(s => s.id === draping.blouseConfig?.sleeve_type)?.label || '—'],
                  ['Fabric',        draping.fabricType || '—'],
                  ['Engine',        mlActive ? 'ML Model' : corruptedWeights ? 'Invalid weights' : 'Geometric'],
                ].map(([key, val]) => (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-gray-400">{key}</span>
                    <span className={`capitalize font-medium ${
                      key === 'Engine'
                        ? mlActive ? 'text-green-300' : corruptedWeights ? 'text-red-300' : 'text-amber-300'
                        : 'text-white'
                    }`}>{val}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Fabric */}
            <div className="card p-5">
              <h2 className="text-white font-semibold mb-3 text-sm">Fabric Type</h2>
              <div className="grid grid-cols-2 gap-2">
                {FABRIC_TYPES.map(f => (
                  <button
                    key={f.id}
                    onClick={() => draping.setFabricType?.(f.id)}
                    className={`px-3 py-2 rounded-xl text-xs font-medium transition-all text-center ${
                      draping.fabricType === f.id
                        ? 'bg-saree-600 text-white'
                        : 'bg-gray-800/60 text-gray-300 hover:bg-gray-700/60 border border-white/10'
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

          </div>
        </div>

        {/* ── Bottom nav ── */}
        <div className="mt-8 flex items-center justify-between">
          <button onClick={() => navigate('/upload')} className="btn-secondary flex items-center gap-2">
            <ArrowLeft className="w-4 h-4" /> Back
          </button>

          <button
            onClick={handleGenerate}
            disabled={genLoading || !hasUploads || trainLoading}
            className="btn-primary flex items-center gap-2 !px-8 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
            title={trainLoading ? 'Training in progress — please wait' : ''}
          >
            {genLoading ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Generating drape...</>
            ) : trainLoading ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Training...</>
            ) : (
              <>Generate Draping <ArrowRight className="w-4 h-4" /></>
            )}
          </button>
        </div>

        {!canGenerate && !genLoading && hasUploads && !trainDone && (
          <p className="text-center text-xs text-amber-400/70 mt-3">
            Tip: You can generate with the geometric engine even without training — it always works.
          </p>
        )}
      </div>
    </div>
  );
}
