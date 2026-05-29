import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, Shirt, Scissors, ArrowRight, Loader2, CheckCircle2 } from 'lucide-react';
import { toast } from 'react-toastify';
import StepIndicator from '../components/StepIndicator';
import UploadZone from '../components/UploadZone';
import { useDraping } from '../context/DrapingContext';
import { uploadBody, uploadSaree, uploadBlouse, detectPose, processSaree } from '../services/api';

const FABRIC_TYPES = ['silk', 'cotton', 'georgette', 'chiffon'];

export default function UploadPage() {
  const navigate  = useNavigate();
  const draping   = useDraping();
  const [loading, setLoading] = useState(false);
  const [stage,   setStage]   = useState(''); // feedback message during processing

  // ── Handle individual file uploads ──────────────────────────────────────────
  const handleBodyFile = async file => {
    draping.setBodyPreview(URL.createObjectURL(file));
    try {
      const res = await uploadBody(file);
      draping.setBodyImagePath(res.data.data.file_path);
      toast.success('Body image uploaded');
    } catch (err) {
      toast.error(err.response?.data?.error || 'Body upload failed');
      draping.setBodyPreview(null);
    }
  };

  const handleSareeFile = async file => {
    draping.setSareePreview(URL.createObjectURL(file));
    try {
      const res = await uploadSaree(file);
      draping.setSareeImagePath(res.data.data.file_path);
      toast.success('Saree image uploaded');
    } catch (err) {
      toast.error(err.response?.data?.error || 'Saree upload failed');
      draping.setSareePreview(null);
    }
  };

  const handleBlouseFile = async file => {
    draping.setBlousePreview(URL.createObjectURL(file));
    try {
      const res = await uploadBlouse(file);
      draping.setBlouseImagePath(res.data.data.file_path);
      toast.success('Blouse image uploaded');
    } catch (err) {
      toast.error(err.response?.data?.error || 'Blouse upload failed');
      draping.setBlousePreview(null);
    }
  };

  // ── Proceed: run pose detection + saree feature extraction ──────────────────
  const handleContinue = async () => {
    if (!draping.bodyImagePath)   return toast.error('Please upload a body image');
    if (!draping.sareeImagePath)  return toast.error('Please upload a saree image');
    if (!draping.blouseImagePath) return toast.error('Please upload a blouse material');

    setLoading(true);
    try {
      // Pose detection
      setStage('Detecting body pose...');
      const poseRes = await detectPose(draping.bodyImagePath);
      draping.setPoseData(poseRes.data.data);

      // Saree feature extraction
      setStage('Analysing saree fabric...');
      const sareeRes = await processSaree(draping.sareeImagePath, draping.fabricType);
      draping.setSareeFeatures(sareeRes.data.data);

      draping.setCurrentStep(2);
      toast.success('Images processed! Customize your look.');
      navigate('/customize');
    } catch (err) {
      toast.error(err.response?.data?.error || 'Processing failed. Please try again.');
    } finally {
      setLoading(false);
      setStage('');
    }
  };

  const allUploaded = draping.bodyImagePath && draping.sareeImagePath && draping.blouseImagePath;

  return (
    <div className="min-h-screen py-10 px-4">
      <div className="max-w-4xl mx-auto">
        <StepIndicator current={1} />

        <div className="text-center mb-10">
          <h1 className="font-display text-4xl font-bold text-white mb-3">Upload Your Images</h1>
          <p className="text-gray-400">Upload a clear full-body photo, saree fabric, and blouse material</p>
        </div>

        {/* Upload grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="card p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="step-badge">1</div>
              <div>
                <h3 className="text-white font-semibold text-sm">Body Photo</h3>
                <p className="text-gray-400 text-xs">Full-body, front-facing</p>
              </div>
            </div>
            <UploadZone
              label="Upload Body Image"
              preview={draping.bodyPreview}
              onFile={handleBodyFile}
              onRemove={() => { draping.setBodyPreview(null); draping.setBodyImagePath(null); }}
              loading={loading}
              icon={User}
            />
          </div>

          <div className="card p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="step-badge">2</div>
              <div>
                <h3 className="text-white font-semibold text-sm">Saree Fabric</h3>
                <p className="text-gray-400 text-xs">Flat or draped photo</p>
              </div>
            </div>
            <UploadZone
              label="Upload Saree Image"
              preview={draping.sareePreview}
              onFile={handleSareeFile}
              onRemove={() => { draping.setSareePreview(null); draping.setSareeImagePath(null); }}
              loading={loading}
              icon={Shirt}
            />
          </div>

          <div className="card p-5">
            <div className="flex items-center gap-2 mb-4">
              <div className="step-badge">3</div>
              <div>
                <h3 className="text-white font-semibold text-sm">Blouse Material</h3>
                <p className="text-gray-400 text-xs">Fabric or colour swatch</p>
              </div>
            </div>
            <UploadZone
              label="Upload Blouse Material"
              preview={draping.blousePreview}
              onFile={handleBlouseFile}
              onRemove={() => { draping.setBlousePreview(null); draping.setBlouseImagePath(null); }}
              loading={loading}
              icon={Scissors}
            />
          </div>
        </div>

        {/* Fabric type selector */}
        <div className="card p-5 mb-8">
          <h3 className="text-white font-semibold mb-3">Saree Fabric Type</h3>
          <div className="flex flex-wrap gap-3">
            {FABRIC_TYPES.map(fab => (
              <button
                key={fab}
                onClick={() => draping.setFabricType(fab)}
                className={`px-4 py-2 rounded-xl text-sm font-medium capitalize transition-all ${
                  draping.fabricType === fab
                    ? 'bg-saree-600 text-white shadow-lg shadow-saree-500/20'
                    : 'bg-gray-800/60 text-gray-300 hover:bg-gray-700/60 border border-white/10'
                }`}
              >
                {fab}
              </button>
            ))}
          </div>
        </div>

        {/* Processing stage feedback */}
        {loading && stage && (
          <div className="flex items-center gap-3 bg-saree-500/10 border border-saree-500/20 rounded-xl px-5 py-4 mb-6">
            <Loader2 className="w-5 h-5 text-saree-400 animate-spin flex-shrink-0" />
            <p className="text-saree-300 text-sm font-medium">{stage}</p>
          </div>
        )}

        {/* Checklist */}
        <div className="card p-5 mb-8">
          <h3 className="text-white font-semibold mb-3 text-sm">Upload checklist</h3>
          <div className="space-y-2">
            {[
              ['Body image',    !!draping.bodyImagePath],
              ['Saree image',   !!draping.sareeImagePath],
              ['Blouse image',  !!draping.blouseImagePath],
            ].map(([label, done]) => (
              <div key={label} className="flex items-center gap-2 text-sm">
                <CheckCircle2 className={`w-4 h-4 ${done ? 'text-green-400' : 'text-gray-600'}`} />
                <span className={done ? 'text-white' : 'text-gray-500'}>{label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleContinue}
            disabled={loading || !allUploaded}
            className="btn-primary flex items-center gap-2 !px-8 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
          >
            {loading ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Processing...</>
            ) : (
              <>Continue to Customise <ArrowRight className="w-4 h-4" /></>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
