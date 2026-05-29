import React, { createContext, useContext, useState } from 'react';

/**
 * DrapingContext holds all state across the multi-step draping wizard:
 * Upload → Customize → Preview
 */
const DrapingContext = createContext(null);

const DEFAULT_BLOUSE_CONFIG = {
  neck_type:   'round',
  sleeve_type: 'short',
  color:       '#8b0057',
};

export function DrapingProvider({ children }) {
  // Uploaded file paths (returned by backend after upload)
  const [bodyImagePath,   setBodyImagePath]   = useState(null);
  const [sareeImagePath,  setSareeImagePath]   = useState(null);
  const [blouseImagePath, setBlouseImagePath]  = useState(null);

  // Local preview URLs (for displaying in UI before API call)
  const [bodyPreview,   setBodyPreview]   = useState(null);
  const [sareePreview,  setSareePreview]  = useState(null);
  const [blousePreview, setBlousePreview] = useState(null);

  // AI processing results
  const [poseData,      setPoseData]      = useState(null);
  const [sareeFeatures, setSareeFeatures] = useState(null);

  // Customization options
  const [drapingStyle,  setDrapingStyle]  = useState('nivi');
  const [fabricType,    setFabricType]    = useState('silk');
  const [blouseConfig,  setBlouseConfig]  = useState(DEFAULT_BLOUSE_CONFIG);

  // Final output
  const [outputId,      setOutputId]      = useState(null);
  const [outputPath,    setOutputPath]    = useState(null);

  // Engine metadata (returned by /api/render/generate)
  const [engineUsed,       setEngineUsed]       = useState(null);
  const [mlModelAvailable, setMlModelAvailable] = useState(null);
  const [fallbackReason,   setFallbackReason]   = useState('');

  // Step tracking
  const [currentStep,   setCurrentStep]   = useState(1);

  const updateBlouseConfig = (key, value) =>
    setBlouseConfig(prev => ({ ...prev, [key]: value }));

  const resetDraping = () => {
    setBodyImagePath(null);   setSareeImagePath(null);  setBlouseImagePath(null);
    setBodyPreview(null);     setSareePreview(null);    setBlousePreview(null);
    setPoseData(null);        setSareeFeatures(null);
    setDrapingStyle('nivi');  setFabricType('silk');
    setBlouseConfig(DEFAULT_BLOUSE_CONFIG);
    setOutputId(null);        setOutputPath(null);
    setEngineUsed(null);      setMlModelAvailable(null); setFallbackReason('');
    setCurrentStep(1);
  };

  return (
    <DrapingContext.Provider value={{
      // Files
      bodyImagePath,   setBodyImagePath,
      sareeImagePath,  setSareeImagePath,
      blouseImagePath, setBlouseImagePath,
      // Previews
      bodyPreview,   setBodyPreview,
      sareePreview,  setSareePreview,
      blousePreview, setBlousePreview,
      // AI data
      poseData,      setPoseData,
      sareeFeatures, setSareeFeatures,
      // Config
      drapingStyle,  setDrapingStyle,
      fabricType,    setFabricType,
      blouseConfig,  updateBlouseConfig,
      // Output
      outputId,      setOutputId,
      outputPath,    setOutputPath,
      // Engine metadata
      engineUsed,       setEngineUsed,
      mlModelAvailable, setMlModelAvailable,
      fallbackReason,   setFallbackReason,
      // Navigation
      currentStep,   setCurrentStep,
      resetDraping,
    }}>
      {children}
    </DrapingContext.Provider>
  );
}

export function useDraping() {
  return useContext(DrapingContext);
}
