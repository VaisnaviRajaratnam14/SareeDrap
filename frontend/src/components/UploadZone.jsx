import React, { useRef } from 'react';
import { Upload, CheckCircle2, X } from 'lucide-react';

/**
 * Reusable drag-and-drop upload zone.
 * Props: label, preview (URL), onFile(File), onRemove(), loading
 */
export default function UploadZone({ label, preview, onFile, onRemove, loading, icon: Icon }) {
  const inputRef = useRef(null);

  const handleDrop = e => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  const handleChange = e => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
    e.target.value = '';
  };

  if (preview) {
    return (
      <div className="relative rounded-2xl overflow-hidden border border-saree-500/30 shadow-xl">
        <img src={preview} alt="preview" className="w-full h-56 object-cover" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end p-4">
          <div className="flex items-center gap-2 text-sm text-white">
            <CheckCircle2 className="w-4 h-4 text-green-400" />
            <span>{label} uploaded</span>
          </div>
        </div>
        {!loading && (
          <button
            onClick={onRemove}
            className="absolute top-3 right-3 w-7 h-7 rounded-full bg-red-600/80 hover:bg-red-600 flex items-center justify-center transition"
          >
            <X className="w-4 h-4 text-white" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
      onClick={() => !loading && inputRef.current?.click()}
      className="upload-zone min-h-[14rem]"
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleChange}
        disabled={loading}
      />
      <div className="w-14 h-14 rounded-2xl bg-saree-500/10 border border-saree-500/20 flex items-center justify-center mb-4">
        {Icon ? <Icon className="w-7 h-7 text-saree-400" /> : <Upload className="w-7 h-7 text-saree-400" />}
      </div>
      <p className="text-white font-semibold mb-1">{label}</p>
      <p className="text-gray-400 text-sm">
        Drag & drop or <span className="text-saree-400 underline">browse</span>
      </p>
      <p className="text-gray-500 text-xs mt-2">PNG, JPG, JPEG up to 16 MB</p>
    </div>
  );
}
