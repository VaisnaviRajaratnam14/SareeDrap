import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 120000, // 2 min for AI processing
});

// Attach JWT token to every request
api.interceptors.request.use(config => {
  const token = localStorage.getItem('saree_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Global error handler
api.interceptors.response.use(
  res => res,
  err => {
    // Only clear token + redirect on a real 401, not on 503 DB-unavailable
    if (err.response?.status === 401 && err.response?.status !== 503) {
      localStorage.removeItem('saree_token');
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

// ── Health ───────────────────────────────────────────────────────────────────
export const getHealth = () => api.get('/health');

// ── Upload helpers ────────────────────────────────────────────────────────────
export const uploadBody   = file => _upload('/upload/body',   file);
export const uploadSaree  = file => _upload('/upload/saree',  file);
export const uploadBlouse = file => _upload('/upload/blouse', file);

function _upload(endpoint, file) {
  const form = new FormData();
  form.append('image', file);
  return api.post(endpoint, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
}

// ── AI processing ─────────────────────────────────────────────────────────────
export const detectPose = body_image_path =>
  api.post('/pose/detect', { body_image_path });

export const processSaree = (saree_image_path, fabric_type) =>
  api.post('/saree/process', { saree_image_path, fabric_type });

export const renderDraping = payload => api.post('/render/generate', payload);

export const downloadOutput = output_id =>
  api.get(`/render/download/${output_id}`, { responseType: 'blob' });

export const cleanupOutput = output_id =>
  api.delete(`/render/cleanup/${output_id}`);

// ── Templates ─────────────────────────────────────────────────────────────────
export const getSareeTemplates  = () => api.get('/saree/templates');
export const getBlouseTemplates = () => api.get('/blouse/templates');

// ── Engine status ─────────────────────────────────────────────────────────────
export const getEngineStatus  = ()  => api.get('/render/engine-status');
export const testInference    = ()  => api.post('/render/test-inference');

// ── Dataset ───────────────────────────────────────────────────────────────────
export const fetchDataset     = (dataset_slug, category, max_images = 150, mode = 'append') =>
  api.post('/dataset/kaggle-download', { dataset_slug, category, max_images, mode });
export const cleanDataset     = ()  => api.post('/dataset/clean');
export const annotateDataset  = ()  => api.post('/dataset/annotate');
export const getDatasetStatus = ()  => api.get('/dataset/status');
export const getLiveCounts    = ()  => api.get('/dataset/counts');

// ── Training ──────────────────────────────────────────────────────────────────
export const startTraining    = (config = {}) => api.post('/train/start', config);
export const getTrainingStatus = ()            => api.get('/train/status');

// ── History ───────────────────────────────────────────────────────────────────
export const getHistory = () => api.get('/render/history');

// ── Admin ─────────────────────────────────────────────────────────────────────
export const getAdminStats   = () => api.get('/admin/stats');
export const getAdminUsers   = () => api.get('/admin/users');
export const getAdminOutputs = () => api.get('/admin/outputs');
export const deleteUser      = id => api.delete(`/admin/users/${id}`);
export const deleteOutput    = id => api.delete(`/admin/outputs/${id}`);

export default api;
