import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Leaf, Shield, Cpu, Activity, Info, CheckCircle2, AlertCircle, ChevronRight, Image as ImageIcon } from 'lucide-react';
import axios from 'axios';

// --- Components ---

const Navbar = () => (
  <nav className="fixed top-0 w-full z-50 glass px-6 py-4 flex justify-between items-center border-b border-white/5">
    <div className="flex items-center gap-2">
      <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-500/20">
        <Leaf className="text-white w-6 h-6" />
      </div>
      <span className="text-2xl font-black tracking-tighter text-white">Agri<span className="text-emerald-400">Vision</span> <span className="text-[10px] uppercase bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-md align-top ml-1">Pro</span></span>
    </div>
    <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-400">
      <a href="#" className="hover:text-emerald-400 transition-colors">Technology</a>
      <a href="#" className="hover:text-emerald-400 transition-colors">Methodology</a>
      <a href="#" className="hover:text-emerald-400 transition-colors">API</a>
      <button className="bg-emerald-600 hover:bg-emerald-500 text-white px-5 py-2 rounded-full transition-all hover:scale-105 active:scale-95 shadow-lg shadow-emerald-600/20">
        Get Started
      </button>
    </div>
  </nav>
);

const Hero = () => (
  <div className="relative pt-32 pb-20 px-6 overflow-hidden">
    <div className="max-w-7xl mx-auto text-center relative z-10">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-bold mb-6"
      >
        <Shield size={14} /> NEXT-GEN PLANT DIAGNOSTICS
      </motion.div>
      <motion.h1 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
        className="text-6xl md:text-8xl font-black tracking-tight mb-8 leading-[0.9]"
      >
        Detect Disease <br /> 
        <span className="text-gradient">Faster Than Ever.</span>
      </motion.h1>
      <motion.p 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto mb-12"
      >
        Harness the power of Classical Machine Learning and Deep Neural Networks to protect your crops with precision AI.
      </motion.p>
    </div>
    {/* Decorative background elements */}
    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[600px] bg-emerald-500/10 blur-[120px] rounded-full -z-10 pointer-events-none"></div>
  </div>
);

// --- Main App ---

const App = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResults(null);
      setError(null);
    }
  };

  const onAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/analyze', formData);
      setResults(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Connection failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-[#0f172a] text-slate-200 selection:bg-emerald-500/30">
      <Navbar />
      
      <Hero />

      <main className="max-w-7xl mx-auto px-6 pb-32">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          
          {/* Left Column: Upload & Input */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-8"
          >
            <div className="glass p-8 rounded-3xl border-white/5 space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-bold flex items-center gap-2">
                  <Upload className="text-emerald-500" size={20} /> Upload Specimen
                </h3>
                <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Input Layer</span>
              </div>
              
              <div className="relative group">
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                />
                <div className={`
                  border-2 border-dashed rounded-2xl p-12 flex flex-col items-center justify-center transition-all duration-300
                  ${preview ? 'border-emerald-500/50 bg-emerald-500/5' : 'border-slate-700 group-hover:border-emerald-500/50 group-hover:bg-emerald-500/5'}
                `}>
                  {preview ? (
                    <img src={preview} alt="Preview" className="max-h-64 rounded-xl shadow-2xl" />
                  ) : (
                    <>
                      <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                        <ImageIcon className="text-slate-500 group-hover:text-emerald-400" size={32} />
                      </div>
                      <p className="text-slate-400 font-medium">Drop image here or click to browse</p>
                      <p className="text-slate-600 text-xs mt-2">Supports JPG, PNG (Max 10MB)</p>
                    </>
                  )}
                </div>
              </div>

              {error && (
                <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex items-center gap-3">
                  <AlertCircle size={18} /> {error}
                </div>
              )}

              <button 
                onClick={onAnalyze}
                disabled={!file || loading}
                className={`
                  w-full py-4 rounded-2xl font-black text-lg transition-all flex items-center justify-center gap-3
                  ${!file || loading 
                    ? 'bg-slate-800 text-slate-600 cursor-not-allowed' 
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-xl shadow-emerald-600/20 hover:scale-[1.02] active:scale-[0.98]'}
                `}
              >
                {loading ? (
                  <motion.div 
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                  >
                    <Activity size={24} />
                  </motion.div>
                ) : (
                  <>Start Analysis <ChevronRight size={20} /></>
                )}
              </button>
            </div>

            {/* Feature Highlights */}
            <div className="grid grid-cols-2 gap-4">
              <div className="glass p-6 rounded-2xl border-white/5">
                <Shield className="text-emerald-500 mb-3" />
                <h4 className="font-bold text-sm">Secure Data</h4>
                <p className="text-slate-500 text-xs mt-1">Processed locally on encrypted endpoints.</p>
              </div>
              <div className="glass p-6 rounded-2xl border-white/5">
                <Cpu className="text-emerald-500 mb-3" />
                <h4 className="font-bold text-sm">Hybrid AI</h4>
                <p className="text-slate-500 text-xs mt-1">MobileNetV2 + Classical ML fusion.</p>
              </div>
            </div>
          </motion.div>

          {/* Right Column: Results */}
          <div className="space-y-8">
            <AnimatePresence mode="wait">
              {results ? (
                <motion.div 
                  key="results"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="space-y-8"
                >
                  {/* Result Card */}
                  <div className="glass p-8 rounded-3xl border-emerald-500/20 bg-emerald-500/5 relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-6 opacity-20">
                      <CheckCircle2 size={120} />
                    </div>
                    
                    <div className="relative z-10">
                      <span className="text-[10px] text-emerald-400 uppercase tracking-widest font-black mb-2 block">Primary Diagnosis</span>
                      <h2 className="text-4xl font-black mb-4">
                        {results.results.dl?.prediction.replace(/___/g, ' ').replace(/_/g, ' ') || results.results.ml?.prediction.replace(/___/g, ' ').replace(/_/g, ' ')}
                      </h2>
                      
                      <div className="flex items-center gap-6">
                        <div>
                          <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">Confidence</div>
                          <div className="text-2xl font-black text-emerald-400">
                            {((results.results.dl?.confidence || results.results.ml?.confidence) * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="w-px h-10 bg-white/10"></div>
                        <div>
                          <div className="text-[10px] text-slate-500 uppercase font-bold mb-1">Status</div>
                          <div className="text-2xl font-black">Success</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Processing Steps Visualization */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-bold">Neural Pipeline Visualization</h3>
                    <div className="grid grid-cols-3 gap-3">
                      {Object.entries(results.processing_steps).map(([name, b64]: any) => (
                        <div key={name} className="glass p-2 rounded-xl border-white/5 group">
                          <img 
                            src={`data:image/png;base64,${b64}`} 
                            alt={name} 
                            className="w-full h-auto rounded-lg mb-2 grayscale group-hover:grayscale-0 transition-all cursor-zoom-in" 
                          />
                          <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider text-center block">{name}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Grad-CAM Highlight */}
                  {results.results.dl?.gradcam && (
                    <div className="glass p-8 rounded-3xl border-white/5 space-y-4">
                      <h3 className="text-lg font-bold flex items-center gap-2">
                        <Activity className="text-emerald-500" size={20} /> Explainable AI (Grad-CAM)
                      </h3>
                      <p className="text-slate-400 text-sm">
                        Neural heatmaps highlight the specific areas the model prioritized for this diagnosis.
                      </p>
                      <img 
                        src={`data:image/png;base64,${results.results.dl.gradcam}`} 
                        alt="Grad-CAM" 
                        className="w-full rounded-2xl shadow-xl shadow-emerald-500/10" 
                      />
                    </div>
                  )}

                </motion.div>
              ) : (
                <motion.div 
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full min-h-[500px] glass rounded-3xl border-white/5 flex flex-col items-center justify-center p-12 text-center"
                >
                  <div className="w-20 h-20 bg-slate-800 rounded-3xl flex items-center justify-center mb-6 animate-pulse">
                    <Info className="text-slate-600" size={40} />
                  </div>
                  <h3 className="text-xl font-bold mb-2">Awaiting Specimen</h3>
                  <p className="text-slate-500 text-sm max-w-xs">
                    Upload an image of a plant leaf to begin the deep-layer feature extraction and disease diagnosis.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-12 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-8">
          <div className="flex items-center gap-2 grayscale opacity-50">
            <Leaf size={20} />
            <span className="font-black tracking-tight">AgriVision Pro</span>
          </div>
          <p className="text-slate-600 text-sm">© 2026 AgriVision AI Systems. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
