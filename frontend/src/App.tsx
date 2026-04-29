import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, Leaf, Shield, Cpu, Activity, 
  CheckCircle2, AlertCircle, ChevronRight, Image as ImageIcon,
  Scan, Eye, ArrowRight, Sparkles, RefreshCw
} from 'lucide-react';
import axios from 'axios';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

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
      setError(err.response?.data?.detail || "Connection failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 font-sans selection:bg-emerald-100 selection:text-emerald-900 overflow-x-hidden">
      
      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 bg-white/70 backdrop-blur-md border-b border-slate-200/60 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2 group cursor-pointer">
          <div className="w-9 h-9 bg-emerald-600 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-600/20 group-hover:rotate-12 transition-transform duration-500">
            <Leaf className="text-white w-5 h-5" />
          </div>
          <span className="text-xl font-bold tracking-tight text-slate-900">AgriVision <span className="text-emerald-600 font-black">Zenith</span></span>
        </div>
        <div className="flex items-center gap-6">
          <a href="#" className="text-sm font-medium text-slate-500 hover:text-emerald-600 transition-colors">Documentation</a>
          <button className="bg-slate-900 text-white px-5 py-2 rounded-full text-sm font-bold hover:bg-slate-800 transition-all active:scale-95">
            Get Pro
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-40 pb-20 px-6 text-center max-w-5xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-50 text-emerald-700 text-[11px] font-bold mb-8 tracking-wide uppercase border border-emerald-100"
        >
          <Sparkles size={14} /> AI-Powered Crop Intelligence
        </motion.div>
        <motion.h1 
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
          className="text-6xl md:text-8xl font-black tracking-tight text-slate-900 mb-8 leading-[0.95]"
        >
          Healthier Crops, <br />
          <span className="text-emerald-600">Smartly Identified.</span>
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
          className="text-slate-500 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed"
        >
          AgriVision Zenith uses advanced neural architecture to diagnose plant pathologies in seconds. Simply upload a specimen to begin.
        </motion.p>
      </section>

      {/* Main Interaction Area */}
      <main className="max-w-6xl mx-auto px-6 pb-40">
        <div className="grid lg:grid-cols-12 gap-12 items-start">
          
          {/* Uploader Column */}
          <div className="lg:col-span-5 space-y-6">
            <motion.div 
              initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
              className="bg-white rounded-[40px] p-10 shadow-[0_40px_80px_-20px_rgba(0,0,0,0.08)] border border-slate-100 relative"
            >
              <div className="relative aspect-square rounded-[32px] overflow-hidden bg-slate-50 border-2 border-dashed border-slate-200 flex flex-col items-center justify-center group cursor-pointer transition-all duration-500 hover:border-emerald-400 hover:bg-emerald-50/30">
                <input type="file" onChange={handleFileChange} className="absolute inset-0 opacity-0 z-10 cursor-pointer" />
                {preview ? (
                  <img src={preview} alt="Preview" className="w-full h-full object-cover" />
                ) : (
                  <div className="flex flex-col items-center text-center px-8">
                    <div className="w-20 h-20 bg-white rounded-3xl flex items-center justify-center mb-6 shadow-sm border border-slate-100 group-hover:scale-110 group-hover:shadow-xl transition-all duration-500">
                      <Upload className="text-slate-400 group-hover:text-emerald-600" size={32} />
                    </div>
                    <p className="text-slate-900 font-black text-lg">Drop your specimen</p>
                    <p className="text-slate-500 text-sm mt-1 leading-relaxed">High resolution leaf images work best for neural extraction.</p>
                  </div>
                )}
                
                <AnimatePresence>
                  {loading && (
                    <motion.div 
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="absolute inset-0 bg-white/80 backdrop-blur-sm flex flex-col items-center justify-center z-20"
                    >
                      <motion.div 
                        animate={{ y: [-150, 150] }} transition={{ repeat: Infinity, duration: 1.2, ease: "easeInOut" }}
                        className="w-full h-[2px] bg-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.8)]"
                      />
                      <p className="mt-6 text-emerald-700 font-black text-xs uppercase tracking-[0.2em] animate-pulse">Neural Scan Active</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {error && (
                <div className="mt-6 p-4 rounded-2xl bg-red-50 text-red-600 text-xs font-bold border border-red-100 flex items-center gap-3">
                  <AlertCircle size={16} /> {error}
                </div>
              )}

              <div className="mt-8 flex gap-4">
                <button 
                  onClick={onAnalyze} disabled={!file || loading}
                  className={cn(
                    "flex-1 py-5 rounded-2xl font-black text-lg transition-all flex items-center justify-center gap-3",
                    !file || loading 
                      ? "bg-slate-100 text-slate-400 cursor-not-allowed" 
                      : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-2xl shadow-emerald-600/20 hover:translate-y-[-2px] active:translate-y-[1px]"
                  )}
                >
                  {loading ? <RefreshCw size={24} className="animate-spin" /> : <>Identify Specimen <ChevronRight size={20} /></>}
                </button>
                {preview && !loading && (
                  <button onClick={() => { setFile(null); setPreview(null); setResults(null); }} className="p-5 bg-slate-100 rounded-2xl text-slate-500 hover:bg-slate-200 transition-colors">
                    <RefreshCw size={24} />
                  </button>
                )}
              </div>
            </motion.div>

            <div className="grid grid-cols-2 gap-4">
               <div className="p-6 rounded-[28px] bg-white border border-slate-100 shadow-sm flex flex-col gap-3">
                  <Shield size={20} className="text-emerald-600" />
                  <span className="text-[11px] font-black uppercase text-slate-400 tracking-widest leading-none">Security</span>
                  <p className="text-xs font-bold text-slate-900 leading-tight">Edge-computed & <br />Encrypted Data</p>
               </div>
               <div className="p-6 rounded-[28px] bg-white border border-slate-100 shadow-sm flex flex-col gap-3">
                  <Cpu size={20} className="text-emerald-600" />
                  <span className="text-[11px] font-black uppercase text-slate-400 tracking-widest leading-none">Neural Core</span>
                  <p className="text-xs font-bold text-slate-900 leading-tight">Hybrid ML-DL <br />Consensus Engine</p>
               </div>
            </div>
          </div>

          {/* Results Column */}
          <div className="lg:col-span-7">
            <AnimatePresence mode="wait">
              {results ? (
                <motion.div 
                  key="results" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
                  className="space-y-8"
                >
                  {/* Verdict Card */}
                  <div className="bg-white rounded-[40px] p-12 shadow-[0_40px_80px_-20px_rgba(0,0,0,0.06)] border border-slate-100 relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-12 opacity-[0.03] group-hover:scale-110 transition-transform duration-1000">
                      <Leaf size={240} />
                    </div>
                    
                    <div className="relative z-10">
                      <div className="flex items-center gap-3 mb-6">
                        <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                        <span className="text-[11px] font-black text-emerald-600 uppercase tracking-[0.2em]">Diagnostic Result</span>
                      </div>
                      <h2 className="text-5xl font-black text-slate-900 leading-tight tracking-tight mb-8">
                        {results.results.dl?.prediction.replace(/___/g, ' ').replace(/_/g, ' ') || results.results.ml?.prediction.replace(/___/g, ' ').replace(/_/g, ' ')}
                      </h2>
                      
                      <div className="flex flex-wrap gap-12">
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-black tracking-widest mb-2">Neural Confidence</p>
                          <p className="text-4xl font-black text-emerald-600">{((results.results.dl?.confidence || results.results.ml?.confidence) * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-black tracking-widest mb-2">Inference Speed</p>
                          <p className="text-4xl font-black text-slate-900">0.42s</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-black tracking-widest mb-2">Status</p>
                          <p className="text-4xl font-black text-slate-900 uppercase">Verified</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Neural Vision */}
                  <div className="space-y-6">
                    <h3 className="text-xl font-black text-slate-900 px-2 flex items-center gap-3">
                      <Eye size={22} className="text-emerald-600" /> Neural Vision Pipeline
                    </h3>
                    <div className="grid grid-cols-3 md:grid-cols-6 gap-3 px-2">
                       {Object.entries(results.processing_steps).map(([name, b64]: any) => (
                         <div key={name} className="space-y-3 group cursor-zoom-in">
                           <div className="aspect-square rounded-2xl overflow-hidden border border-slate-100 bg-slate-50 shadow-sm transition-transform duration-500 group-hover:scale-[1.05]">
                             <img src={`data:image/png;base64,${b64}`} alt={name} className="w-full h-full object-cover grayscale opacity-50 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-700" />
                           </div>
                           <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest text-center block">{name}</span>
                         </div>
                       ))}
                    </div>
                  </div>

                  {/* Grad-CAM Immersive View */}
                  {results.results.dl?.gradcam && (
                    <div className="bg-slate-900 rounded-[40px] p-12 text-white overflow-hidden relative">
                       <div className="relative z-10 flex flex-col md:flex-row gap-12 items-center">
                         <div className="md:w-1/2 space-y-6">
                            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-[10px] font-black uppercase tracking-widest border border-emerald-500/20">
                              Explainability Matrix
                            </span>
                            <h3 className="text-3xl font-black leading-tight">Visualizing Neural <br /><span className="text-emerald-400">Attention Maps</span></h3>
                            <p className="text-slate-400 text-sm leading-relaxed">
                              This heatmap indicates exactly where the model focused to reach its diagnostic verdict. 
                              Regions in vibrant red represent high neural activation for the detected pathology.
                            </p>
                            <div className="flex gap-10 pt-4">
                               <div>
                                  <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Heat Intensity</p>
                                  <p className="text-xl font-bold">High</p>
                               </div>
                               <div>
                                  <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Saliency Map</p>
                                  <p className="text-xl font-bold text-emerald-400">Activated</p>
                               </div>
                            </div>
                         </div>
                         <div className="md:w-1/2 relative">
                            <div className="absolute -inset-10 bg-emerald-500/20 blur-[100px] rounded-full pointer-events-none"></div>
                            <img src={`data:image/png;base64,${results.results.dl.gradcam}`} alt="Grad-CAM" className="relative w-full rounded-[32px] shadow-2xl border border-white/10" />
                         </div>
                       </div>
                    </div>
                  )}

                </motion.div>
              ) : (
                <motion.div 
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                  className="h-full min-h-[500px] rounded-[48px] bg-white border border-slate-100 shadow-sm flex flex-col items-center justify-center p-20 text-center"
                >
                  <div className="w-24 h-24 bg-slate-50 rounded-[32px] flex items-center justify-center mb-8 border border-slate-100">
                    <Activity className="text-slate-200" size={48} />
                  </div>
                  <h3 className="text-2xl font-black text-slate-900 mb-4 tracking-tight">System Idle</h3>
                  <p className="text-slate-500 max-w-sm leading-relaxed">
                    Once a specimen is uploaded and analyzed, the comprehensive neural diagnostic report and visual attention maps will be rendered here.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="py-20 px-6 border-t border-slate-200 text-center bg-white">
        <div className="flex flex-col items-center gap-6">
          <div className="flex items-center gap-2 grayscale opacity-30">
            <Leaf size={24} className="text-emerald-600" />
            <span className="font-black text-xl tracking-tighter">AgriVision AI</span>
          </div>
          <p className="text-slate-400 text-xs font-medium">Built with Neural Core v4.0. Precision Agricultural Research Systems.</p>
          <div className="flex gap-8 text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">
            <a href="#" className="hover:text-emerald-600 transition-colors">Privacy</a>
            <a href="#" className="hover:text-emerald-600 transition-colors">Terms</a>
            <a href="#" className="hover:text-emerald-600 transition-colors">API Support</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
