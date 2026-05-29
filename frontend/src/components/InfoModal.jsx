import { useState, useEffect } from 'react'

function InfoModal({ onClose, selectedModel, models }) {
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  const currentModel = models.find(m => m.id === selectedModel) || {}

  useEffect(() => {
    fetch('https://SatRag-chat-doctor-api.hf.space/v1/models', {
      headers: { 'Authorization': 'Bearer test-key-123' }
    })
      .then(r => r.json())
      .then(data => {
        const match = data.data?.find(m => m.id === selectedModel)
        setModelInfo(match || null)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [selectedModel])

  if (loading) return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-chat-panel border border-chat-accent/20 rounded-2xl p-8 max-w-md w-full mx-4">
        <div className="text-chat-accent animate-pulse">Loading...</div>
      </div>
    </div>
  )

  const info = modelInfo || currentModel

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-chat-panel/95 border border-chat-accent/20 rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl shadow-chat-accent/10" onClick={e => e.stopPropagation()}>
        
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-chat-text">About ChatDoc</h2>
          <button onClick={onClose} className="text-chat-muted hover:text-chat-text transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-4">
          {/* Active Model Badge */}
          <div className="p-3 rounded-xl bg-chat-accent/10 border border-chat-accent/20">
            <p className="text-xs text-chat-accent mb-1">Active Model</p>
            <p className="text-sm text-chat-text font-medium">{info.name || selectedModel}</p>
            <p className="text-xs text-chat-muted mt-0.5">{info.description}</p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-xl bg-chat-bg/50 border border-chat-panel">
              <p className="text-xs text-chat-muted mb-1">Base Model</p>
              <p className="text-sm text-chat-text font-medium">google/gemma-4-E2B-it</p>
            </div>
            <div className="p-3 rounded-xl bg-chat-bg/50 border border-chat-panel">
              <p className="text-xs text-chat-muted mb-1">Parameters</p>
              <p className="text-sm text-chat-text font-medium">4B</p>
            </div>
            <div className="p-3 rounded-xl bg-chat-bg/50 border border-chat-panel">
              <p className="text-xs text-chat-muted mb-1">Quantization</p>
              <p className="text-sm text-chat-text font-medium">{info.quantization || 'Q8_0'}</p>
            </div>
            <div className="p-3 rounded-xl bg-chat-bg/50 border border-chat-panel">
              <p className="text-xs text-chat-muted mb-1">Context</p>
              <p className="text-sm text-chat-text font-medium">2048 tokens</p>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-chat-bg/50 border border-chat-panel">
            <p className="text-xs text-chat-muted mb-1">Architecture</p>
            <p className="text-sm text-chat-text font-medium">Gemma 4</p>
          </div>

          <p className="text-xs text-chat-muted text-center pt-2">
            {info.quantization === 'Q4_K_M' 
              ? 'Fast mode: Optimized for speed with slight quality trade-off'
              : 'Accurate mode: Best quality, slower inference'
            }
          </p>
        </div>
      </div>
    </div>
  )
}

export default InfoModal