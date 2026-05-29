import { useState } from 'react'
import InfoModal from './InfoModal.jsx'

function Header({ selectedModel, onModelChange, models }) {
  const [showInfo, setShowInfo] = useState(false)
  const [showDropdown, setShowDropdown] = useState(false)

  const currentModel = models.find(m => m.id === selectedModel) || {}

  return (
    <>
      <header className="border-b border-chat-panel bg-chat-bg px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-chat-accent/15 flex items-center justify-center border border-chat-accent/20">
              <svg className="w-5 h-5 text-chat-accent" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 6v12M6 12h12" />
                <circle cx="12" cy="12" r="10" strokeOpacity="0.3" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-semibold text-chat-text tracking-tight">ChatDoc</h1>
              <p className="text-xs text-chat-muted">AI Medical Assistant</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Model Selector */}
            <div className="relative">
              <button 
                onClick={() => setShowDropdown(!showDropdown)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-chat-panel/80 border border-chat-accent/20 hover:border-chat-accent/40 transition-all text-xs"
              >
                <span className={`w-2 h-2 rounded-full ${selectedModel === 'chat-doctor-q4' ? 'bg-green-400' : 'bg-chat-accent'}`} />
                <span className="text-chat-text">{currentModel.name || selectedModel}</span>
                <svg className="w-3 h-3 text-chat-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {showDropdown && (
                <div className="absolute right-0 top-full mt-1 w-56 rounded-xl bg-chat-panel border border-chat-accent/20 shadow-xl shadow-black/20 overflow-hidden z-50">
                  {models.map(model => (
                    <button
                      key={model.id}
                      onClick={() => {
                        onModelChange(model.id)
                        setShowDropdown(false)
                      }}
                      className={`w-full text-left px-4 py-3 hover:bg-chat-accent/10 transition-colors flex items-start gap-3 ${selectedModel === model.id ? 'bg-chat-accent/5' : ''}`}
                    >
                      <span className={`w-2 h-2 rounded-full mt-1 flex-shrink-0 ${model.id === 'chat-doctor-q4' ? 'bg-green-400' : 'bg-chat-accent'}`} />
                      <div>
                        <div className="text-sm text-chat-text font-medium">{model.name}</div>
                        <div className="text-xs text-chat-muted">{model.description}</div>
                        <div className="text-xs text-chat-accent mt-0.5">{model.quantization}</div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Info Button */}
            <button 
              onClick={() => setShowInfo(true)}
              className="w-8 h-8 rounded-lg bg-chat-panel/80 backdrop-blur-sm border border-chat-accent/20 flex items-center justify-center hover:bg-chat-accent/10 hover:border-chat-accent/40 transition-all group"
            >
              <svg className="w-4 h-4 text-chat-muted group-hover:text-chat-accent transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      {showInfo && <InfoModal onClose={() => setShowInfo(false)} selectedModel={selectedModel} models={models} />}
    </>
  )
}

export default Header