import { useState, useRef } from 'react'

function ChatInput({ onSend, disabled }) {
  const [input, setInput] = useState('')
  const textareaRef = useRef(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || disabled) return
    
    onSend(input.trim())
    setInput('')
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-end gap-2 bg-chat-panel rounded-2xl border border-chat-panel/50 p-2 focus-within:border-chat-accent/30 transition-colors">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your symptoms or ask a medical question..."
          rows={1}
          disabled={disabled}
          className="flex-1 bg-transparent text-chat-text placeholder-chat-muted text-sm px-3 py-2.5 resize-none outline-none max-h-32"
          style={{ minHeight: '44px' }}
        />
        
        <button
          type="submit"
          disabled={!input.trim() || disabled}
          className={`
            p-2.5 rounded-xl transition-all duration-200
            ${input.trim() && !disabled
              ? 'bg-chat-accent text-chat-bg hover:bg-chat-accent/90 shadow-lg shadow-chat-accent/20' 
              : 'bg-chat-panel text-chat-muted cursor-not-allowed'
            }
          `}
        >
          {disabled ? (
            <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
            </svg>
          )}
        </button>
      </div>
    </form>
  )
}

export default ChatInput