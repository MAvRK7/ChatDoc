
import { useState } from 'react'

function ChatMessage({ message, isLast }) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-fade-in`}>
      <div className={`max-w-[85%] md:max-w-[75%] ${isUser ? 'order-2' : 'order-1'}`}>
        {/* Label */}
        <div className={`flex items-center gap-2 mb-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
          <span className="text-xs font-medium text-chat-muted">
            {isUser ? 'You' : 'ChatDoc'}
          </span>
          {!isUser && message.isStreaming && (
            <span className="text-xs text-chat-accent animate-pulse">●</span>
          )}
        </div>

        {/* Message bubble */}
        <div className={`
          rounded-2xl px-4 py-3 
          ${isUser 
            ? 'bg-chat-accent text-chat-bg rounded-br-md' 
            : 'bg-chat-panel text-chat-text rounded-bl-md border border-chat-panel/50'
          }
          ${!isUser && isLast ? 'glow-accent' : ''}
        `}>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content || (message.isStreaming ? '' : '...')}
          </p>
        </div>

        {/* Copy button for assistant */}
        {!isUser && message.content && !message.isStreaming && (
          <button
            onClick={handleCopy}
            className="mt-2 flex items-center gap-1 text-xs text-chat-muted hover:text-chat-accent transition-colors"
          >
            {copied ? (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Copied
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy
              </>
            )}
          </button>
        )}
      </div>
    </div>
  )
}

export default ChatMessage