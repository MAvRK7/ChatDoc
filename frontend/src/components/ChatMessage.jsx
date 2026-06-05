import { useState, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'

function ChatMessage({ message, isLast }) {
  const isUser = message.role === 'user'
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Convert model output to proper markdown
  const toMarkdown = (text) => {
    if (!text) return ''
    
    let formatted = text
      .replace(/([a-z])If\b/g, '$1 If')
      .replace(/([a-z])And\b/g, '$1 and')
      .replace(/([a-z])Or\b/g, '$1 or')
      .replace(/(\d+\.)/g, '\n$1')
      .replace(/(\d+)\.([A-Za-z])/g, '$1. $2')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/^\n+/, '')
      .replace(/\.([A-Za-z])/g, '. $1')
    
    return formatted
  }

  // ONLY recompute when message.content changes
  const processedContent = useMemo(() => {
    return !isUser ? toMarkdown(message.content) : message.content
  }, [message.content, isUser])

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-fade-in`}>
      <div className={`max-w-[85%] md:max-w-[75%] ${isUser ? 'order-2' : 'order-1'}`}>
        <div className={`flex items-center gap-2 mb-1 ${isUser ? 'justify-end' : 'justify-start'}`}>
          <span className="text-xs font-medium text-chat-muted">
            {isUser ? 'You' : 'ChatDoc'}
          </span>
          {!isUser && message.isStreaming && (
            <span className="text-xs text-chat-accent animate-pulse">●</span>
          )}
        </div>

        <div className={`
          rounded-2xl px-4 py-3 
          ${isUser 
            ? 'bg-chat-accent text-chat-bg rounded-br-md' 
            : 'bg-chat-panel text-chat-text rounded-bl-md border border-chat-panel/50'
          }
          ${!isUser && isLast ? 'glow-accent' : ''}
        `}>
          {isUser ? (
            <div className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content || (message.isStreaming ? '' : '...')}
            </div>
          ) : (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown>
                {processedContent}
              </ReactMarkdown>
            </div>
          )}
        </div>

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