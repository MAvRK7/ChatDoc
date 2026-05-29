function ThinkingIndicator({ modelName }) {
  return (
    <div className="flex items-center gap-3 mb-4 animate-fade-in">
      <div className="w-8 h-8 rounded-lg bg-chat-accent/20 flex items-center justify-center flex-shrink-0">
        <svg className="w-5 h-5 text-chat-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      </div>
      <div className="bg-chat-panel rounded-2xl rounded-bl-md px-4 py-3 border border-chat-panel/50">
        <div className="flex items-center gap-1.5">
          <span className="text-sm text-chat-muted">
            {modelName ? `${modelName} is thinking` : 'ChatDoc is thinking'}
          </span>
          <div className="flex gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-chat-accent animate-pulse-dot" style={{ animationDelay: '0ms' }} />
            <span className="w-1.5 h-1.5 rounded-full bg-chat-accent animate-pulse-dot" style={{ animationDelay: '200ms' }} />
            <span className="w-1.5 h-1.5 rounded-full bg-chat-accent animate-pulse-dot" style={{ animationDelay: '400ms' }} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default ThinkingIndicator