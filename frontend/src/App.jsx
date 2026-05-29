import { useState, useRef, useEffect } from 'react'
import Header from './components/Header.jsx'
import ChatMessage from './components/ChatMessage.jsx'
import ChatInput from './components/ChatInput.jsx'
import SuggestionCards from './components/SuggestionCards.jsx'
import ThinkingIndicator from './components/ThinkingIndicator.jsx'
import useChatStream from './hooks/useChatStream.js'

function App() {
  const [messages, setMessages] = useState([])
  const [isThinking, setIsThinking] = useState(false)
  const [selectedModel, setSelectedModel] = useState('chat-doctor-q4') // Default to accurate
  const [models, setModels] = useState([])
  const messagesEndRef = useRef(null)

  // Fetch available models on mount
  useEffect(() => {
    fetch('https://SatRag-chat-doctor-api.hf.space/v1/models', {
      headers: { 'Authorization': 'Bearer test-key-123' }
    })
      .then(r => r.json())
      .then(data => setModels(data.data || []))
      .catch(() => setModels([
        { id: 'chat-doctor-q8', name: 'ChatDoc Accurate', description: 'Higher quality, slower', quantization: 'Q8_0' },
        { id: 'chat-doctor-q4', name: 'ChatDoc Fast', description: 'Faster responses', quantization: 'Q4_K_M' },
      ]))
  }, [])

  const { sendMessage, isStreaming } = useChatStream({
    selectedModel,
    onStart: () => setIsThinking(true),
    onFirstToken: () => setIsThinking(false),
    onMessage: (content) => {
      setMessages(prev => {
        const last = prev[prev.length - 1]
        if (last && last.role === 'assistant' && last.isStreaming) {
          return [...prev.slice(0, -1), { ...last, content: last.content + content }]
        }
        return prev
      })
    },
    onComplete: () => {
      setMessages(prev => {
        const last = prev[prev.length - 1]
        if (last && last.isStreaming) {
          return [...prev.slice(0, -1), { ...last, isStreaming: false }]
        }
        return prev
      })
    }
  })

  const handleSend = async (content) => {
    const userMessage = { role: 'user', content, id: Date.now() }
    const assistantMessage = { role: 'assistant', content: '', isStreaming: true, id: Date.now() + 1 }
    
    setMessages(prev => [...prev, userMessage, assistantMessage])
    
    const conversationHistory = [...messages, userMessage].map(m => ({
      role: m.role,
      content: m.content
    }))
    
    await sendMessage(conversationHistory)
  }

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isThinking])

  const hasMessages = messages.length > 0

  return (
    <div className="h-screen flex flex-col bg-chat-bg text-chat-text">
      <Header 
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        models={models}
      />
      
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto">
          {!hasMessages && (
            <div className="mt-20">
              <SuggestionCards onSelect={handleSend} />
            </div>
          )}
          
          {messages.map((msg, index) => (
            <ChatMessage 
              key={msg.id || index} 
              message={msg} 
              isLast={index === messages.length - 1}
            />
          ))}
          
          {isThinking && <ThinkingIndicator modelName={models.find(m => m.id === selectedModel)?.name} />}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="border-t border-chat-panel bg-chat-bg px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <ChatInput onSend={handleSend} disabled={isStreaming} />
          
          <p className="text-center text-xs text-chat-muted mt-3 leading-relaxed">
            ChatDoc is an AI assistant for informational purposes only and is not a substitute for professional medical advice.
          </p>
        </div>
      </div>
    </div>
  )
}

export default App