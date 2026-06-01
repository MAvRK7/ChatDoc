import { useState, useCallback } from 'react'

const API_URL = 'https://SatRag-chat-doctor-api.hf.space/v1/chat/completions'
const API_KEY = 'test-key-123'

export default function useChatStream({ selectedModel, onStart, onFirstToken, onMessage, onComplete }) {
  const [isStreaming, setIsStreaming] = useState(false)

  // Add this formatting function
  const formatResponse = (text) => {
    // Fix numbered lists: "1.text" -> "1. text"
    let formatted = text.replace(/(\d+)\.([A-Za-z])/g, '$1. $2')
    
    // Add newlines before numbers if missing
    formatted = formatted.replace(/([^.])(\d+\.)/g, '$1\n$2')
    
    // Fix spacing after periods
    formatted = formatted.replace(/\.([A-Z])/g, '. $1')
    
    // Clean up multiple spaces
    formatted = formatted.replace(/\s+/g, ' ')
    
    // Clean up multiple newlines
    formatted = formatted.replace(/\n{3,}/g, '\n\n')
    
    return formatted
  }

  const sendMessage = useCallback(async (messages) => {
    setIsStreaming(true)
    onStart?.()

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
          model: selectedModel,  // "chat-doctor-q4" or "chat-doctor-q8"
          messages: messages,
          stream: true,
          max_tokens: 160,
          temperature: 0.3
        })
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || `HTTP ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let firstTokenReceived = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || !trimmed.startsWith('data: ')) continue
          
          const data = trimmed.slice(6)
          if (data === '[DONE]') continue

          try {
            const parsed = JSON.parse(data)
            const content = parsed.choices?.[0]?.delta?.content
            
            if (content) {
              if (!firstTokenReceived) {
                firstTokenReceived = true
                onFirstToken?.()
              }
              // Apply formatting to the content
              const formattedContent = formatResponse(content)
              onMessage?.(formattedContent)
            }
          } catch (e) {
            // Skip malformed
          }
        }
      }

      onComplete?.()
    } catch (error) {
      console.error('Stream error:', error)
      onMessage?.(`\n\n[Error: ${error.message}]`)
      onComplete?.()
    } finally {
      setIsStreaming(false)
    }
  }, [selectedModel, onStart, onFirstToken, onMessage, onComplete])

  return { sendMessage, isStreaming }
}