import { useState, useCallback, useRef } from 'react'

const API_URL = 'https://SatRag-chat-doctor-api.hf.space/v1/chat/completions'
const API_KEY = 'test-key-123'

export default function useChatStream({ selectedModel, onStart, onFirstToken, onMessage, onComplete }) {
  const [isStreaming, setIsStreaming] = useState(false)
  const fullMessageRef = useRef('')
  const lastFormattedLengthRef = useRef(0)

  const formatResponse = (text) => {
    let formatted = text.replace(/(\d+)\.([A-Za-z])/g, '$1. $2')
    formatted = formatted.replace(/([^\n])(\d+\.)/g, '$1\n$2')
    formatted = formatted.replace(/\.([A-Za-z0-9])/g, '. $1')
    formatted = formatted.replace(/\s{2,}/g, ' ')
    formatted = formatted.replace(/:([A-Za-z])/g, ': $1')
    return formatted
  }

  const sendMessage = useCallback(async (messages) => {
    setIsStreaming(true)
    onStart?.()
    fullMessageRef.current = ''
    lastFormattedLengthRef.current = 0

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
          model: selectedModel,
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
      let formatTimeout = null

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
              
              fullMessageRef.current += content
              
              // Only format and update every few chunks or on natural breaks (periods, newlines)
              const shouldFormat = 
                fullMessageRef.current.length - lastFormattedLengthRef.current > 50 || // Every 50 chars
                /[.!?]\s*$/.test(fullMessageRef.current) || // After sentence endings
                /\d+\./.test(content) // When we see list numbers
              
              if (shouldFormat) {
                const formattedMessage = formatResponse(fullMessageRef.current)
                onMessage?.(formattedMessage)
                lastFormattedLengthRef.current = fullMessageRef.current.length
              }
            }
          } catch (e) {
            // Skip malformed
          }
        }
      }
      
      // Final format at the end
      const finalFormatted = formatResponse(fullMessageRef.current)
      onMessage?.(finalFormatted)

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