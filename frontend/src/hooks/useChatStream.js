import { useState, useCallback } from 'react'

const API_URL = 'https://SatRag-chat-doctor-api.hf.space/v1/chat/completions'
const API_KEY = 'test-key-123'

export default function useChatStream({ selectedModel, onStart, onFirstToken, onMessage, onComplete }) {
  const [isStreaming, setIsStreaming] = useState(false)

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
              onMessage?.(content)
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