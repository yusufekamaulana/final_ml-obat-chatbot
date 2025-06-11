"use client"

import { useEffect, useState } from "react"
import { ChatMessages } from "@/components/ui/chat"
import { MessageInput } from "@/components/ui/message-input"
import { MessageList } from "@/components/ui/message-list"
import { Card, CardContent } from "@/components/ui/card"
import { type Message } from "@/components/ui/chat-message"

export function ChatbotLayout() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  function getCookie(name: string) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop()?.split(";").shift();
    return null;
  }

  const token =
  typeof window !== "undefined" ? getCookie("token") : null;

  useEffect(() => {
    fetchChatHistory()
  }, [])

  const fetchChatHistory = async () => {
    if (!token) return
    try {
      const res = await fetch("http://localhost:8000/chat/history", {
        // headers: {
        //   Authorization: `Bearer ${token}`,
        // },
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        }
      })
      const data = await res.json()
      const formatted = data.flatMap((msg: any): Message[] => [
        { id: `u-${msg._id}`, role: "user", content: msg.question },
        { id: `a-${msg._id}`, role: "assistant", content: msg.answer },
      ])
      setMessages(formatted)
    } catch (error) {
      console.error("Gagal mengambil riwayat", error)
    }
  }

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!input.trim()) return

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ query: input }),
      })
      const data = await res.json()
      const botMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.answer,
      }
      setMessages((prev) => [...prev, botMessage])
    } catch (err) {
      console.error("Gagal fetch chatbot", err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Card className="flex flex-col h-full overflow-hidden">
      <CardContent className="flex flex-col flex-1 p-0 h-full overflow-x-hidden">
        <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-2">
          <ChatMessages messages={messages}>
            <MessageList messages={messages} isTyping={isLoading} />
          </ChatMessages>
        </div>

        <form onSubmit={handleSubmit} className="border-t p-4 bg-white">
          <MessageInput
            value={input}
            onChange={(e) => setInput(e.target.value)}
            isGenerating={isLoading}
            stop={() => setIsLoading(false)}
            allowAttachments
            files={[]}
            setFiles={() => {}}
          />
        </form>
      </CardContent>
    </Card>
  )
}
