"use client"

import { useEffect } from "react"
import { ChatbotLayout } from "@/components/chatbot-ui/chatbot-page"

export default function ChatbotPage() {
  useEffect(() => {
    // Kunci scroll halaman saat chatbot aktif
    document.body.style.overflow = "hidden"
    return () => {
      // Pulihkan scroll saat keluar dari chatbot
      document.body.style.overflow = ""
    }
  }, [])

  return (
    <div className="h-screen overflow-hidden p-4">
      <ChatbotLayout />
    </div>
  )
}
