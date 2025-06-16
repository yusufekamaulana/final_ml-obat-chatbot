// app/(main)/layout.tsx
"use client"

import { usePathname } from "next/navigation"
import { AppSidebar } from "@/components/app-sidebar"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"

export default function MainLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center px-4 gap-2">
          <SidebarTrigger className="-ml-1" />

          {/* tampilkan judul hanya di /chatbot */}
          {pathname === "/chatbot" && (
            <h1 className="text-xl font-bold">MediBot AI Assistant</h1>
          )}
        </header>

        <main className="flex-1 h-screen">{children}</main>
      </SidebarInset>
    </SidebarProvider>
  )
}
