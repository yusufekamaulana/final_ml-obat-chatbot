"use client"

import * as React from "react"
import {
  Home,
  Pill,
  History,
  User,
  Bot,
} from "lucide-react"

import { NavUser } from "@/components/nav-user"
import { NavMain } from "@/components/nav-main"
import { TeamSwitcher } from "@/components/team-switcher"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
} from "@/components/ui/sidebar"
import { usePathname } from "next/navigation"

const user = {
  name: "shadcn",
  email: "m@example.com",
  avatar: "/avatars/shadcn.jpg",
}

const navMain = [
  {
    title: "Main Menu",
    url: "/dashboard",
    icon: Pill,
    items: undefined,
  },
  // {
  //   title: "Etalase Obat",
  //   url: "/etalase",
  //   icon: Pill,
  //   items: undefined,
  // },
  // {
  //   title: "Riwayat & Bookmark",
  //   url: "/riwayat",
  //   icon: History,
  //   items: undefined,
  // },
  {
    title: "Chatbot",
    url: "/chatbot",
    icon: Bot,
    items: undefined,
  },
]

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="icon" {...props}>
      <SidebarHeader>
        <TeamSwitcher />
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={navMain} />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={user} />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
