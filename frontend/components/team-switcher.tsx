"use client"

import Image from "next/image"
import { useSidebar } from "@/components/ui/sidebar"

export function TeamSwitcher() {
  const { state } = useSidebar()

  return (
    <div className="flex items-center justify-center h-16 w-full">
      {state === "collapsed" ? (
        <Image
          src="/logoicon.svg"
          alt="Logo Icon"
          width={72}
          height={72}
          className="h-14 w-14 object-contain"
        />
      ) : (
        <Image
          src="/logotype.svg"
          alt="Logo Type"
          width={180}
          height={48}
          className="h-10 w-auto object-contain pl-2"
        />
      )}
    </div>
  )
}
