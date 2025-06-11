import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export function middleware(request: NextRequest) {
  const token = request.cookies.get("token")?.value

  const protectedPaths = ["/dashboard", "/chatbot", "/profile"]
  const isProtected = protectedPaths.some((path) =>
    request.nextUrl.pathname.startsWith(path)
  )

  if (isProtected && !token) {
    return NextResponse.redirect(new URL("/login", request.url))
  }

  return NextResponse.next()
}

// Konfigurasi matcher juga harus diekspor
export const config = {
  matcher: ["/dashboard/:path*", "/chatbot/:path*", "/profile/:path*"],
}
