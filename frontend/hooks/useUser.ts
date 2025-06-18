import { useState, useEffect } from "react"
import Cookies from "js-cookie"
import { fetchWithFallback } from "@/lib/api"

type User = {
  name: string
  email: string
  avatar: string
}

export function useUser() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = Cookies.get("token")
    if (!token) {
      setLoading(false)
      return
    }

    fetchWithFallback("/auth/me", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    })
      .then((data) => {
        setUser(data)
      })
      .catch(() => {
        setUser(null)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  return { user, loading }
}
