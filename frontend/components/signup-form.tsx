"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export function SignupForm({
  className,
  ...props
}: React.ComponentProps<"form">) {
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [message, setMessage] = useState("")
  const [error, setError] = useState("")
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setMessage("")
    setError("")

    if (password !== confirmPassword) {
      setError("Password tidak cocok.")
      return
    }

    try {
      const res = await fetch("http://localhost:8000/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      })

      const data = await res.json()
      if (res.ok) {
        setMessage("Pendaftaran berhasil. Silakan login.")
        setTimeout(() => {
          router.push("/login") // redirect otomatis ke login
        }, 1500)
      } else {
        setError(data.detail || "Gagal mendaftar.")
      }
    } catch (err) {
      setError("Gagal terhubung ke server.")
    }
  }

  return (
    <form onSubmit={handleSubmit} className={cn("flex flex-col gap-6", className)} {...props}>
      <div className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-2xl font-bold">Create your account</h1>
        <p className="text-muted-foreground text-sm text-balance">
          Enter your details to create a new account
        </p>
      </div>
      <div className="grid gap-6">
        <div className="grid gap-3">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="grid gap-3">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <div className="grid gap-3">
          <Label htmlFor="confirm-password">Confirm Password</Label>
          <Input
            id="confirm-password"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
          />
        </div>
        <Button type="submit" className="w-full">
          Sign up
        </Button>
        {(error || message) && (
          <p
            className={cn(
              "text-sm text-center",
              error ? "text-red-500" : "text-green-600"
            )}
          >
            {error || message}
          </p>
        )}
      </div>
      <div className="text-center text-sm">
        Already have an account?{" "}
        <a href="/login" className="underline underline-offset-4">
          Login
        </a>
      </div>
    </form>
  )
}



// import { cn } from "@/lib/utils"
// import { Button } from "@/components/ui/button"
// import { Input } from "@/components/ui/input"
// import { Label } from "@/components/ui/label"

// export function SignupForm({
//   className,
//   ...props
// }: React.ComponentProps<"form">) {
//   return (
//     <form className={cn("flex flex-col gap-6", className)} {...props}>
//       <div className="flex flex-col items-center gap-2 text-center">
//         <h1 className="text-2xl font-bold">Create your account</h1>
//         <p className="text-muted-foreground text-sm text-balance">
//           Enter your details to create a new account
//         </p>
//       </div>
//       <div className="grid gap-6">
//         <div className="grid gap-3">
//           <Label htmlFor="email">Email</Label>
//           <Input id="email" type="email" placeholder="m@example.com" required />
//         </div>
//         <div className="grid gap-3">
//           <Label htmlFor="password">Password</Label>
//           <Input id="password" type="password" required />
//         </div>
//         <div className="grid gap-3">
//           <Label htmlFor="confirm-password">Confirm Password</Label>
//           <Input id="confirm-password" type="password" required />
//         </div>
//         <Button type="submit" className="w-full">
//           Sign up
//         </Button>
//       </div>
//       <div className="text-center text-sm">
//         Already have an account?{" "}
//         <a href="/login" className="underline underline-offset-4">
//           Login
//         </a>
//       </div>
//     </form>
//   )
// }
