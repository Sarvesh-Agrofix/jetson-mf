"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

function NavLink({ href, label }: { href: string; label: string }) {
  const pathname = usePathname()
  const active = pathname === href || pathname.startsWith(href + "/")
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={cn(
        "text-sm transition-colors hover:text-foreground",
        active ? "text-foreground font-medium" : "text-muted-foreground",
      )}
    >
      {label}
    </Link>
  )
}

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 border-b bg-card/80 backdrop-blur supports-[backdrop-filter]:bg-card/60">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link href="/livestream" className="text-base font-semibold">
          Cam Dashboard
        </Link>
        <nav className="flex items-center gap-6">
          <NavLink href="/livestream" label="Livestream" />
          <NavLink href="/clips" label="Clips" />
        </nav>
      </div>
    </header>
  )
}
