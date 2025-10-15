"use client"

import useSWR from "swr"
import { useState } from "react"
import { API_BASE } from "@/lib/config"
import { cn } from "@/lib/utils"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

type SignedUrlResponse = {
  url: string
}

const jsonFetcher = async (url: string) => {
  const res = await fetch(url, { cache: "no-store" })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    throw new Error(text || `Request failed: ${res.status}`)
  }
  return (await res.json()) as SignedUrlResponse
}

export default function LivestreamPage() {
  const { data, error, isLoading } = useSWR(`${API_BASE}/get_signed_url`, jsonFetcher, {
    revalidateOnFocus: true,
  })
  const [isActive, setIsActive] = useState(false)

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold text-pretty">Livestream</h1>
        <p className="text-sm text-muted-foreground">Viewing the live MJPEG feed from your camera.</p>
      </header>

      {error ? (
        <Alert variant="destructive" role="alert">
          <AlertTitle>Unable to connect to camera feed</AlertTitle>
          <AlertDescription>Check that your FastAPI server is running at {API_BASE}.</AlertDescription>
        </Alert>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "inline-block h-2.5 w-2.5 rounded-full",
                isActive ? "bg-emerald-500" : "bg-amber-500 animate-pulse",
              )}
              aria-hidden="true"
            />
            <span className="text-sm text-muted-foreground" aria-live="polite">
              {isActive ? "Live Stream Active" : isLoading ? "Connecting to stream…" : "Loading stream…"}
            </span>
          </div>

          <div className="rounded-xl border bg-card p-2">
            {data?.url ? (
              <img
                src={`${API_BASE}${data.url}`}
                alt="Live Video Feed"
                className="rounded-lg border w-full max-h-[70vh] object-contain bg-muted"
                onLoad={() => setIsActive(true)}
                onError={() => setIsActive(false)}
              />
            ) : (
              <div className="aspect-video w-full rounded-lg bg-muted" />
            )}
          </div>
        </div>
      )}
    </div>
  )
}
