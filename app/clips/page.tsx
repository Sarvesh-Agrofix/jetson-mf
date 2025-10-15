"use client"

import useSWR from "swr"
import { useMemo, useState } from "react"
import { API_BASE } from "@/lib/config"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

type Clip = {
  filename: string
  created_at?: string
  created?: string
  createdAt?: string
  // any other fields ignored
}

const jsonFetcher = async (url: string) => {
  const res = await fetch(url, { cache: "no-store" })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    throw new Error(text || `Request failed: ${res.status}`)
  }
  return (await res.json()) as Clip[]
}

function formatDate(input?: string) {
  if (!input) return "Unknown date"
  const d = new Date(input)
  if (isNaN(d.getTime())) return "Unknown date"
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(d)
}

export default function ClipsPage() {
  const { data, error, isLoading } = useSWR(`${API_BASE}/clips`, jsonFetcher, {
    revalidateOnFocus: true,
    refreshInterval: 30_000,
  })
  const [selected, setSelected] = useState<Clip | null>(null)

  const clips = useMemo(() => data ?? [], [data])

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold text-pretty">Clips</h1>
        <p className="text-sm text-muted-foreground">Browse recorded clips. List auto-refreshes every 30 seconds.</p>
      </header>

      {error && (
        <Alert variant="destructive" role="alert">
          <AlertTitle>Failed to load clips</AlertTitle>
          <AlertDescription>Ensure your backend is running at {API_BASE}.</AlertDescription>
        </Alert>
      )}

      {selected && (
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="text-balance">{selected.filename}</CardTitle>
            <CardDescription>
              {formatDate(selected.created_at || selected.created || selected.createdAt)}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="rounded-lg border bg-card">
              <video
                className="w-full rounded-lg"
                controls
                preload="metadata"
                src={`${API_BASE}/clips/${encodeURIComponent(selected.filename)}`}
              />
            </div>
          </CardContent>
        </Card>
      )}

      <section className="space-y-3">
        <h2 className="sr-only">Clip list</h2>

        {isLoading && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="rounded-lg border bg-card p-4">
                <div className="aspect-video rounded-md bg-muted" />
                <div className="mt-3 h-4 w-2/3 rounded bg-muted" />
                <div className="mt-2 h-3 w-1/3 rounded bg-muted" />
              </div>
            ))}
          </div>
        )}

        {!isLoading && clips.length === 0 && !error && (
          <div className="rounded-lg border bg-card p-6 text-center text-muted-foreground">No clips available yet.</div>
        )}

        {clips.length > 0 && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
            {clips.map((clip) => {
              const dt = clip.created_at || clip.created || clip.createdAt
              return (
                <button
                  key={clip.filename}
                  onClick={() => setSelected(clip)}
                  className={cn(
                    "group rounded-lg border bg-card p-3 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  )}
                  aria-label={`Play clip ${clip.filename}`}
                >
                  <div className="relative">
                    <img
                      src="/video-thumbnail-placeholder.png"
                      alt=""
                      className="aspect-video w-full rounded-md border object-cover"
                    />
                    <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
                      <div className="h-10 w-10 rounded-full bg-primary/80 text-primary-foreground grid place-items-center transition-transform group-hover:scale-105">
                        <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true" className="fill-current">
                          <path d="M8 5v14l11-7z" />
                        </svg>
                      </div>
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="text-sm font-medium text-pretty">{clip.filename}</div>
                    <div className="text-xs text-muted-foreground">{formatDate(dt)}</div>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </section>
    </div>
  )
}
