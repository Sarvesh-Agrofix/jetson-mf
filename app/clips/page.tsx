"use client"

import { useState, useMemo } from "react"

const CLIP_BASE = "http://localhost:8080"

type Clip = {
  filename: string
  size_mb: number
  size_bytes: number
  modified: string
  url: string
  playable: boolean
}

const jsonFetcher = async (url: string) => {
  const res = await fetch(url, { cache: "no-store" })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    throw new Error(text || `Request failed: ${res.status}`)
  }
  const json = await res.json()

  // Handle both array response (old) and object response (new)
  if (Array.isArray(json)) {
    return json
  }

  // New format: { clips: [...], total: 0, directory: "..." }
  return json.clips || []
}

function formatDate(input?: string) {
  if (!input) return "Unknown date"
  try {
    const d = new Date(input)
    if (isNaN(d.getTime())) return "Unknown date"
    return new Intl.DateTimeFormat(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(d)
  } catch {
    return "Unknown date"
  }
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 B"
  const k = 1024
  const sizes = ["B", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i]
}

export default function ClipsPage() {
  const [clips, setClips] = useState<Clip[]>([])
  const [selected, setSelected] = useState<Clip | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<string>("Never")

  // Fetch clips on mount and set up refresh interval
  const [initialized, setInitialized] = useState(false)

  if (!initialized) {
    setInitialized(true)

    const fetchClips = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const data = await jsonFetcher(`${CLIP_BASE}/clips`)
        setClips(data || [])
        setLastRefresh(new Date().toLocaleTimeString())
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load clips")
        setClips([])
      } finally {
        setIsLoading(false)
      }
    }

    fetchClips()

    // Set up auto-refresh every 30 seconds
    const interval = setInterval(fetchClips, 30000)
    return () => clearInterval(interval)
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <header className="space-y-1">
          <h1 className="text-3xl font-bold">Video Clips</h1>
          <p className="text-sm text-gray-600">
            Browse recorded clips • Last updated: {lastRefresh}
          </p>
        </header>

        {/* Error Alert */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <h3 className="font-semibold text-red-900">Failed to load clips</h3>
            <p className="text-sm text-red-700 mt-1">{error}</p>
            <p className="text-xs text-red-600 mt-2">
              Make sure backend is running at {CLIP_BASE}
            </p>
          </div>
        )}

        {/* Video Player */}
        {selected && (
          <div className="border rounded-lg shadow-lg overflow-hidden bg-white">
            <div className="bg-black p-4">
              <video
                className="w-full max-h-96 rounded"
                controls
                preload="metadata"
                src={`${CLIP_BASE}${selected.url}`}
              />
            </div>
            <div className="p-4 bg-white">
              <h2 className="font-semibold text-lg break-all">{selected.filename}</h2>
              <div className="text-sm text-gray-600 mt-2 space-y-1">
                <p>Modified: {formatDate(selected.modified)}</p>
                <p>Size: {selected.size_mb} MB ({formatFileSize(selected.size_bytes)})</p>
              </div>
            </div>
          </div>
        )}

        {/* Clips Grid */}
        <section className="space-y-4">
          <h2 className="text-xl font-semibold">
            {clips.length} Clip{clips.length !== 1 ? "s" : ""} Available
          </h2>

          {isLoading && !clips.length && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <div
                  key={i}
                  className="rounded-lg border bg-gray-100 p-4 animate-pulse"
                >
                  <div className="aspect-video rounded-md bg-gray-300" />
                  <div className="mt-3 h-4 w-2/3 rounded bg-gray-300" />
                  <div className="mt-2 h-3 w-1/3 rounded bg-gray-300" />
                </div>
              ))}
            </div>
          )}

          {!isLoading && clips.length === 0 && !error && (
            <div className="rounded-lg border-2 border-dashed bg-gray-50 p-12 text-center text-gray-500">
              <svg
                className="mx-auto h-12 w-12 mb-4 text-gray-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M7 4v16m10-16v16M4 8h16M4 16h16"
                />
              </svg>
              <p className="text-lg">No clips available yet</p>
              <p className="text-sm">Recorded clips will appear here</p>
            </div>
          )}

          {clips.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {clips.map((clip) => (
                <button
                  key={clip.filename}
                  onClick={() => setSelected(clip)}
                  className={`rounded-lg border overflow-hidden transition-all hover:shadow-lg hover:border-blue-500 ${selected?.filename === clip.filename
                      ? "ring-2 ring-blue-500 shadow-lg"
                      : ""
                    }`}
                  aria-label={`Play clip ${clip.filename}`}
                >
                  {/* Thumbnail */}
                  <div className="relative bg-gray-900 aspect-video flex items-center justify-center group">
                    <img
                      src="/video-thumbnail-placeholder.png"
                      alt=""
                      className="absolute inset-0 w-full h-full object-cover opacity-50"
                      onError={(e) => {
                        ; (e.target as HTMLImageElement).style.display = "none"
                      }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-br from-black/30 to-black/50" />
                    <div className="relative h-12 w-12 rounded-full bg-blue-500/80 flex items-center justify-center transition-transform group-hover:scale-110">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="white"
                        aria-hidden="true"
                      >
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    </div>
                  </div>

                  {/* Info */}
                  <div className="p-3 bg-white">
                    <div className="text-sm font-semibold line-clamp-2 break-all">
                      {clip.filename}
                    </div>
                    <div className="text-xs text-gray-600 mt-2 space-y-1">
                      <div>{formatDate(clip.modified)}</div>
                      <div className="font-mono">{clip.size_mb} MB</div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  )
}