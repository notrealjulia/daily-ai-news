import { useEffect, useState } from "react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Sparkles } from "@/components/sparkles"
import { Win98Audio } from "@/components/win98-audio"
import { Win98LinkButton } from "@/components/win98-link-button"
import { Win98Window } from "@/components/win98-window"

// The shape of /dashboard.json (see web/server.py). All the text is worked out in Python,
// by ainews.dashboard.
type Story = {
  title: string
  summary: string
  sources: { name: string; url: string | null }[]
  significance: number | null // null: the story is unscored
  release: number | null
  published: string | null // the newest article's time, "2026-10-27T08:50:53Z"
}

type Category = {
  name: string
  headline: string | null
  digest: string | null
  audio: string | null
  expander_label: string
  stories: Story[]
}

type Payload = {
  database?: string
  data: {
    header: string
    briefing_audio: string | null
    empty_text: string
    sources: string | null
    categories: Category[]
  } | null
}

type FeedStory = Story & { category: string }

// Highest significance + release first, then higher significance, then the newest.
// Unscored stories (shown on purpose, see defaults.qualifies) go after all scored ones.
function byRank(a: FeedStory, b: FeedStory) {
  const total = (s: FeedStory) =>
    s.significance === null ? -1 : s.significance + (s.release ?? 0)
  return (
    total(b) - total(a) ||
    (b.significance ?? -1) - (a.significance ?? -1) ||
    (b.published ?? "").localeCompare(a.published ?? "")
  )
}

function FeedItem({ story }: { story: FeedStory }) {
  return (
    <article className="flex flex-col gap-1">
      <h2 className="text-xl leading-snug font-bold">{story.title}</h2>
      <p className="font-ui text-[11px] text-muted-foreground">
        {story.category} · Sources:{" "}
        {story.sources.map((source, i) => (
          <span key={i}>
            {i > 0 && " · "}
            {source.url ? (
              <a
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary underline underline-offset-3"
              >
                {source.name}
              </a>
            ) : (
              source.name
            )}
          </span>
        ))}
      </p>
      <p className="leading-[1.625rem]">{story.summary}</p>
    </article>
  )
}

export function App() {
  const [payload, setPayload] = useState<Payload | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    fetch("/dashboard.json")
      .then((response) => {
        if (!response.ok) throw new Error(response.statusText)
        return response.json()
      })
      .then(setPayload)
      .catch(() => setFailed(true))
  }, [])

  const data = payload?.data
  const feed = (data?.categories ?? [])
    .flatMap((c) => c.stories.map((story) => ({ ...story, category: c.name })))
    .sort(byRank)

  return (
    <>
      <Sparkles />
      {/* Page margins: the middle 14 of 16 columns, capped at a readable line length. */}
      <main className="mx-auto flex w-7/8 max-w-4xl flex-col gap-2 py-8">
        <header className="flex flex-wrap items-end gap-x-4 gap-y-1">
          <h1 className="text-[2rem] font-bold">Daily AI News Briefing</h1>
          {data && (
            <p className="pb-1.5 font-ui text-[11px] text-muted-foreground">
              {data.header}
            </p>
          )}
          <Win98LinkButton
            href="/timeline.html"
            className="ml-auto self-center"
          >
            AI Timeline →
          </Win98LinkButton>
        </header>

        {failed && (
          <Alert variant="destructive">
            <AlertTitle>Couldn't load the briefing</AlertTitle>
            <AlertDescription>
              Try reloading the page. Running locally? Start the data server
              from the project root with <code>python -m web.server</code>{" "}
              first.
            </AlertDescription>
          </Alert>
        )}

        {payload && !data && (
          <Alert>
            <AlertDescription>
              No completed run yet. Run <code>python -m ainews cluster</code>,
              then <code>python -m ainews digest</code>, and reload this page.
              (This page is reading {payload.database}.)
            </AlertDescription>
          </Alert>
        )}

        {data && (
          <>
            {data.briefing_audio && (
              <>
                <p className="font-ui text-[11px] font-bold">
                  Listen to the entire AI briefing
                </p>
                <Win98Audio src={data.briefing_audio} />
              </>
            )}
            <Win98Window title="Today's stories" minimizable={false}>
              {feed.length === 0 ? (
                <p className="font-ui text-[11px] text-muted-foreground">
                  {data.empty_text}
                </p>
              ) : (
                <div className="flex flex-col gap-6">
                  {feed.map((story, index) => (
                    <FeedItem key={index} story={story} />
                  ))}
                </div>
              )}
            </Win98Window>
            {data.sources && (
              <p className="font-ui text-[11px] leading-relaxed text-muted-foreground">
                {data.sources}
              </p>
            )}
          </>
        )}
      </main>
    </>
  )
}

export default App
