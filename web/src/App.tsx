import { useEffect, useState } from "react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Sparkles } from "@/components/sparkles"
import { Win98Audio } from "@/components/win98-audio"
import { Win98Window } from "@/components/win98-window"

// The shape of /dashboard.json (see web/server.py). All the text is worked out in Python,
// by ainews.dashboard.
type Story = {
  title: string
  summary: string
  sources: { name: string; url: string | null }[]
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

function Stories({ category }: { category: Category }) {
  return (
    <Collapsible className="flex flex-col gap-1">
      {/* Raised while closed, pressed in while open; the arrow has its own little button. */}
      <CollapsibleTrigger className="group flex w-full items-center justify-between bg-window p-0.5 pl-2 font-ui text-[11px] bevel-raised outline-offset-[-4px] outline-foreground focus-visible:outline-1 focus-visible:outline-dotted data-[panel-open]:bevel-sunken">
        {category.expander_label}
        <span className="flex size-5 items-center justify-center bg-window bevel-raised">
          <svg
            viewBox="0 0 7 4"
            width={7}
            height={4}
            className="fill-foreground group-data-[panel-open]:rotate-180"
            shapeRendering="crispEdges"
            aria-hidden="true"
          >
            <path d="M0 0h7v1H6v1H5v1H4v1H3V3H2V2H1V1H0z" />
          </svg>
        </span>
      </CollapsibleTrigger>
      {/* The bevel is on the panel and the scrolling on the pane inside it, so the edge
          stays put while the stories move. Short lists just end; long ones scroll. */}
      <CollapsibleContent className="bg-[#fae6f8] p-0.5 bevel-sunken">
        <div className="scrollbar-win98 flex max-h-80 flex-col gap-4 overflow-y-auto p-2.5">
          {category.stories.map((story, index) => (
            <div key={index} className="flex flex-col gap-1">
              <strong className="leading-snug">{story.title}</strong>
              <p className="leading-[1.625rem]">{story.summary}</p>
              <p className="font-ui text-[11px]">
                Sources:{" "}
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
            </div>
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

function CategoryCard({
  category,
  emptyText,
}: {
  category: Category
  emptyText: string
}) {
  return (
    <Win98Window title={category.name}>
      {category.stories.length === 0 ? (
        <p className="font-ui text-[11px] text-muted-foreground">{emptyText}</p>
      ) : (
        <>
          {category.headline && (
            <h4 className="text-2xl leading-tight font-bold">
              {category.headline}
            </h4>
          )}
          {category.digest && (
            <p className="leading-[1.625rem]">{category.digest}</p>
          )}
          {category.audio && <Win98Audio src={category.audio} />}
          <Stories category={category} />
        </>
      )}
    </Win98Window>
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

  return (
    <>
      <Sparkles />
      {/* Page margins: the content takes the middle 14 of 16 columns. */}
      <main className="mx-auto flex w-7/8 flex-col gap-2 py-8">
        <header className="flex flex-wrap items-end gap-x-4 gap-y-1">
          <h1 className="text-[2rem] font-bold">Daily AI News Briefing</h1>
          {data && (
            <p className="pb-1.5 font-ui text-[11px] text-muted-foreground">
              {data.header}
            </p>
          )}
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
                <p className="font-ui text-[11px] text-muted-foreground">
                  Or scroll down to listen by category.
                </p>
              </>
            )}
            {/* Cards in a row stretch to the taller one, content at the top. */}
            <div className="grid gap-2 md:grid-cols-2">
              {data.categories.map((category) => (
                <CategoryCard
                  key={category.name}
                  category={category}
                  emptyText={data.empty_text}
                />
              ))}
            </div>
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
