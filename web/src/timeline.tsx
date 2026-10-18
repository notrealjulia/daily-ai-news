import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { Popover } from "@base-ui/react/popover"

import "./index.css"
import { Sparkles } from "@/components/sparkles"
import { Win98LinkButton } from "@/components/win98-link-button"
import { Win98Window } from "@/components/win98-window"
import { companies, releases, type Release } from "@/lib/releases"

// The AI Timeline (timeline.html): foundation model releases and company
// milestones (the same cards in a darker shade of the company's color), newest first, a
// column per company in the CSV. The columns share one scale, the date axis at the far
// left: a row per release date, so releases on the same day sit level with each other
// and the order down the page is the order in time. More companies just add columns;
// the pane scrolls sideways when they don't fit.

// Each company's color, by column; cycles if there are ever more companies than colors.
const FACES = [
  "bg-pastel-pink",
  "bg-pastel-blue",
  "bg-pastel-mint",
  "bg-[oklch(0.93_0.07_95)]", // pastel yellow
  "bg-pastel-lavender",
  "bg-[oklch(0.9_0.06_50)]", // pastel peach
  "bg-[oklch(0.92_0.07_130)]", // pastel lime
  "bg-pastel-cream",
]
const face = (company: string) =>
  FACES[companies.indexOf(company) % FACES.length]
// A milestone card: the company's color under a half-black layer, with white text (dark
// enough for it to read). A layer rather than a filter, which would dim the text too.
const MILESTONE =
  "bg-[image:linear-gradient(rgb(0_0_0/0.5),rgb(0_0_0/0.5))] text-white outline-white"
// Companies whose milestone cards have their own colors instead.
const MILESTONE_COLORS: Record<string, string> = {
  OpenAI: "bg-[#322128] text-[#fbcbe0] outline-[#fbcbe0]!",
  Anthropic: "bg-[#2a404c] text-[#b9def4] outline-[#b9def4]!",
}
const cardColor = (release: Release) =>
  !release.milestone
    ? face(release.company)
    : (MILESTONE_COLORS[release.company] ??
      `${face(release.company)} ${MILESTONE}`)

const MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ")

/** "Oct 7" (or just "Jan" when the CSV only gives a month); the year is shown separately. */
function shortDate(date: string) {
  const [, month, day] = date.split("-")
  return `${MONTHS[Number(month) - 1]}${day ? ` ${Number(day)}` : ""}`
}

function longDate(date: string) {
  return `${shortDate(date)}, ${date.slice(0, 4)}`
}

/** Each link's site, numbered when a release has more than one link to the same site. */
function linkLabels(links: string[]) {
  const hosts = links.map((url) => new URL(url).hostname.replace(/^www\./, ""))
  return hosts.map((host, i) => {
    const repeat = hosts.slice(0, i).filter((h) => h === host).length
    return repeat ? `${host} (${repeat + 1})` : host
  })
}

/** A release name that opens its summary and links: on hover with a mouse, on click or
 *  tap everywhere. The popup is a small Win98 window. */
function ReleaseButton({ release }: { release: Release }) {
  return (
    <Popover.Root>
      <Popover.Trigger
        openOnHover
        delay={150}
        closeDelay={150}
        className={`${cardColor(release)} max-w-full px-2.5 py-1.5 text-left leading-tight bevel-raised outline-offset-[-4px] outline-foreground focus-visible:outline-1 focus-visible:outline-dotted active:bevel-sunken data-[popup-open]:bevel-sunken`}
      >
        {release.name}
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Positioner
          side="bottom"
          sideOffset={4}
          collisionPadding={12}
          className="z-50"
        >
          <Popover.Popup className="flex w-[min(22rem,calc(100vw-24px))] flex-col bg-window p-[3px] bevel-raised drop-shadow-[3px_3px_0_oklch(from_var(--muted-purple)_l_c_h_/_30%)] outline-none">
            <Popover.Title className="bg-primary px-1.5 py-0.5 font-ui text-[11px] font-bold text-primary-foreground">
              {release.company} · {longDate(release.date)}
              {release.milestone && " · Milestone"}
            </Popover.Title>
            <div className="flex flex-col gap-2 p-3">
              <Popover.Description className="leading-snug">
                {release.summary}
              </Popover.Description>
              {release.links.length > 0 && (
                <p className="font-ui text-[11px]">
                  Sources:{" "}
                  {release.links.map((link, i) => (
                    <span key={link}>
                      {i > 0 && " · "}
                      <a
                        href={link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary underline underline-offset-3"
                      >
                        {linkLabels(release.links)[i]}
                      </a>
                    </span>
                  ))}
                </p>
              )}
            </div>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  )
}

// The date axis is the first column and stays pinned at the left edge while the company
// columns scroll sideways past it. Columns share the pane's width while they fit; when
// they don't, each keeps its minimum width and the pane scrolls.
const COLUMN_REM = 11
const gridTemplateColumns = `auto repeat(${companies.length}, minmax(${COLUMN_REM}rem, 1fr))`

// The window is see-through (the page's pink/purple shows through it), and every other
// row has a light cream band, to follow a date across many columns. What sticks while the
// pane scrolls (the column heads, the date axis) is nearly opaque in the same tints, and
// blurs what still shows through, so cards passing under it don't clash with the dates.
const ROW_SHADES = ["", "bg-pastel-cream/55"]
const AXIS_SHADES = ["bg-window/85", "bg-pastel-cream/85"]
const STICKY = "backdrop-blur-md"

// One row per date that has a release or milestone, newest first.
const dates = [...new Set(releases.map((release) => release.date))]
const onDate = (date: string, company: string) =>
  releases.filter((r) => r.date === date && r.company === company)

function Timeline() {
  return (
    <>
      <Sparkles />
      <main className="mx-auto flex w-7/8 flex-col gap-2 py-8">
        <header className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <Win98LinkButton href="/">← Daily Brief</Win98LinkButton>
          <h1 className="text-[2rem] font-bold">AI Timeline</h1>
        </header>

        <Win98Window
          title="Model releases & milestones"
          minimizable={false}
          className="bg-window/45"
        >
          {/* A pane that scrolls both ways, so the column heads stay on top while the rows
              scroll, and the dates stay at the left while the columns scroll. */}
          <div className="scrollbar-win98 max-h-[calc(100dvh-8rem)] overflow-auto">
            {/* As wide as the pane, or as the columns' minimum widths when that is more:
                the pinned axis can only stay pinned within the grid's own width. */}
            <div
              className="grid w-fit min-w-full"
              style={{ gridTemplateColumns }}
            >
              <div
                className={`sticky top-0 left-0 z-20 bg-window/85 ${STICKY}`}
              />
              {companies.map((company) => (
                <div
                  key={company}
                  className={`sticky top-0 z-10 bg-window/85 px-1 py-1 sm:px-2 ${STICKY}`}
                >
                  <div
                    className={`${face(company)} truncate px-2 py-1 text-center text-base font-bold uppercase bevel-sunken`}
                  >
                    {company}
                  </div>
                </div>
              ))}

              {dates.map((date, index) => {
                const year = date.slice(0, 4)
                const newYear =
                  index === 0 || dates[index - 1].slice(0, 4) !== year
                const shade = ROW_SHADES[index % 2]
                return (
                  <div key={date} className="contents">
                    {newYear && (
                      <div className="col-span-full flex items-center gap-2 px-2 pt-4 pb-1">
                        <span className="sticky left-2 bg-primary px-2 py-0.5 font-ui text-[11px] font-bold text-primary-foreground">
                          {year}
                        </span>
                        <span className="h-px flex-1 bg-border" />
                      </div>
                    )}
                    {/* The axis: a dotted line through every row, the date on top of it. */}
                    <div
                      className={`sticky left-0 z-[5] flex w-14 justify-center py-2 before:absolute before:inset-y-0 before:left-1/2 before:border-l-2 before:border-dotted before:border-muted-purple/50 sm:w-16 ${AXIS_SHADES[index % 2]} ${STICKY}`}
                    >
                      <time
                        dateTime={date}
                        className="relative mt-1 self-start bg-window px-1 py-0.5 text-center font-ui text-[11px] whitespace-nowrap bevel-raised"
                      >
                        {shortDate(date)}
                      </time>
                    </div>
                    {companies.map((company) => (
                      <div
                        key={company}
                        className={`flex flex-col items-start gap-1.5 px-1 py-2 sm:px-2 ${shade}`}
                      >
                        {onDate(date, company).map((release) => (
                          <ReleaseButton key={release.name} release={release} />
                        ))}
                      </div>
                    ))}
                  </div>
                )
              })}
            </div>
          </div>
        </Win98Window>
      </main>
    </>
  )
}
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Timeline />
  </StrictMode>
)
