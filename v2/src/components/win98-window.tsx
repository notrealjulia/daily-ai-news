import * as React from "react"
import { cn } from "cn"

/** A Windows 98-style application window: a beveled frame, a title bar with a minimize
 *  button, and a content area. Minimizing leaves just the title bar; the content stays
 *  mounted (only hidden), so audio that is playing keeps playing. Colors come from the
 *  pastel palette in index.css, where `bg-window` and `bevel-raised` are defined. */
function Win98Window({
  title,
  className,
  children,
  ...props
}: Omit<React.ComponentProps<"section">, "title"> & {
  title: React.ReactNode
}) {
  const [minimized, setMinimized] = React.useState(false)
  const contentId = React.useId()

  return (
    <section
      className={cn(
        "flex flex-col bg-window p-[3px] bevel-raised drop-shadow-[3px_3px_0_oklch(from_var(--muted-purple)_l_c_h_/_30%)]",
        // Don't stretch to the height of the other window in the row.
        minimized && "self-start",
        className
      )}
      {...props}
    >
      <header className="flex items-center justify-between gap-2 bg-primary py-0.5 pr-0.5 pl-1.5 text-primary-foreground">
        <h3 className="truncate text-base font-bold uppercase">{title}</h3>
        <button
          type="button"
          aria-label={minimized ? "Restore" : "Minimize"}
          title={minimized ? "Restore" : "Minimize"}
          aria-expanded={!minimized}
          aria-controls={contentId}
          onClick={() => setMinimized(!minimized)}
          className="flex h-5 w-6 shrink-0 items-center justify-center bg-window bevel-raised outline-offset-[-4px] outline-foreground focus-visible:outline-1 focus-visible:outline-dotted active:bevel-sunken"
        >
          <svg
            viewBox="0 0 6 5"
            width={12}
            height={10}
            className="fill-foreground"
            shapeRendering="crispEdges"
            aria-hidden="true"
          >
            <path d="M1 3h4v1H1z" />
          </svg>
        </button>
      </header>
      <div
        id={contentId}
        hidden={minimized}
        className="flex flex-1 flex-col gap-2 p-3"
      >
        {children}
      </div>
    </section>
  )
}

export { Win98Window }
