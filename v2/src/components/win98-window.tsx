import * as React from "react"
import { cn } from "cn"

/** A Windows 98-style application window: a beveled frame, a title bar and a content
 *  area. Colors come from the pastel palette in index.css, where `bg-window` and
 *  `bevel-raised` are defined. */
function Win98Window({
  title,
  className,
  children,
  ...props
}: Omit<React.ComponentProps<"section">, "title"> & {
  title: React.ReactNode
}) {
  return (
    <section
      className={cn(
        "flex flex-col bg-window p-[3px] bevel-raised drop-shadow-[3px_3px_0_oklch(from_var(--muted-purple)_l_c_h_/_30%)]",
        className
      )}
      {...props}
    >
      <header className="bg-primary px-1.5 py-0.5 text-primary-foreground">
        <h3 className="truncate text-base font-bold uppercase">{title}</h3>
      </header>
      <div className="flex flex-1 flex-col gap-2 p-3">{children}</div>
    </section>
  )
}

export { Win98Window }
