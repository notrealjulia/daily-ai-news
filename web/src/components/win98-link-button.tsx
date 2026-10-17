import * as React from "react"
import { cn } from "cn"

/** A plain link drawn as a Windows 98 push button: raised, a pastel tint on hover,
 *  pressed in while held, and the dotted focus ring. Being a real <a href>, navigation,
 *  new tabs and the browser's back and forward buttons all work as usual. */
function Win98LinkButton({ className, ...props }: React.ComponentProps<"a">) {
  return (
    <a
      className={cn(
        "inline-flex items-center bg-window px-3 py-1.5 font-ui text-[11px] font-bold whitespace-nowrap text-foreground bevel-raised outline-offset-[-4px] outline-foreground select-none hover:bg-pastel-pink focus-visible:outline-1 focus-visible:outline-dotted active:translate-x-px active:translate-y-px active:bevel-sunken",
        className
      )}
      {...props}
    />
  )
}

export { Win98LinkButton }
