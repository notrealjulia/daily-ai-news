import type { CSSProperties } from "react"

// Decorative pixel stars over the desktop background, behind everything else. Fixed to
// the viewport like the background itself, so they don't scroll with the page.
//
// Every star holds all three frames of the animation, drawn on one 7x7 grid around the
// same center pixel: a dot, a small cross and its sparkle (one of PEAKS). CSS in index.css
// (.pixel-star) shows one frame at a time: dot, cross, sparkle, cross, dot, then a long
// rest. With reduced motion, each star just shows its `rest` frame and stays still.

const DOT = "M3 3h1v1H3z"
const CROSS = "M3 2h1v1h1v1H4v1H3V4H2V3h1z"

// The sparkle each star peaks at, so they don't all resolve into the same shape.
const PEAKS = {
  // four points around a 3x3 core
  classic: "M3 0h1v2h1v1h2v1H5v1H4v2H3V5H2V4H0V3h2V2h1z",
  // one long vertical axis, a shorter horizontal one
  tall: "M3 0h1v3h1v1H4v3H3V4H2V3h1z",
  // a small solid diamond
  diamond: "M3 1h1v1h1v1h1v1H5v1H4v1H3V5H2V4H1V3h1V2h1z",
  // long cross plus four diagonal glints: eight points
  burst:
    "M3 0h1v3h3v1H4v3H3V4H0V3h3zM1 1h1v1H1zM5 1h1v1H5zM1 5h1v1H1zM5 5h1v1H5z",
}

type Frame = "dot" | "cross" | "sparkle"

// [left %, top %, rest frame, peak, seconds per cycle, delay in seconds]. Hand-placed
// rather than random so they stay sparse and mostly fall where the windows don't cover
// them; the peaks are mixed so no two neighbours along an edge match.
const STARS: [number, number, Frame, keyof typeof PEAKS, number, number][] = [
  [2, 6, "sparkle", "classic", 7, 0],
  [4, 31, "dot", "diamond", 5, 2.1],
  [3, 58, "cross", "burst", 6, 4.3],
  [2.5, 84, "dot", "tall", 4.5, 1.2],
  [5, 72, "sparkle", "diamond", 8, 5.6],
  [1.5, 18, "dot", "burst", 6, 3.7],
  [3.5, 40, "cross", "tall", 5.5, 0.9],
  [1, 49, "dot", "classic", 7.5, 5.9],
  [4, 65, "dot", "tall", 5, 2.6],
  [2, 94, "sparkle", "diamond", 6.5, 4.6],
  [96, 9, "cross", "tall", 6.5, 3.4],
  [97.5, 27, "dot", "classic", 5, 0.7],
  [95, 47, "sparkle", "burst", 7.5, 2.8],
  [97, 66, "dot", "diamond", 4, 4.9],
  [96, 90, "cross", "classic", 6, 1.8],
  [98, 18, "dot", "diamond", 5.5, 1.4],
  [95.5, 37, "cross", "tall", 7, 4.4],
  [97.5, 56, "dot", "classic", 4.5, 0.2],
  [95, 77, "sparkle", "burst", 8, 3.1],
  [97, 97, "dot", "tall", 6, 5.4],
  [18, 2, "dot", "burst", 5.5, 3.1],
  [34, 4, "cross", "diamond", 7, 0.4],
  [52, 1.5, "dot", "classic", 4.5, 5.2],
  [68, 3.5, "sparkle", "tall", 8.5, 2.4],
  [83, 2, "dot", "diamond", 5, 4.1],
  [44, 97, "cross", "tall", 6.5, 1.5],
  [72, 96, "dot", "burst", 5, 3.8],
  [23, 98, "sparkle", "classic", 7, 6.2],
]

function Sparkles() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
    >
      {STARS.map(([left, top, rest, peak, duration, delay]) => {
        // The stars that rest as sparkles are drawn a size up, so the field has depth.
        const size = rest === "sparkle" ? 21 : 14
        return (
          <svg
            key={`${left}-${top}`}
            viewBox="0 0 7 7"
            width={size}
            height={size}
            shapeRendering="crispEdges"
            className="pixel-star absolute fill-pastel-cream opacity-60"
            style={
              {
                left: `${left}%`,
                top: `${top}%`,
                "--sparkle-duration": `${duration}s`,
                "--sparkle-delay": `${delay}s`,
              } as CSSProperties
            }
          >
            {Object.entries({
              dot: DOT,
              cross: CROSS,
              sparkle: PEAKS[peak],
            }).map(([frame, d]) => (
              <path
                key={frame}
                d={d}
                className={frame}
                data-rest={frame === rest || undefined}
              />
            ))}
          </svg>
        )
      })}
    </div>
  )
}

export { Sparkles }
