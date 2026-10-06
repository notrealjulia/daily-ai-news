import { useRef, useState } from "react"

// Pixel icons, each drawn on a 9-row grid and shown at exactly twice that size.
const ICONS = {
  play: { width: 9, d: "M2 1h1v1h1v1h1v1h1v1H5v1H4v1H3v1H2z" },
  pause: { width: 9, d: "M2 1h2v7H2zM5 1h2v7H5z" },
  stop: { width: 9, d: "M2 2h5v5H2z" },
  sound: {
    width: 10,
    d: "M1 3h2v3H1zM3 3h1V2h1V1h1v7H5V7H4V6H3zM8 3h1v3H8z",
  },
  muted: {
    width: 10,
    d: "M1 3h2v3H1zM3 3h1V2h1V1h1v7H5V7H4V6H3zM7 3h1v1H7zM9 3h1v1H9zM8 4h1v1H8zM7 5h1v1H7zM9 5h1v1H9z",
  },
}

function PixelButton({
  icon,
  label,
  onClick,
}: {
  icon: keyof typeof ICONS
  label: string
  onClick: () => void
}) {
  const { width, d } = ICONS[icon]
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="flex size-8 shrink-0 items-center justify-center bg-window bevel-raised outline-offset-[-5px] outline-foreground focus-visible:outline-1 focus-visible:outline-dotted active:bevel-sunken"
    >
      <svg
        viewBox={`0 0 ${width} 9`}
        width={width * 2}
        height={18}
        className="fill-foreground"
        shapeRendering="crispEdges"
        aria-hidden="true"
      >
        <path d={d} />
      </svg>
    </button>
  )
}

function clock(seconds: number) {
  if (!Number.isFinite(seconds)) return "--:--"
  const whole = Math.floor(seconds)
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`
}

/** A Windows 98-style audio player: play/pause, stop, a segmented progress bar you can
 *  click, drag or arrow-key to seek, mute, and a status bar. The sound itself is a plain
 *  <audio> element; this only replaces the browser's own controls. */
function Win98Audio({ src }: { src: string }) {
  const audio = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(NaN)

  const status = playing ? "Playing" : time > 0 ? "Paused" : "Stopped"
  const percent = duration > 0 ? (time / duration) * 100 : 0

  return (
    <div className="flex flex-col gap-1 bg-window p-1.5 bevel-raised">
      <audio
        ref={audio}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
        onDurationChange={(event) => setDuration(event.currentTarget.duration)}
        onVolumeChange={(event) => setMuted(event.currentTarget.muted)}
      />
      <div className="flex items-center gap-1">
        <PixelButton
          icon={playing ? "pause" : "play"}
          label={playing ? "Pause" : "Play"}
          onClick={() =>
            playing ? audio.current?.pause() : audio.current?.play()
          }
        />
        <PixelButton
          icon="stop"
          label="Stop"
          onClick={() => {
            if (!audio.current) return
            audio.current.pause()
            audio.current.currentTime = 0
          }}
        />
        {/* The blocks are the picture; the invisible range input on top is the control. */}
        <div className="relative h-8 flex-1 bg-pastel-cream p-1 bevel-sunken outline-offset-[-5px] outline-foreground has-focus-visible:outline-1 has-focus-visible:outline-dotted">
          <div
            className="h-full bg-[repeating-linear-gradient(to_right,var(--muted-purple)_0_10px,transparent_10px_12px)]"
            style={{ width: `${percent}%` }}
          />
          <input
            type="range"
            aria-label="Seek"
            min={0}
            max={Number.isFinite(duration) ? duration : 0}
            step="any"
            value={time}
            onChange={(event) => {
              if (audio.current)
                audio.current.currentTime = Number(event.target.value)
            }}
            className="absolute inset-0 size-full cursor-pointer opacity-0"
          />
        </div>
        <PixelButton
          icon={muted ? "muted" : "sound"}
          label={muted ? "Unmute" : "Mute"}
          onClick={() => {
            if (audio.current) audio.current.muted = !audio.current.muted
          }}
        />
      </div>
      <div className="flex gap-1 font-ui text-[11px]">
        <span className="flex-1 px-1.5 py-1 bevel-sunken">{status}</span>
        <span className="px-1.5 py-1 bevel-sunken">
          {clock(time)} / {clock(duration)}
        </span>
      </div>
    </div>
  )
}

export { Win98Audio }
