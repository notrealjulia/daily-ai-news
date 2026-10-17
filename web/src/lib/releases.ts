// The model release timeline's data: web/src/data/ai_releases.csv, bundled into the
// page as text at build time. Columns, read by their header: Company name, Release date
// (YYYY-MM-DD, or YYYY-MM when the day is unknown), Release name, Notes, Link 1-3, and
// Release type: "Model release" or "Company milestone". Only model releases are shown;
// the milestones stay in the CSV.
import csvText from "@/data/ai_releases.csv?raw"

export type Release = {
  company: string
  date: string
  name: string
  summary: string
  links: string[]
}

/** Rows of fields; handles quoted fields with commas, newlines and "" escapes. */
function parseCsv(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ""
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const char = text[i]
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"'
        i++
      } else if (char === '"') quoted = false
      else field += char
    } else if (char === '"') quoted = true
    else if (char === ",") {
      row.push(field)
      field = ""
    } else if (char === "\n" || char === "\r") {
      if (char === "\r" && text[i + 1] === "\n") i++
      row.push(field)
      rows.push(row)
      row = []
      field = ""
    } else field += char
  }
  if (field || row.length) rows.push([...row, field])
  return rows
}

/** Markdown marks the notes use (`code`, **bold**) dropped, whitespace collapsed. */
const plain = (text: string) =>
  text
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/`([^`]*)`/g, "$1")
    .replace(/\s+/g, " ")
    .trim()

/** One sentence about a release, from its notes. The names are model names ("Claude
 *  Opus 5.5"). Notes either stand on their own ("Preview; GA May 7", "Qwen (Alibaba): 2.4T
 *  parameters") or carry on from the name ("the first Haiku model since ...", "(private
 *  beta) - the first GPT model ..."), and then the name goes in front. Long notes are cut
 *  to their first sentence, and to its first clause if that is still long. */
export function summarize(name: string, notes: string): string {
  const title = plain(name)
  const rest = plain(notes)
  let text: string
  if (!rest) text = title
  else if (/^[A-Z0-9"“]/.test(rest)) text = rest
  else if (/[.!?]$/.test(title) || rest.startsWith("("))
    text = `${title} ${rest}`
  else text = `${title}, ${rest.replace(/^[,:]\s*/, "")}`
  // Not at an initial ("U.S.", "N. Virginia").
  let sentence =
    text.match(/^(.*?)(?<!\b[A-Z])[.!?](?=\s+[A-Z"“(]|$)/)?.[1] ?? text
  if (sentence.length > 200) sentence = sentence.split(";")[0]
  return sentence.replace(/[\s,:;.-]+$/, "") + "."
}

const BOM = String.fromCharCode(0xfeff) // the CSV starts with one
const [header, ...rows] = parseCsv(
  csvText.startsWith(BOM) ? csvText.slice(1) : csvText
)
const column = (name: string) => header.indexOf(name)
const field = (row: string[], index: number) => row[index]?.trim() ?? ""
const [COMPANY, DATE, NAME, NOTES, TYPE] = [
  "Company name",
  "Release date",
  "Release name",
  "Notes",
  "Release type",
].map(column)
const LINKS = header.flatMap((name, i) => (/^Link \d+$/.test(name) ? [i] : []))

/** Every model release, newest first. (Without a Release type column, every row.) */
export const releases: Release[] = rows
  .filter((row) => field(row, COMPANY) && field(row, DATE))
  .filter((row) => TYPE < 0 || field(row, TYPE) === "Model release")
  .map((row) => ({
    company: field(row, COMPANY),
    date: field(row, DATE),
    name: plain(field(row, NAME)),
    summary: summarize(field(row, NAME), field(row, NOTES)),
    // Only plain web links become links.
    links: LINKS.map((i) => field(row, i)).filter((link) =>
      /^https?:\/\//.test(link)
    ),
  }))
  .sort((a, b) => b.date.localeCompare(a.date))

// The column order, left to right (names matched ignoring case). A company not listed
// here gets a column just before "miscellaneous", the smaller companies grouped in the
// CSV, which always comes last.
const COLUMN_ORDER = [
  "OpenAI",
  "Anthropic",
  "Google",
  "Meta",
  "xAI",
  "DeepSeek",
  "Mistral",
].map((name) => name.toLowerCase())
const GROUPED = "miscellaneous"

function columnRank(company: string) {
  const name = company.toLowerCase()
  if (name === GROUPED) return COLUMN_ORDER.length + 1
  const listed = COLUMN_ORDER.indexOf(name)
  return listed < 0 ? COLUMN_ORDER.length : listed
}

/** The companies with model releases, in COLUMN_ORDER: one timeline column each. */
export const companies = [...new Set(releases.map((r) => r.company))].sort(
  (a, b) => columnRank(a) - columnRank(b) || a.localeCompare(b)
)
