# Experiment: OpenAI This Week

A local prototype for evaluating retrieval quality, not production code. Nothing in
`ainews/`, the web dashboard, the database schema or the GitHub Actions workflows uses it.

## Run

From the project root (PowerShell). The last 7 days of articles are in the hosted
database, which is only read:

```powershell
$env:AINEWS_BACKEND = "turso"
python experiments/openai_weekly/run.py          # --days N for another window
```

The OpenAI key is read like everywhere else in the project: the real environment first,
then `.env`. (On this machine a stale Windows user-level `OPENAI_API_KEY` shadowed the
working `.env` key; `Remove-Item Env:OPENAI_API_KEY` in the terminal fixes that for the
session.)

## What it does

1. **Load**: every article from the window with extracted text (`body_status = 'ready'`).
2. **Embed**: each body in ~2000-character overlapping chunks (never truncated), each
   prefixed with its title, with `text-embedding-3-small`. An article's score is its best
   chunk's cosine similarity to any of four "OpenAI news" queries.
3. **Retrieve** (recall first): an article is a candidate if it names an unambiguous OpenAI
   entity (OpenAI, ChatGPT, GPT models, gpt-oss, DALL-E, Altman, Brockman, Simo,
   Pachocki), or scores ≥ 0.50, or names an ambiguous one (Sora, Codex, o-series, Whisper,
   Operator, Atlas, Stargate) and scores ≥ 0.40.
4. **Judge**: `gpt-5.6-luna` (via `ainews.llm`) answers relevant yes/no with a short reason
   for each candidate, seeing the matched names (ambiguous ones flagged), the most
   OpenAI-like chunk, and the start of the article.
5. **Report**: one more call writes a summary and a dated timeline from the accepted
   articles. Timeline URLs not in the accepted set are dropped (and noted).
6. **Recall audit**: the judge also rates the 15 highest-scoring non-candidates and every
   non-candidate that matched an ambiguous name. Any it accepts are misses; they're
   reported, never added to the digest.

The same pipeline powers `../google_weekly/`, which only overrides the target-specific
settings (`COMPANY`, `TITLE`, queries, entities, prompts, thresholds). `../compare.py`
compares the two runs.

## Output (`output/`)

| File | Contents |
|---|---|
| `report.md` | Summary, timeline with source URLs, accepted articles with reasons, rejected candidates with reasons, and the recall audit with verdicts |
| `retrieval.csv` | Every article: score, best-matching query, strong/weak entity counts, candidate yes/no and why |
| `judgments.csv` | Every judged article (`set` = candidate or audit): decision, reason, score, and the context around each entity match |
| `settings.json` | The run's thresholds, queries and entity names, for `../compare.py` |
| `cache/` | Embeddings and judge answers, so re-runs are free (gitignored; delete to start over) |

Tuning knobs are constants at the top of `run.py`: `QUERIES`, the two thresholds, the
entity lists, and the judge prompt. Bump `JUDGE_VERSION` after changing the judge prompt.

## First run (7 days to 2026-10-07)

- 250 articles → 83 candidates → 34 accepted → 12 timeline milestones; no judge errors.
- Every accepted article had a strong entity match; semantic similarity alone found no
  extra OpenAI story (its best non-entity hits, 0.45–0.50, were all unrelated). Similarity
  was still useful for ranking: accepted articles cluster at the top of `retrieval.csv`.
- "OpenAI" alone is a weak signal: it appears in about a third of the week's articles,
  mostly in passing. The judge rejected these consistently (benchmarks, competitor lists,
  "unlike OpenAI…").
- Ambiguous names behaved as expected: every Atlas, Operator and Codex-only match was
  something else (e.g. "Sam Altman" matching Jack Altman was correctly rejected).
- Borderline accepts worth reviewing: articles where ChatGPT is the subject but OpenAI
  did nothing new (a lawyer citing ChatGPT-invented witnesses, relationship advice).
- Cost: about 640 embedded chunks (~250k tokens) plus 84 small LLM calls.
