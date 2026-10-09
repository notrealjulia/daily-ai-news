"""OpenAI This Week: an experiment in finding a week's OpenAI news among stored articles.

Run from the project root. The last 7 days live in the hosted database, so:

    $env:AINEWS_BACKEND = "turso"          # PowerShell; read-only use
    python experiments/openai_weekly/run.py

Steps: load the week's articles and their extracted text -> chunk and embed them
(text-embedding-3-small) -> pick candidates by semantic similarity plus keyword/entity
matches, tuned for recall -> an LLM judge (the project's gpt-5.6-luna integration) keeps
or drops each candidate with a reason -> one more call writes the summary and timeline.

Everything it writes goes to experiments/openai_weekly/output/. Embeddings and judge
answers are cached in output/cache/ (gitignored) so retrieval can be re-tuned without
paying again; delete that folder to start fresh. The database is only ever read.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import openai
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so `ainews` imports when run as a plain script
from ainews import db, llm  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
CACHE = OUT / "cache"

# What this run is about. experiments/google_weekly/run.py reuses this pipeline by
# overriding these and the other upper-case settings below before calling main().
COMPANY = "OpenAI"
TITLE = "OpenAI This Week"

# Recall check: the judge also rates this many of the highest-scoring articles that were
# NOT candidates, plus every non-candidate that matched an ambiguous name (e.g. a plain
# "Google"). Any it accepts are misses. They never go into the report.
AUDIT_COUNT = 15

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_CHARS = 2000  # ~500 tokens: small enough that one OpenAI paragraph isn't drowned out
CHUNK_OVERLAP = 200

# What "OpenAI news" looks like, from a few angles. An article scores the best cosine
# similarity of any of its chunks against any of these.
QUERIES = [
    "OpenAI news: an announcement, product launch or company development at OpenAI",
    "OpenAI releases a new GPT model or updates ChatGPT",
    "Sam Altman and OpenAI leadership, funding, valuation, partnerships or lawsuits",
    "OpenAI research, safety policy, or deals with Microsoft and other companies",
]

# Recall-first candidate rule (see is_candidate): every strong-entity match goes to the
# judge, plus anything scoring this high on similarity alone. text-embedding-3-small
# cosines are low in absolute terms. Set from the first run's output/retrieval.csv (250
# articles): median 0.39, top 10% from 0.51; semantic-only articles at 0.45-0.50 were all
# unrelated, so 0.50 keeps the net wide without flooding the judge.
SEMANTIC_THRESHOLD = 0.50
WEAK_KEYWORD_THRESHOLD = 0.40

# Unambiguous OpenAI entities: any match makes an article a candidate.
STRONG_ENTITIES = {
    "OpenAI": r"\bOpen ?AI\b",
    "ChatGPT": r"\bChatGPT\b",
    "GPT model": r"(?i)\bGPT-?[3-9](\.\d+)?(o|-?turbo|-?mini|-?nano|-?pro)?\b",
    "gpt-oss": r"(?i)\bgpt-oss\b",
    "DALL-E": r"(?i)\bDALL[·-]?E\b",
    "Sam Altman": r"\bAltman\b",
    "Greg Brockman": r"\bBrockman\b",
    "Fidji Simo": r"\bFidji Simo\b",
    "Jakub Pachocki": r"\bPachocki\b",
}
# OpenAI names that are also ordinary words or other things: they only count together
# with some semantic similarity, and the judge sees them flagged as ambiguous.
WEAK_ENTITIES = {
    "Sora": r"\bSora\b",
    "Codex": r"\bCodex\b",
    "o-series model": r"\bo[134](-mini|-pro)?\b",
    "Whisper": r"\bWhisper\b",
    "Operator": r"\bOperator\b",
    "Atlas": r"\bAtlas\b",
    "Stargate": r"\bStargate\b",
}

JUDGE_VERSION = "j1"  # bump to re-ask the judge instead of reusing cached answers
JUDGE_INSTRUCTIONS = """\
You decide whether a news article belongs in a weekly digest called "OpenAI This Week".

Answer relevant = true only if OpenAI (the company, its products such as ChatGPT, Sora or
Codex, its models such as GPT-5, its research, business, people, legal matters or deals) is a
main subject of the article: the article reports something OpenAI did, announced, released,
or that happened to OpenAI.

Answer relevant = false when OpenAI is only mentioned in passing: listed among other AI
companies, used as a point of comparison, a benchmark entry, a passing quote, or background.
Also false when a matched name does not refer to OpenAI at all (e.g. "Atlas" or "Operator"
meaning something else, "o3" meaning ozone, "Sora" as a person's name).

Give a reason of at most 25 words that names what the article says about OpenAI, or why
the mention is incidental or ambiguous."""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["relevant", "reason"],
    "additionalProperties": False,
}

REPORT_INSTRUCTIONS = """\
You write "OpenAI This Week" from the articles below, all of which are about OpenAI.

summary: 3 to 5 sentences on the week's most important OpenAI developments, factual and
specific, using only what the articles say.

timeline: the week's meaningful milestones in chronological order, one entry per distinct
event (merge articles that report the same event). Each entry has the event's date
(YYYY-MM-DD, the earliest article date that reports it), one sentence describing it, and the
URLs of the articles that report it, copied exactly from the list. Skip commentary, opinion
pieces and minor mentions that are not events."""
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "milestone": {"type": "string"},
                    "source_urls": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["date", "milestone", "source_urls"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "timeline"],
    "additionalProperties": False,
}


# --- 1. Load ------------------------------------------------------------------


def load_articles(days: int) -> list[dict]:
    """The window's articles that have extracted text. Read-only: SELECTs only."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(db.TIMESTAMP_FORMAT)
    conn = db.connect_readonly()
    try:
        rows = conn.execute(
            """
            SELECT id, source, url, title, COALESCE(published_at, fetched_at) AS happened_at, body
            FROM articles
            WHERE body_status = 'ready' AND COALESCE(published_at, fetched_at) >= ?
            ORDER BY happened_at
            """,
            (since,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


# --- 2. Embed -----------------------------------------------------------------


def chunks(article: dict) -> list[str]:
    """The whole body in overlapping pieces (never truncated), each prefixed with the title."""
    body = " ".join(article["body"].split())
    step = CHUNK_CHARS - CHUNK_OVERLAP
    pieces = [body[i : i + CHUNK_CHARS] for i in range(0, max(len(body) - CHUNK_OVERLAP, 1), step)]
    return [f"{article['title']}\n\n{piece}" for piece in pieces]


def embed(texts: list[str], client: openai.OpenAI) -> list[list[float]]:
    """Embeddings for `texts`, from the cache when possible (keyed by model + text)."""
    path = CACHE / "embeddings.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    key = lambda text: hashlib.sha256(f"{EMBEDDING_MODEL}\n{text}".encode()).hexdigest()  # noqa: E731
    missing = list(dict.fromkeys(t for t in texts if key(t) not in cache))
    for i in range(0, len(missing), 100):
        batch = missing[i : i + 100]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for text, item in zip(batch, response.data):
            cache[key(text)] = [round(x, 6) for x in item.embedding]
        print(f"  embedded {min(i + 100, len(missing))}/{len(missing)} new chunks")
    if missing:
        path.write_text(json.dumps(cache), encoding="utf-8")
    return [cache[key(t)] for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# --- 3. Retrieve --------------------------------------------------------------


def entity_matches(text: str, entities: dict[str, str]) -> dict[str, dict]:
    """{entity: {"count": n, "example": "...context..."}} for every entity found."""
    found = {}
    for name, pattern in entities.items():
        hits = list(re.finditer(pattern, text))
        if hits:
            start = max(hits[0].start() - 60, 0)
            example = " ".join(text[start : hits[0].end() + 60].split())
            found[name] = {"count": len(hits), "example": f"...{example}..."}
    return found


def is_candidate(article: dict) -> list[str]:
    """Why the article is a candidate (empty list: it isn't)."""
    why = []
    if article["strong"]:
        why.append("strong entity")
    if article["score"] >= SEMANTIC_THRESHOLD:
        why.append(f"semantic >= {SEMANTIC_THRESHOLD}")
    if article["weak"] and article["score"] >= WEAK_KEYWORD_THRESHOLD:
        why.append(f"weak entity + semantic >= {WEAK_KEYWORD_THRESHOLD}")
    return why


def retrieve(articles: list[dict], client: openai.OpenAI) -> None:
    """Adds score, best chunk, entity matches and candidate reasons to every article."""
    article_chunks = [chunks(a) for a in articles]
    all_chunks = [c for cs in article_chunks for c in cs]
    print(f"Embedding {len(all_chunks)} chunks from {len(articles)} articles + {len(QUERIES)} queries")
    vectors = embed(all_chunks + QUERIES, client)
    query_vectors = vectors[len(all_chunks) :]
    position = 0
    for article, cs in zip(articles, article_chunks):
        scored = [
            (cosine(vectors[position + i], q), i, qi)
            for i in range(len(cs))
            for qi, q in enumerate(query_vectors)
        ]
        position += len(cs)
        article["score"], article["best_chunk"], article["best_query"] = max(scored)
        article["chunks"] = cs
        text = f"{article['title']}\n{article['body']}"
        article["strong"] = entity_matches(text, STRONG_ENTITIES)
        article["weak"] = entity_matches(text, WEAK_ENTITIES)
        article["candidate_reasons"] = is_candidate(article)


# --- 4. Judge -----------------------------------------------------------------


def judge_input(article: dict) -> str:
    matches = [f"- {name} (x{m['count']}): {m['example']}" for name, m in article["strong"].items()]
    matches += [f"- {name} (x{m['count']}, AMBIGUOUS NAME): {m['example']}" for name, m in article["weak"].items()]
    text = "\n".join(
        [
            f"Title: {article['title']}",
            f"Source: {article['source']}",
            f"Date: {article['happened_at'][:10]}",
            "Matched names:",
            *(matches or ["- none (found by semantic similarity only)"]),
            "",
            f"Most {COMPANY}-like passage:",
            article["chunks"][article["best_chunk"]],
            "",
            "Beginning of the article:",
            article["body"][:4000],
        ]
    )
    return text[:12000]


def judge(candidates: list[dict], model: llm.OpenAIStructuredLLM) -> None:
    path = CACHE / "judgments.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def decide(article: dict) -> dict:
        key = f"{article['id']}:{JUDGE_VERSION}:{model.model}"
        if key not in cache:
            try:
                cache[key] = model.generate(
                    instructions=JUDGE_INSTRUCTIONS,
                    input_text=judge_input(article),
                    schema_name="relevance",
                    schema=JUDGE_SCHEMA,
                )
            except llm.LLMError as e:  # recorded, not cached: retried on the next run
                return {"relevant": False, "reason": f"JUDGE ERROR: {e}"}
        return cache[key]

    print(f"Judging {len(candidates)} articles with {model.model}")
    with ThreadPoolExecutor(max_workers=6) as pool:
        for article, decision in zip(candidates, pool.map(decide, candidates)):
            article["relevant"], article["reason"] = decision["relevant"], decision["reason"]
    path.write_text(json.dumps(cache, indent=1), encoding="utf-8")


# --- 5. Report ----------------------------------------------------------------


def write_report_content(accepted: list[dict], model: llm.OpenAIStructuredLLM) -> dict:
    listing = "\n\n".join(
        f"[{a['happened_at'][:10]}] {a['title']} ({a['source']})\nURL: {a['url']}\n"
        f"Why it is about {COMPANY}: {a['reason']}\nExcerpt: {' '.join(a['body'][:2500].split())}"
        for a in accepted
    )
    report = model.generate(
        instructions=REPORT_INSTRUCTIONS,
        input_text=listing,
        schema_name="weekly_digest",
        schema=REPORT_SCHEMA,
    )
    known = {a["url"] for a in accepted}
    for item in report["timeline"]:
        invented = [u for u in item["source_urls"] if u not in known]
        item["source_urls"] = [u for u in item["source_urls"] if u in known]
        if invented:
            item["dropped_urls"] = invented  # not one of the accepted articles
    report["timeline"].sort(key=lambda item: item["date"])
    return report


# --- Output -------------------------------------------------------------------


def write_outputs(articles: list[dict], audit: list[dict], report: dict | None, days: int) -> None:
    by_score = sorted(articles, key=lambda a: a["score"], reverse=True)
    with open(OUT / "retrieval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["score", "candidate", "candidate_reasons", "strong_entities", "weak_entities",
                    "best_query", "date", "source", "title", "url"])  # fmt: skip
        for a in by_score:
            w.writerow([
                f"{a['score']:.3f}", bool(a["candidate_reasons"]), "; ".join(a["candidate_reasons"]),
                "; ".join(f"{k} x{v['count']}" for k, v in a["strong"].items()),
                "; ".join(f"{k} x{v['count']}" for k, v in a["weak"].items()),
                a["best_query"], a["happened_at"][:10], a["source"], a["title"], a["url"],
            ])  # fmt: skip

    candidates = [a for a in by_score if a["candidate_reasons"]]
    with open(OUT / "judgments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "relevant", "reason", "score", "candidate_reasons", "entity_examples", "title", "url"])
        for kind, group in (("candidate", candidates), ("audit", audit)):
            for a in group:
                examples = " | ".join(f"{k}: {v['example']}" for k, v in {**a["strong"], **a["weak"]}.items())
                w.writerow([kind, a["relevant"], a["reason"], f"{a['score']:.3f}", "; ".join(a["candidate_reasons"]),
                            examples, a["title"], a["url"]])  # fmt: skip

    accepted = sorted((a for a in candidates if a["relevant"]), key=lambda a: a["happened_at"])
    rejected = [a for a in candidates if not a["relevant"]]
    lines = [
        f"# {TITLE}",
        "",
        f"_Experiment run {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, last {days} days: "
        f"{len(articles)} articles, {len(candidates)} candidates, {len(accepted)} accepted._",
        "",
    ]
    if report:
        lines += ["## Summary", "", report["summary"], "", "## Timeline", ""]
        for item in report["timeline"]:
            sources = ", ".join(f"<{u}>" for u in item["source_urls"]) or "(no valid source)"
            lines.append(f"- **{item['date']}** {item['milestone']}  \n  Sources: {sources}")
            if item.get("dropped_urls"):
                lines.append(f"  _(dropped URLs not in the accepted set: {item['dropped_urls']})_")
    lines += ["", "## Accepted articles", ""]
    lines += [f"- {a['happened_at'][:10]} · {a['source']} · [{a['title']}]({a['url']}) "
              f"(score {a['score']:.3f}) — {a['reason']}" for a in accepted]  # fmt: skip
    lines += ["", "## Rejected candidates", ""]
    lines += [f"- [{a['title']}]({a['url']}) (score {a['score']:.3f}; "
              f"{', '.join(a['strong']) or ', '.join(a['weak']) or 'semantic only'}) — {a['reason']}"
              for a in rejected]  # fmt: skip
    lines += ["", f"## Recall audit: {len(audit)} non-candidates (the top {AUDIT_COUNT} by score, "
              "plus every ambiguous-name match), judged", ""]
    lines += [f"- **{'MISSED, relevant' if a['relevant'] else 'not relevant'}** [{a['title']}]({a['url']}) "
              f"(score {a['score']:.3f}; weak: {', '.join(a['weak']) or 'none'}) — {a['reason']}"
              for a in audit]  # fmt: skip
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = {
        "company": COMPANY, "title": TITLE, "days": days, "queries": QUERIES,
        "semantic_threshold": SEMANTIC_THRESHOLD, "weak_keyword_threshold": WEAK_KEYWORD_THRESHOLD,
        "strong_entities": list(STRONG_ENTITIES), "weak_entities": list(WEAK_ENTITIES),
        "judge_version": JUDGE_VERSION, "audit_count": AUDIT_COUNT,
    }  # fmt: skip
    (OUT / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{TITLE} (experiment)")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # article titles aren't ASCII

    print(f"Reading {'Turso' if db.uses_turso() else 'the local SQLite file'} (read-only)")
    articles = load_articles(args.days)
    if not articles:
        print("No articles with extracted text in that window.")
        return 1
    api_key = os.environ.get(llm.API_KEY_VARIABLE) or dotenv_values(llm.ENV_PATH).get(llm.API_KEY_VARIABLE)
    retrieve(articles, openai.OpenAI(api_key=api_key, timeout=llm.REQUEST_TIMEOUT_SECONDS))

    by_score = sorted(articles, key=lambda a: a["score"], reverse=True)
    candidates = [a for a in by_score if a["candidate_reasons"]]
    non_candidates = [a for a in by_score if not a["candidate_reasons"]]
    audit = non_candidates[:AUDIT_COUNT] + [a for a in non_candidates[AUDIT_COUNT:] if a["weak"]]
    model = llm.create()
    judge(candidates + audit, model)
    accepted = sorted((a for a in candidates if a["relevant"]), key=lambda a: a["happened_at"])

    print(f"\n{'score':>6}  {'judge':5}  {'entities':28}  title")
    for a in candidates:
        entities = ", ".join([*a["strong"], *(f"{w}?" for w in a["weak"])])[:28]
        print(f"{a['score']:6.3f}  {'yes' if a['relevant'] else 'no':5}  {entities:28}  {a['title'][:70]}")
        print(f"{'':15}{a['reason']}")

    missed = [a for a in audit if a["relevant"]]
    print(f"\nRecall audit: {len(missed)} of {len(audit)} non-candidates judged relevant")
    for a in missed:
        print(f"  MISSED {a['score']:.3f}  {a['title'][:80]}  ({a['reason']})")

    report = None
    if accepted:
        print(f"\nWriting the report from {len(accepted)} accepted articles")
        report = write_report_content(accepted, model)
        print(f"\n{report['summary']}\n")
        for item in report["timeline"]:
            print(f"  {item['date']}  {item['milestone']}  ({len(item['source_urls'])} sources)")
    write_outputs(articles, audit, report, args.days)
    print(f"\n{len(articles)} articles -> {len(candidates)} candidates -> {len(accepted)} accepted.")
    print(f"See {OUT.relative_to(ROOT)}/report.md, retrieval.csv, judgments.csv and settings.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
