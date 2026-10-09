"""Compare retrieval quality between the weekly-company experiments.

Reads each experiment's output/ (retrieval.csv, judgments.csv, settings.json) and writes
experiments/comparison.md. Run after both experiments:

    python experiments/compare.py

Definitions, per experiment:
- keyword retrieval (K): articles naming at least one strong entity.
- semantic retrieval (S): articles scoring >= that experiment's semantic threshold.
- relevant: the judge's yes, over everything it saw (candidates, plus a recall audit of the
  15 highest-scoring non-candidates and every non-candidate matching an ambiguous name).
  Articles the judge never saw count as not relevant, so recall figures are relative to
  what was found, not absolute.
"""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = ["openai_weekly", "google_weekly"]

# Filled in by hand after reading each experiment's accepted and rejected lists: the
# judge's own decisions can't measure its precision.
MANUAL_REVIEW = """Reviewed by hand on 2026-10-07: every accepted article (title, judge reason) and every
rejected candidate's title, for both experiments. "Wrong" means it fails the judge's own
rubric (the company's AI work must be a main subject); "borderline" could go either way.

**OpenAI (34 accepted):** about 29 clearly right, 2 borderline, 3 wrong, so judge
precision is roughly 85-90%.
- Wrong: *Opus 5.5 loves to tell you 'this matters'* (writing tells across many models),
  *Brian Chesky interview* (Airbnb CEO; ChatGPT's app store is an aside), *Hot Girl
  Hotline* (relationship-advice service; ChatGPT is context).
- Borderline: *Lawyer cites ChatGPT-invented fake witnesses* (ChatGPT is the subject, but
  OpenAI did nothing), *Google drops Gems for Skills* (mostly a Google story).
- No wrong rejections spotted among the 49 rejected candidates: Gemini, Mistral, Anthropic
  and Meta stories were all correctly turned down despite naming OpenAI.

**Google (22 accepted candidates + 3 audit finds):** about 18 clearly right, 3 borderline, 4
wrong, so judge precision is roughly 80-85%.
- Wrong: *Opus 5.5 loves to tell you 'this matters'* again (accepted by both judges for
  the same passing mention), *AutoSynthData* and *Satlyt* (third parties using Gemma; the
  rubric asks for Google's own work), *Internet Infrastructure Services Empower Deepfake
  Abuse* (audit: about Google's SSL certificates and ads, not its AI).
- Borderline: *Trump plan to combat AI risks* (Google is one signatory among many), *Google
  froze its open source bug bounty program* (audit: Google reacting to AI-generated reports,
  not shipping AI), *MCP for agent-to-agent comms* (audit: Google acknowledging flaws in its
  agent systems, a small part of the article).
- No wrong rejections spotted among the 31 rejected candidates.

**Where the judge goes wrong** is the same in both: it accepts articles where a product is
the topic or the tool but the company did nothing (ChatGPT misuse, others building on
Gemma), and it accepted one multi-model article for both companies. Tightening the rubric
on "used by others" and "one of several models" is the obvious next prompt change.
"""


def load(name: str) -> dict:
    out = HERE / name / "output"
    settings = json.loads((out / "settings.json").read_text(encoding="utf-8"))
    with open(out / "retrieval.csv", encoding="utf-8") as f:
        articles = list(csv.DictReader(f))
    with open(out / "judgments.csv", encoding="utf-8") as f:
        judged = {row["url"]: row for row in csv.DictReader(f)}
    for a in articles:
        a["score"] = float(a["score"])
        a["K"] = bool(a["strong_entities"])
        a["S"] = a["score"] >= settings["semantic_threshold"]
        a["W"] = bool(a["weak_entities"]) and a["score"] >= settings["weak_keyword_threshold"]
        verdict = judged.get(a["url"])
        a["judged"] = verdict["set"] if verdict else None
        a["relevant"] = bool(verdict) and verdict["relevant"] == "True"
        a["reason"] = verdict["reason"] if verdict else ""
    return {"name": name, "settings": settings, "articles": articles}


def ratio(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "–"


def section(exp: dict) -> list[str]:
    s, arts = exp["settings"], exp["articles"]
    rel = [a for a in arts if a["relevant"]]
    groups = {
        "Keyword (K), all": [a for a in arts if a["K"]],
        "Semantic (S), all": [a for a in arts if a["S"]],
        "Both K and S": [a for a in arts if a["K"] and a["S"]],
        "Keyword only (K, not S)": [a for a in arts if a["K"] and not a["S"]],
        "Semantic only (S, not K)": [a for a in arts if a["S"] and not a["K"]],
        "Ambiguous name + similarity only": [a for a in arts if a["W"] and not a["K"] and not a["S"]],
    }
    lines = [
        f"## {s['title']}",
        "",
        f"{len(arts)} articles · semantic threshold {s['semantic_threshold']} · ambiguous-name threshold "
        f"{s['weak_keyword_threshold']} · {len(rel)} judged relevant in total",
        "",
        "| Retrieval route | Articles | Precision (judged relevant) | Recall (of all relevant found) |",
        "|---|---|---|---|",
    ]
    for label, group in groups.items():
        hits = sum(a["relevant"] for a in group)
        lines.append(f"| {label} | {len(group)} | {ratio(hits, len(group))} | {ratio(hits, len(rel))} |")

    audit = [a for a in arts if a["judged"] == "audit"]
    missed = [a for a in audit if a["relevant"]]
    ranked = sorted(arts, key=lambda a: a["score"], reverse=True)
    lines += ["", "**Similarity ranking** (precision among the top-k by score, all judged):", ""]
    for k in (10, 20, 30):
        top = ranked[:k]
        unjudged = sum(a["judged"] is None for a in top)
        note = f", {unjudged} never judged" if unjudged else ""
        lines.append(f"- top {k}: {ratio(sum(a['relevant'] for a in top), k)}{note}")
    rel_scores = sorted(a["score"] for a in rel)
    rejected = [a for a in arts if a["judged"] and not a["relevant"]]
    lines += [
        f"- median score: relevant {rel_scores[len(rel_scores) // 2]:.3f}, judged not relevant "
        f"{sorted(a['score'] for a in rejected)[len(rejected) // 2]:.3f}; lowest relevant {rel_scores[0]:.3f}",
        "",
        f"**Recall audit:** {len(missed)} of {len(audit)} non-candidates judged relevant (the "
        f"{s['audit_count']} highest-scoring, plus every other ambiguous-name match; scores "
        f"{min(a['score'] for a in audit):.3f}–{max(a['score'] for a in audit):.3f}).",
        "",
    ]
    lines += [f"- MISSED {a['score']:.3f} · {a['title']} ({a['weak_entities'] or 'no keyword'}) — _{a['reason']}_"
              for a in missed] + ([""] if missed else [])  # fmt: skip

    def listing(title: str, items: list[dict], empty: str) -> None:
        lines.extend([f"**{title}** ({len(items)})", ""])
        lines.extend(f"- {a['score']:.3f} · {a['title']} — _{a['reason']}_" for a in items)
        lines.extend([empty] if not items else [])
        lines.append("")

    listing("Relevant articles found only by semantic search", [a for a in rel if a["S"] and not a["K"]],
            "- none: every relevant article also matched a keyword")  # fmt: skip
    listing("Relevant articles keywords found but semantic search ranked below the threshold",
            sorted((a for a in rel if a["K"] and not a["S"]), key=lambda a: a["score"]), "- none")  # fmt: skip
    fp = [a for a in arts if a["judged"] == "candidate" and not a["relevant"]]
    by_route = {
        "keyword only": sum(a["K"] and not a["S"] for a in fp),
        "semantic only": sum(a["S"] and not a["K"] for a in fp),
        "both": sum(a["K"] and a["S"] for a in fp),
        "ambiguous name": sum(a["W"] and not a["K"] and not a["S"] for a in fp),
    }
    lines += [
        f"**False positives** (candidates the judge rejected): {len(fp)} — "
        + ", ".join(f"{route} {n}" for route, n in by_route.items()),
        "",
    ]
    listing("Semantic-only false positives", [a for a in fp if a["S"] and not a["K"]], "- none")
    return lines


def main() -> None:
    exps = [load(name) for name in EXPERIMENTS]
    lines = ["# Weekly-company experiments: retrieval comparison", "", __doc__[__doc__.index("Definitions") :].strip(), ""]
    lines += ["| | " + " | ".join(e["settings"]["title"] for e in exps) + " |", "|---|" + "---|" * len(exps)]
    for label, fn in [
        ("Articles", lambda e: len(e["articles"])),
        ("Keyword hits (K)", lambda e: sum(a["K"] for a in e["articles"])),
        ("Semantic hits (S)", lambda e: sum(a["S"] for a in e["articles"])),
        ("Overlap K ∩ S", lambda e: sum(a["K"] and a["S"] for a in e["articles"])),
        ("Candidates judged", lambda e: sum(a["judged"] == "candidate" for a in e["articles"])),
        ("Judged relevant (incl. audit finds)", lambda e: sum(a["relevant"] for a in e["articles"])),
        ("Relevant found only semantically", lambda e: sum(a["relevant"] and a["S"] and not a["K"] for a in e["articles"])),
        ("Misses in recall audit", lambda e: sum(a["relevant"] and a["judged"] == "audit" for a in e["articles"])),
    ]:  # fmt: skip
        lines.append(f"| {label} | " + " | ".join(str(fn(e)) for e in exps) + " |")
    lines.append("")
    for e in exps:
        lines += section(e)
    lines += ["## Judge precision: manual spot check", "", MANUAL_REVIEW]
    (HERE / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((HERE / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
