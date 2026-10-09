"""Google AI This Week: the OpenAI weekly experiment, pointed at Google's AI ecosystem.

Same pipeline (experiments/openai_weekly/run.py): only the target-specific settings below
differ. Run from the project root:

    $env:AINEWS_BACKEND = "turso"
    python experiments/google_weekly/run.py

Output goes to experiments/google_weekly/output/ (caches in output/cache/, gitignored).
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("weekly_pipeline", HERE.parent / "openai_weekly" / "run.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)

pipeline.OUT = HERE / "output"
pipeline.CACHE = pipeline.OUT / "cache"
pipeline.COMPANY = "Google"
pipeline.TITLE = "Google AI This Week"

pipeline.QUERIES = [
    "Google AI news: an announcement, product launch or development in Google's or Google DeepMind's AI work",
    "Google releases or updates a Gemini or Gemma model, or AI features in Search, Android or Workspace",
    "Google DeepMind research and scientific breakthroughs such as AlphaFold, robotics or AI safety",
    "Sundar Pichai and Demis Hassabis on Google's AI strategy, investments, chips, partnerships or regulation",
]

# Set from this run's first retrieval-only pass (see README.md).
pipeline.SEMANTIC_THRESHOLD = 0.50
pipeline.WEAK_KEYWORD_THRESHOLD = 0.45

# Names that point at Google's AI work specifically. Plain "Google" is deliberately NOT
# here: most articles mention Google for Search, ads, Android or antitrust, not AI.
pipeline.STRONG_ENTITIES = {
    "DeepMind": r"\bDeepMind\b",
    "Gemini": r"\bGemini\b",
    "Gemma": r"Gemma\b",  # also EmbeddingGemma, CodeGemma, MedGemma
    "NotebookLM": r"\bNotebookLM\b",
    "Veo": r"\bVeo\b",
    "Imagen": r"\bImagen\b",
    "Google AI Studio": r"\bGoogle AI Studio\b",
    "Google AI": r"\bGoogle AI\b",
    "Vertex AI": r"\bVertex AI\b",
    "AI Overviews / AI Mode": r"\bAI (Overviews|Mode)\b",
    "Alpha models": r"\bAlpha(Fold|Evolve|Go|Zero|Proteo|Genome|Geometry|Dev|Code|Star)\b",
    "Nano Banana": r"(?i)\bnano banana\b",
    "SynthID": r"\bSynthID\b",
    "Lyria": r"\bLyria\b",
    "Project Astra / Mariner": r"\bProject (Astra|Mariner)\b",
    "Google Labs": r"\bGoogle Labs\b",
    "TPU": r"\bTPUs?\b",
    "Demis Hassabis": r"\bHassabis\b",
    "Sundar Pichai": r"\bPichai\b",
    "Koray Kavukcuoglu": r"\bKavukcuoglu\b",
    "Jeff Dean": r"\bJeff Dean\b",
    "Josh Woodward": r"\bJosh Woodward\b",
    "Logan Kilpatrick": r"\bLogan Kilpatrick\b",
    "Noam Shazeer": r"\bShazeer\b",
    "Oriol Vinyals": r"\bVinyals\b",
    "Shane Legg": r"\bShane Legg\b",
    "James Manyika": r"\bManyika\b",
    "Pushmeet Kohli": r"\bPushmeet Kohli\b",
}
# Count only with some semantic similarity, and flagged as ambiguous to the judge.
pipeline.WEAK_ENTITIES = {
    "Google": r"\bGoogle\b",
    "Bard": r"\bBard\b",
    "Astra": r"\bAstra\b",  # also OpenAI's GPT-6 Astra in this week's news
    "Jules": r"\bJules\b",
    "Genie": r"\bGenie\b",
    "Mariner": r"\bMariner\b",
    "AI Studio": r"\bAI Studio\b",
    "Gems": r"\bGems\b",
    "Ironwood": r"\bIronwood\b",
    "Antigravity": r"\bAntigravity\b",
}

pipeline.JUDGE_VERSION = "g1"
pipeline.JUDGE_INSTRUCTIONS = """\
You decide whether a news article belongs in a weekly digest called "Google AI This Week".

Answer relevant = true only if Google's AI work is a main subject of the article: something
Google, Google DeepMind or Google Labs did, announced, released or decided about AI: models
(Gemini, Gemma, Veo, Imagen and others), AI products and AI features (in Search, Android,
Workspace, Cloud, NotebookLM, AI Studio), AI research (AlphaFold and others), AI chips (TPUs),
or AI strategy, investments, partnerships, policy and leadership.

Answer relevant = false when:
- Google is mentioned but the story is not about its AI work (search ads, antitrust, Android
  or hardware without an AI angle, Google as a distribution channel or employer of record).
- Google's AI is only mentioned in passing: listed among other AI companies, a benchmark or
  comparison point, background, or a passing quote.
- A matched name does not refer to Google's AI at all (e.g. "Gemini" the crypto exchange or
  zodiac sign, "Astra" meaning another company's model, "Genie" or "Jules" meaning something else).

Give a reason of at most 25 words that names what the article says about Google's AI work,
or why the mention is incidental or unrelated."""
pipeline.REPORT_INSTRUCTIONS = """\
You write "Google AI This Week" from the articles below, all of which are about Google's AI
work (Google, Google DeepMind, Google Labs).

summary: 3 to 5 sentences on the week's most important Google AI developments, factual and
specific, using only what the articles say.

timeline: the week's meaningful milestones in chronological order, one entry per distinct
event (merge articles that report the same event). Each entry has the event's date
(YYYY-MM-DD, the earliest article date that reports it), one sentence describing it, and the
URLs of the articles that report it, copied exactly from the list. Skip commentary, opinion
pieces and minor mentions that are not events."""

if __name__ == "__main__":
    raise SystemExit(pipeline.main())
