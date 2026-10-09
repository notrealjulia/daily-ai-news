# Experiment: Google AI This Week

The OpenAI weekly experiment (`../openai_weekly/`), pointed at Google's AI ecosystem.
`run.py` loads that pipeline and replaces only the target-specific settings: the
similarity queries, the entity lists, the judge and report prompts, the thresholds and the
output folder. See `../openai_weekly/README.md` for how the pipeline works and what each
output file contains.

```powershell
$env:AINEWS_BACKEND = "turso"
python experiments/google_weekly/run.py
python experiments/compare.py        # writes experiments/comparison.md
```

## Google-specific choices

- **Plain "Google" is an ambiguous name, not an entity.** Most articles mention Google for
  Search, ads, Android or antitrust. It only makes an article a candidate together with a
  similarity score of 0.45 or more, and the judge sees it flagged as ambiguous.
- **Strong entities:** DeepMind, Gemini, Gemma (including EmbeddingGemma and similar),
  NotebookLM, Veo, Imagen, Google AI Studio, "Google AI", Vertex AI, AI Overviews/AI Mode,
  the Alpha models (AlphaFold, AlphaEvolve and others), Nano Banana, SynthID, Lyria, Project
  Astra/Mariner, Google Labs, TPUs, and leaders and researchers (Hassabis, Pichai,
  Kavukcuoglu, Jeff Dean, Josh Woodward, Logan Kilpatrick, Shazeer, Vinyals, Shane Legg,
  Manyika, Pushmeet Kohli).
- **Ambiguous names:** Google, Bard, Astra (also OpenAI's GPT-6 Astra this week), Jules,
  Genie, Mariner, AI Studio, Gems, Ironwood, Antigravity.
- **The judge requires Google's own AI work** (models, AI products and features, research,
  TPUs, AI strategy and leadership). It rejects Google stories without an AI angle,
  passing mentions, and other things called Gemini, Astra, Genie or Jules.
- The semantic threshold is the same 0.50 as the OpenAI run, so the two compare directly.

## First run (7 days to 2026-10-07)

250 articles → 53 candidates → 22 accepted, plus 3 more found by the recall audit among
plain "Google" mentions. See `../comparison.md` for the retrieval comparison with OpenAI
and a manual check of the judge's precision.
