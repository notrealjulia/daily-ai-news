"""The two defaults the read-only dashboard needs, in a module with no imports.

The dashboard shows the digests written with the current digest prompt and the default
model. llm.py and digest.py import the OpenAI SDK, which the dashboard must never load,
so these values live here; llm.py and digest.py import them (as llm.DEFAULT_MODEL and
digest.DIGEST_PROMPT_VERSION).
"""

# The model is a proposal until it has been tried on real articles; it is stored with
# every enrichment, so results from different models can sit side by side.
DEFAULT_MODEL = "gpt-5.6-luna"

# Bump when the digest prompt, its schema or its input format change.
DIGEST_PROMPT_VERSION = "v2"  # v2: synthesis only; counts are context, not text to repeat
