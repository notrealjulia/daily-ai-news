"""Settings shared between the pipeline and the dashboard, in a module with no other
ainews imports (only the standard library), so the dashboard can read them without
loading the OpenAI SDK or any pipeline code.

The current digest prompt version, the other thing the dashboard needs, lives in
ainews.prompts instead, alongside every other prompt version; it is likewise import-free.
"""

from pathlib import Path

# The model is a proposal until it has been tried on real articles; it is stored with
# every enrichment, so results from different models can sit side by side.
DEFAULT_MODEL = "gpt-5.6-luna"

# Where `narrate` writes each category's narration: one fixed file per category, always
# overwritten. ainews.narrate and the dashboard both need this path; ainews.narrate can't
# be imported here or from the dashboard, since it imports the OpenAI SDK.
AUDIO_DIR = Path("audio")


def audio_slug(category: str) -> str:
    """Filename-safe form of a category name, e.g. "Product Release" -> "product-release"."""
    return category.lower().replace(" & ", "-").replace(" ", "-")


def audio_path(category: str) -> Path:
    return AUDIO_DIR / f"{audio_slug(category)}.mp3"
