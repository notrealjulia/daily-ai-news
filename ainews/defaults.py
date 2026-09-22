"""The default model, in a module with no imports.

llm.py imports the OpenAI SDK, which the dashboard must never load, so this value lives
here instead; llm.py imports it (as llm.DEFAULT_MODEL). The dashboard's other default,
the current digest prompt version, lives in ainews.prompts alongside every other prompt
version, which is likewise import-free and safe for the dashboard to read.
"""

# The model is a proposal until it has been tried on real articles; it is stored with
# every enrichment, so results from different models can sit side by side.
DEFAULT_MODEL = "gpt-5.6-luna"
