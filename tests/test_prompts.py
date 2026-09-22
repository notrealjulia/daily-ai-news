"""Every stage is wired to its own prompt in ainews.prompts, not another stage's.

ainews.prompts holds the actual wording (see its own module docstring); the point of
these tests is only that enrich, stories and digest each import the *right* constant
from it, not what that constant says. A stage importing the wrong prompt would otherwise
go unnoticed: nothing else in the suite compares instructions text across stages.
"""

from ainews import digest, enrich, prompts, stories


def test_enrich_uses_its_own_prompt():
    assert enrich.INSTRUCTIONS is prompts.ENRICH_INSTRUCTIONS
    assert enrich.PROMPT_VERSION == prompts.ENRICH_PROMPT_VERSION
    assert enrich.CATEGORY_LIST is prompts.CATEGORY_LIST


def test_cluster_uses_its_own_prompts():
    assert stories.GROUPING_INSTRUCTIONS is prompts.GROUPING_INSTRUCTIONS
    assert stories.COMBINE_INSTRUCTIONS is prompts.COMBINE_INSTRUCTIONS
    assert stories.STORY_PROMPT_VERSION == prompts.STORY_PROMPT_VERSION


def test_digest_uses_its_own_prompt():
    assert digest.INSTRUCTIONS is prompts.DIGEST_INSTRUCTIONS
    assert digest.DIGEST_PROMPT_VERSION == prompts.DIGEST_PROMPT_VERSION


def test_the_three_prompt_versions_are_independent():
    # Not a hard requirement, but three unrelated version counters sharing a value by
    # accident (e.g. a copy-paste of the wrong constant) is the exact mistake worth
    # catching here.
    versions = [prompts.ENRICH_PROMPT_VERSION, prompts.STORY_PROMPT_VERSION, prompts.DIGEST_PROMPT_VERSION]
    assert len(set(versions)) == len(versions)


def test_prompts_module_has_no_ainews_imports():
    # So it stays safe for the read-only dashboard to import (no SQL, no OpenAI SDK,
    # no pipeline code), the same guarantee defaults.py makes.
    assert not prompts.__dict__.get("db") and not prompts.__dict__.get("llm")
    import ast
    from pathlib import Path

    source = Path(prompts.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "ainews" and not (node.module or "").startswith("ainews.")
        elif isinstance(node, ast.Import):
            assert not any(alias.name.startswith("ainews") for alias in node.names)
