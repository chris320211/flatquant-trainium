"""
Load trainium-model-translation skill assets from the repo (SKILL.md, reference/*.md, scripts).
Used by Trainium-phase LangGraph nodes.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SKILL_DIR = _REPO_ROOT / ".claude" / "skills" / "trainium-model-translation"
_SKILL_MD = _SKILL_DIR / "SKILL.md"
_LEGACY_FLAT_SKILL = _REPO_ROOT / ".claude" / "skills" / "trainium-model-translation.md"
_REF_SCAFFOLDING = _SKILL_DIR / "reference" / "scaffolding_integration.md"
_REF_WEIGHT = _SKILL_DIR / "reference" / "weight_mapping.md"
_REF_VLM = _SKILL_DIR / "reference" / "vlm_translation.md"
_BLOCK_TESTING_UTILS = _SKILL_DIR / "scripts" / "block_testing_utils.py"

_DEFAULT_EXCERPT = 14_000


def skill_directory() -> Path:
    return _SKILL_DIR


def load_skill_markdown() -> str:
    """Full SKILL.md plus bundled resource paths (same resolution as nxdi_port)."""
    for path in (_SKILL_MD, _LEGACY_FLAT_SKILL):
        try:
            body = path.read_text()
            break
        except OSError:
            pass
    else:
        return (
            f"(Skill not found. Expected {_SKILL_MD} or legacy {_LEGACY_FLAT_SKILL}.)"
        )

    extra = (
        "\n\n--- Bundled resources (same skill directory) ---\n"
        "- reference/vlm_translation.md\n"
        "- reference/scaffolding_integration.md\n"
        "- reference/weight_mapping.md\n"
        "- scripts/block_testing_utils.py\n"
    )
    return body + extra


def _read_bounded(path: Path, max_chars: int) -> str:
    if not path.exists():
        return f"(Missing: {path})"
    text = path.read_text()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [truncated, {len(text)} total chars] ...\n"


def excerpt_scaffolding(max_chars: int = _DEFAULT_EXCERPT) -> str:
    return _read_bounded(_REF_SCAFFOLDING, max_chars)


def excerpt_weight_mapping(max_chars: int = _DEFAULT_EXCERPT) -> str:
    return _read_bounded(_REF_WEIGHT, max_chars)


def excerpt_vlm(max_chars: int = 10_000) -> str:
    return _read_bounded(_REF_VLM, max_chars)


def load_block_testing_utils() -> str:
    if not _BLOCK_TESTING_UTILS.exists():
        return f"# Missing {_BLOCK_TESTING_UTILS}\n"
    return _BLOCK_TESTING_UTILS.read_text()


def full_reference_scaffolding() -> str:
    """Entire Phase 3 guide (small file in repo)."""
    if not _REF_SCAFFOLDING.exists():
        return f"(Missing: {_REF_SCAFFOLDING})"
    return _REF_SCAFFOLDING.read_text()


def full_reference_weight_mapping() -> str:
    """Entire Phase 4 guide (small file in repo)."""
    if not _REF_WEIGHT.exists():
        return f"(Missing: {_REF_WEIGHT})"
    return _REF_WEIGHT.read_text()


def is_likely_vlm(modeling_source: str, model_config: dict | None) -> bool:
    """Heuristic: multimodal models need vlm_translation.md first."""
    src = (modeling_source or "").lower()
    hints = (
        "pixel_values",
        "image_grid_thw",
        "vision_tower",
        "visual",
        "image_embeds",
        "get_image_features",
    )
    if any(h in src for h in hints):
        return True
    cfg = model_config or {}
    for key in ("vision_config", "image_size", "num_channels", "image_token_index"):
        if key in cfg and cfg[key] is not None:
            return True
    return False
