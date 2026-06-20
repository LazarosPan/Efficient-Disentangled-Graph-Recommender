"""Central EDGRec naming aliases and legacy identity helpers."""

from __future__ import annotations

EDGREC_DISPLAY_NAME = "EDGRec"
EDGREC_PUBLIC_PRESET = "edgrec"
_LEGACY_METHOD_PARTS = ("u", "ca", "gnn")

# Legacy tokens are retained only for checkpoint, SQLite, MLflow, and Optuna
# compatibility. Do not use them for thesis-facing display.
EDGREC_LEGACY_PRESET = "".join(_LEGACY_METHOD_PARTS)
EDGREC_PUBLIC_PREFIX = "edgrec"
EDGREC_LEGACY_PREFIX = EDGREC_LEGACY_PRESET


def is_edgrec_token(value: object) -> bool:
    """Return whether a token names EDGRec or its legacy storage alias."""
    return str(value).lower() in {EDGREC_PUBLIC_PRESET, EDGREC_LEGACY_PRESET}


def canonical_preset_for_identity(preset: str | None) -> str | None:
    """Return the checkpoint-hash preset token for a public or legacy preset."""
    if preset is None:
        return None
    return EDGREC_LEGACY_PRESET if is_edgrec_token(preset) else preset


def public_preset_name(preset: str | None) -> str | None:
    """Return the public preset token for display and new CLI examples."""
    if preset is None:
        return None
    return EDGREC_PUBLIC_PRESET if is_edgrec_token(preset) else preset


def display_method_label(value: object | None) -> str:
    """Return a thesis-facing label for a preset, profile, or method token."""
    if value is None:
        return "-"
    text = str(value)
    if is_edgrec_token(text):
        return EDGREC_DISPLAY_NAME
    return text


def _swap_method_token(value: str, *, source: str, target: str) -> str:
    """Replace the method token inside a persisted identifier."""
    return value.replace(source, target)


def legacy_method_identifier(value: str | None) -> str | None:
    """Return the legacy identifier used by existing persisted experiment data."""
    if value is None:
        return None
    return _swap_method_token(value, source=EDGREC_PUBLIC_PREFIX, target=EDGREC_LEGACY_PREFIX)


def public_method_identifier(value: str | None) -> str | None:
    """Return the public EDGRec identifier for a persisted legacy identifier."""
    if value is None:
        return None
    return _swap_method_token(value, source=EDGREC_LEGACY_PREFIX, target=EDGREC_PUBLIC_PREFIX)


def method_identifier_aliases(value: str) -> tuple[str, ...]:
    """Return public and legacy aliases for a method-scoped identifier."""
    return tuple(
        dict.fromkeys(
            (
                value,
                public_method_identifier(value) or value,
                legacy_method_identifier(value) or value,
            ),
        ),
    )
