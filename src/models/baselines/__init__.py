"""Canonical paper-baseline model adapters."""

from .dice import PaperGCNDICE
from .lightgcn import PaperLightGCN

__all__ = ["PaperGCNDICE", "PaperLightGCN"]
