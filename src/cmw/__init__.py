"""ViabilityGrid: a deterministic testbed for cognitive primitives.

The distribution version is the semantic version recorded in every run
manifest; replay compares it across runs.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__: str = version("cognitive-miniworld")
