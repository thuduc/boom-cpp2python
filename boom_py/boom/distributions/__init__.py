"""Probability distributions for BOOM."""

from .rng import GlobalRng, seed_rng

# Create global RNG instance
rng = GlobalRng()

__all__ = ["rng", "seed_rng", "GlobalRng"]