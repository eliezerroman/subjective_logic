"""Subjective Logic for Agentic AI: a Python implementation of Josang's
Subjective Logic formalism, applied to measuring decision degradation
in agentic AI systems under resource constraints."""

from .binomial import BinomialOpinion, NON_INFORMATIVE_PRIOR_WEIGHT
from .domain import base_rate_of_value, composite_set, hyperdomain, relative_base_rate
from .hyperopinion import HyperOpinion
from .multinomial import MultinomialOpinion

__all__ = [
    "BinomialOpinion",
    "MultinomialOpinion",
    "HyperOpinion",
    "NON_INFORMATIVE_PRIOR_WEIGHT",
    "hyperdomain",
    "composite_set",
    "base_rate_of_value",
    "relative_base_rate",
]