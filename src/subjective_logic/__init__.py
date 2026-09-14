"""Subjective Logic for Agentic AI: a Python implementation of Josang's
Subjective Logic formalism, applied to measuring decision degradation
in agentic AI systems under resource constraints."""

from .binomial import BinomialOpinion, NON_INFORMATIVE_PRIOR_WEIGHT
from .multinomial import MultinomialOpinion

__all__ = ["BinomialOpinion", "MultinomialOpinion", "NON_INFORMATIVE_PRIOR_WEIGHT"]