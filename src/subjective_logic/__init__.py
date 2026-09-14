"""Subjective Logic for Agentic AI: a Python implementation of Josang's
Subjective Logic formalism, applied to measuring decision degradation
in agentic AI systems under resource constraints."""

from .binomial import BinomialOpinion, NON_INFORMATIVE_PRIOR_WEIGHT

__all__ = ["BinomialOpinion", "NON_INFORMATIVE_PRIOR_WEIGHT"]