"""
Shared helper functions for the experiment notebooks (Phases 0-6).

These are NOT part of the subjective_logic library itself. They belong
to the experimental/methodological layer built on top of it: ways of
checking whether an opinion's stated uncertainty is honest relative to
a known ground truth. Kept here, separate from src/subjective_logic/,
so that library code (validated against the book) and research-specific
methodology (validated against these experiments) don't get mixed up.

Both functions were introduced and validated in Phase 0
(notebooks/01_phase0_synthetic.ipynb, Experiments 0.1 and 0.3).
"""

from __future__ import annotations

import math

from scipy import stats


def credible_interval(r_eq: float, s_eq: float, base_rate: float = 0.5, level: float = 0.90,
                       prior_weight: float = 2.0) -> tuple:
    """
    Central credible interval of the Beta PDF equivalent to an opinion
    built from evidence (r_eq, s_eq) (Josang 2016, Definition 3.3 /
    Eq. 3.11): alpha = r_eq + base_rate*W, beta = s_eq + (1-base_rate)*W.

    Use opinion.to_evidence() to get (r_eq, s_eq) from any BinomialOpinion,
    including one that came from a fusion/deduction/discount chain rather
    than directly from from_evidence().

    Args:
        r_eq, s_eq: equivalent Beta evidence counts (e.g. from
            BinomialOpinion.to_evidence()).
        base_rate: a, the opinion's base rate (prior).
        level: credible mass to enclose (e.g. 0.90 for a 90% interval).
        prior_weight: W, the non-informative prior weight (2.0 in the
            book; only change this if you also changed it when building
            the opinion).

    Returns:
        (lower, upper): the interval bounds, in [0, 1].
    """
    tail = (1.0 - level) / 2.0
    alpha = r_eq + base_rate * prior_weight
    beta = s_eq + (1.0 - base_rate) * prior_weight
    return stats.beta.ppf(tail, alpha, beta), stats.beta.ppf(1.0 - tail, alpha, beta)


def standardized_sq_error(opinion, p_true: float) -> float:
    """
    Squared error standardized by the opinion's own claimed standard
    deviation: ((P - p_true) / sd)^2, where sd = sqrt(opinion.variance)
    (Josang 2016, Eq. 3.10 for BinomialOpinion, Eq. 3.13/3.18 for
    MultinomialOpinion -- both classes expose .variance the same way).

    Averaged over many runs (MSSE, mean standardized squared error), this
    is a calibration check for the opinion's UNCERTAINTY, not just its
    point estimate: MSSE around 1 means the stated uncertainty matches
    the actual error typically made; MSSE > 1 means the opinion is
    overconfident (understates its own error); MSSE < 1 means it is
    conservative (overstates its own error).

    Requires a known ground truth p_true, so this is only usable in
    controlled/synthetic settings (Phase 0) or on labelled evaluation
    data where the "correct" probability is known or assumed (e.g.
    p_true = 1.0 for the true class in a classification dataset).

    Args:
        opinion: any opinion object exposing .projected_probability
            (or .projected_probabilities[label] passed as a scalar
            through a wrapper) and .variance.
        p_true: the known/assumed true probability to compare against.
    """
    return ((opinion.projected_probability - p_true) / math.sqrt(opinion.variance)) ** 2