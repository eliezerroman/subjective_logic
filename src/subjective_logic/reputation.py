"""
Bayesian reputation systems.

Implements Chapter 16 (Josang, 2016, pp. 289-302): reputation score
computation is mostly a direct reuse of the Beta/Dirichlet expected-
probability machinery already implemented in binomial.py/multinomial.py
since Chapter 3 -- the book itself says as much (p. 291: "The Beta PDF
itself only provides the underlying statistical foundation, and is
otherwise not used in the reputation system"). This module adds only
what's genuinely new: time-decayed rating aggregation (Section 16.2),
the dynamic community base rate (Section 16.3.5), and reputation
representation as a point estimate / converted binomial rating
(Section 16.4.2).

Reputation score itself is NOT a separate function here -- use
BinomialOpinion.from_evidence(r, s, a).projected_probability (Eq. 16.1)
or MultinomialOpinion.from_evidence(ratings, a).projected_probabilities
(Eq. 16.2/16.15) directly.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .binomial import NON_INFORMATIVE_PRIOR_WEIGHT
from .multinomial import MultinomialOpinion


def aggregate_with_decay(previous: Mapping, new_ratings: Mapping, longevity: float) -> dict:
    """
    Time-decayed rating aggregation, Eq. 16.5 (p. 293):

        R_(tau+1) = longevity * R_tau + r_(tau+1)

    longevity = 0: ratings are completely forgotten after one period.
    longevity = 1: ratings are never forgotten (equivalent to simple
    cumulative addition, no decay).
    """
    if not (0.0 <= longevity <= 1.0):
        raise ValueError(f"longevity must be in [0, 1] (got {longevity!r}).")
    levels = set(previous) | set(new_ratings)
    return {level: longevity * previous.get(level, 0.0) + new_ratings.get(level, 0.0) for level in levels}


def aggregate_with_decay_n_periods(previous: Mapping, new_ratings: Mapping, longevity: float, periods: int) -> dict:
    """
    Eq. 16.6: generalisation of aggregate_with_decay over `periods`
    elapsed periods with no intermediate ratings, using longevity**periods.
    """
    if periods < 0:
        raise ValueError("periods must be non-negative.")
    levels = set(previous) | set(new_ratings)
    factor = longevity ** periods
    return {level: factor * previous.get(level, 0.0) + new_ratings.get(level, 0.0) for level in levels}


def convergence_value(constant_rating: Mapping, longevity: float) -> dict:
    """
    Convergence value of the accumulated rating vector under a constant
    per-period rating, Eq. 16.7-16.8 (pp. 293-294):

        R_infinity = e / (1 - longevity)

    Requires longevity < 1 (otherwise the geometric series diverges,
    matching Eq. 16.7's convergence condition).
    """
    if not (0.0 <= longevity < 1.0):
        raise ValueError(f"longevity must be in [0, 1) for convergence to be defined (got {longevity!r}).")
    return {level: value / (1.0 - longevity) for level, value in constant_rating.items()}


def individual_base_rate(
    evidence: Mapping, community_base_rate: Mapping, prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT
) -> dict:
    """
    Individual base rate for a service object, Eq. 16.9 (p. 294): the
    community base rate as prior, updated by the object's own
    accumulated evidence Q (from Sections 16.3.2-16.3.4: total history,
    a sliding window, or a high-longevity aggregate -- any of those
    just produce the `evidence` argument here).

    This is mathematically identical in form to the reputation score
    itself (Eq. 16.2): both are expected-probability projections of an
    evidence vector against a base rate. Reuses
    MultinomialOpinion.from_evidence directly rather than duplicating
    the formula.
    """
    return MultinomialOpinion.from_evidence(evidence, community_base_rate, prior_weight).projected_probabilities


def community_base_rate(
    all_service_ratings: Sequence[Mapping],
    default_base_rate: Mapping = None,
    prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT,
) -> dict:
    """
    Dynamic community base rate, Definition 16.1 / Eq. 16.13-16.14 (p.
    296): aggregate every service object's rating vector (Eq. 16.13),
    then compute that aggregate's own reputation score (Eq. 16.2) --
    the result becomes next period's base rate for every object in the
    community, including new arrivals (solving the cold-start problem).

    Args:
        all_service_ratings: one rating-vector dict per service object.
        default_base_rate: prior for the aggregate itself; defaults to
            uniform over whatever levels appear in the data.
    """
    levels = set()
    for ratings in all_service_ratings:
        levels.update(ratings.keys())
    if not levels:
        raise ValueError("all_service_ratings must contain at least one rating level.")

    aggregate = {level: sum(ratings.get(level, 0.0) for ratings in all_service_ratings) for level in levels}
    if default_base_rate is None:
        default_base_rate = {level: 1.0 / len(levels) for level in levels}

    return MultinomialOpinion.from_evidence(aggregate, default_base_rate, prior_weight).projected_probabilities


def point_estimate(score: Mapping, ordered_levels: Sequence) -> float:
    """
    Point estimate reputation score, Eq. 16.18 (p. 298): collapses a
    multinomial score distribution to a single value in [0, 1], using
    evenly spaced point values nu(Li) = (i-1)/(k-1) for k ordered levels.

    Note (p. 298): this necessarily discards information -- e.g. ten
    "average" ratings and five "bad" plus five "excellent" ratings can
    give the SAME point estimate (0.5) despite being very different
    distributions (Figure 16.2's own example, validated in tests).
    """
    levels = list(ordered_levels)
    k = len(levels)
    if k < 2:
        raise ValueError("point_estimate requires at least 2 ordered levels.")
    return sum((i / (k - 1)) * score[level] for i, level in enumerate(levels))


def multinomial_to_binomial(
    ratings: Mapping, ordered_levels: Sequence, base_rate: Mapping, prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT
) -> tuple:
    """
    Converts multinomial ratings to equivalent binomial evidence (r, s),
    Eq. 16.19 (p. 299), via the point estimate: r is the point-estimate-
    weighted share of the total ratings, s is the rest.
    """
    score = MultinomialOpinion.from_evidence(ratings, base_rate, prior_weight).projected_probabilities
    sigma = point_estimate(score, ordered_levels)
    total = sum(ratings.values())
    r = sigma * total
    s = total - r
    return r, s