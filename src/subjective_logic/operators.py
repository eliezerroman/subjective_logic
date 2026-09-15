"""
Addition, subtraction and complement operators for binomial opinions.

Implements Chapter 6 (Josang, 2016, pp. 95-100): the belief-space
analogues of set union, set difference and set complement.

Important distinction from Chapter 12 (Belief Fusion, not yet
implemented): these operators combine opinions about DIFFERENT
(disjoint) subsets of the same domain into an opinion about their union
or difference. They are NOT for combining several agents' opinions
about the SAME variable -- that is cumulative/averaging fusion, a
fundamentally different operation working in evidence space rather than
belief space (p. 96): "The cumulative fusion operator is based on
addition of evidence in the evidence space, whereas the addition
operator is based on addition of belief mass in the belief space."
"""

from __future__ import annotations

from .binomial import BinomialOpinion

_TOLERANCE = 1e-9


def add(opinion_x1: BinomialOpinion, opinion_x2: BinomialOpinion) -> BinomialOpinion:
    """
    Addition of opinions, Definition 6.1 / Eq. 6.1 (p. 95-96): given
    opinions about two disjoint values x1 and x2 of the same domain,
    computes the opinion about their union x1 U x2.

    Preserves addition of projected probabilities (Eq. 6.2):
    P(x1Ux2) = P(x1) + P(x2).

    Note (p. 97): this generates VAGUE belief mass on the union from the
    SHARP belief masses of x1 and x2 -- the resulting belief no longer
    discriminates between x1 and x2 (see module docstring).

    Raises:
        ValueError: if a_x1 + a_x2 == 0 (division by zero in Eq. 6.1).
    """
    base_rate_sum = opinion_x1.base_rate + opinion_x2.base_rate
    if base_rate_sum == 0:
        raise ValueError("Cannot add opinions whose base rates sum to zero (division by zero in Eq. 6.1).")

    belief = opinion_x1.belief + opinion_x2.belief
    disbelief = (
        opinion_x1.base_rate * (opinion_x1.disbelief - opinion_x2.belief)
        + opinion_x2.base_rate * (opinion_x2.disbelief - opinion_x1.belief)
    ) / base_rate_sum
    uncertainty = (
        opinion_x1.base_rate * opinion_x1.uncertainty + opinion_x2.base_rate * opinion_x2.uncertainty
    ) / base_rate_sum
    base_rate = base_rate_sum

    return BinomialOpinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty, base_rate=base_rate)


def subtract(opinion_union: BinomialOpinion, opinion_x2: BinomialOpinion) -> BinomialOpinion:
    """
    Subtraction of opinions, Definition 6.2 / Eq. 6.3 (p. 97-98): the
    inverse of addition. Given the opinion about a union (x1Ux2) and the
    opinion about one of its parts x2, computes the opinion about the
    remainder (x1Ux2)\\x2.

    Preserves subtraction of projected probabilities (Eq. 6.5):
    P(x1) = P(x1Ux2) - P(x2).

    Raises:
        ValueError: if a_(x1Ux2) - a_x2 == 0 (division by zero in Eq. 6.3),
            or if the result would have negative uncertainty or disbelief
            mass, meaning opinion_x2 is not actually consistent with
            being a subset of opinion_union (Eq. 6.4's constraints).
    """
    base_rate_diff = opinion_union.base_rate - opinion_x2.base_rate
    if base_rate_diff == 0:
        raise ValueError(
            "Cannot subtract opinions whose base rates are equal (division by zero in Eq. 6.3)."
        )

    belief = opinion_union.belief - opinion_x2.belief
    disbelief = (
        opinion_union.base_rate * (opinion_union.disbelief + opinion_x2.belief)
        - opinion_x2.base_rate * (1 + opinion_x2.belief - opinion_union.belief - opinion_x2.uncertainty)
    ) / base_rate_diff
    uncertainty = (
        opinion_union.base_rate * opinion_union.uncertainty - opinion_x2.base_rate * opinion_x2.uncertainty
    ) / base_rate_diff
    base_rate = base_rate_diff

    if uncertainty < -_TOLERANCE or disbelief < -_TOLERANCE:
        raise ValueError(
            "Subtraction produced a negative uncertainty or disbelief mass "
            f"(u={uncertainty!r}, d={disbelief!r}). opinion_x2 is not "
            "consistent with being a genuine subset of opinion_union "
            "(non-negativity constraints of Eq. 6.4 are violated)."
        )

    # Clamp tiny negative values caused by floating point rounding.
    uncertainty = max(uncertainty, 0.0)
    disbelief = max(disbelief, 0.0)

    return BinomialOpinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty, base_rate=base_rate)


def complement(opinion: BinomialOpinion) -> BinomialOpinion:
    """
    Complement of an opinion, Definition 6.3 / Eq. 6.6 (p. 99): swaps
    belief and disbelief, keeps uncertainty, and complements the base
    rate.

    Preserves complement of projected probabilities (Eq. 6.8):
    P(not x) = 1 - P(x).
    """
    return BinomialOpinion(
        belief=opinion.disbelief,
        disbelief=opinion.belief,
        uncertainty=opinion.uncertainty,
        base_rate=1.0 - opinion.base_rate,
    )