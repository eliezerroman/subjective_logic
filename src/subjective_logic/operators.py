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

def multiply(opinion_x: BinomialOpinion, opinion_y: BinomialOpinion) -> BinomialOpinion:
    """
    Binomial multiplication (product), Definition 7.1 / Eq. 7.1 (p. 102):
    given INDEPENDENT opinions about x and y from separate domains,
    computes the opinion about the conjunction x AND y.

    Preserves exact multiplication of projected probabilities (Eq. 7.2):
    P(x AND y) = P(x) * P(y). The belief/disbelief/uncertainty split,
    however, is only an APPROXIMATION of the analytically correct Beta
    product (Theorem 7.1, p. 106) -- very good in practice, worse only
    in the high-uncertainty extreme.

    WARNING -- independence assumption: using this on DEPENDENT opinions
    (e.g. two agents sharing training data, or two outputs of the same
    model) gives a mathematically wrong result, typically overconfident
    (understated uncertainty). Conditional multiplication for dependent
    opinions is Section 11.2, not yet implemented.
    """
    ax, ay = opinion_x.base_rate, opinion_y.base_rate
    bx, dx, ux = opinion_x.belief, opinion_x.disbelief, opinion_x.uncertainty
    by, dy, uy = opinion_y.belief, opinion_y.disbelief, opinion_y.uncertainty

    denominator = 1.0 - ax * ay
    if denominator == 0:
        raise ValueError("Cannot multiply opinions with ax * ay = 1 (division by zero in Eq. 7.1).")

    belief = bx * by + ((1 - ax) * ay * bx * uy + ax * (1 - ay) * ux * by) / denominator
    disbelief = dx + dy - dx * dy
    uncertainty = ux * uy + ((1 - ay) * bx * uy + (1 - ax) * ux * by) / denominator
    base_rate = ax * ay

    return BinomialOpinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty, base_rate=base_rate)


def comultiply(opinion_x: BinomialOpinion, opinion_y: BinomialOpinion) -> BinomialOpinion:
    """
    Binomial comultiplication (coproduct), Definition 7.2 / Eq. 7.3
    (p. 103): given INDEPENDENT opinions about x and y, computes the
    opinion about the disjunction x OR y.

    Preserves exact probabilistic OR of projected probabilities (Eq. 7.4):
    P(x OR y) = P(x) + P(y) - P(x)*P(y). Same approximation caveat and
    independence assumption as multiply() -- see its docstring.
    """
    ax, ay = opinion_x.base_rate, opinion_y.base_rate
    bx, dx, ux = opinion_x.belief, opinion_x.disbelief, opinion_x.uncertainty
    by, dy, uy = opinion_y.belief, opinion_y.disbelief, opinion_y.uncertainty

    denominator = ax + ay - ax * ay
    if denominator == 0:
        raise ValueError("Cannot comultiply opinions with ax + ay - ax*ay = 0 (division by zero in Eq. 7.3).")

    belief = bx + by - bx * by
    disbelief = dx * dy + (ax * (1 - ay) * dx * uy + (1 - ax) * ay * ux * dy) / denominator
    uncertainty = ux * uy + (ay * dx * uy + ax * ux * dy) / denominator
    base_rate = ax + ay - ax * ay

    return BinomialOpinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty, base_rate=base_rate)


def divide(opinion_xy: BinomialOpinion, opinion_y: BinomialOpinion) -> BinomialOpinion:
    """
    Binomial division, Definition 7.3 / Eq. 7.12 (p. 110-111): the
    inverse of multiplication. Given the opinion about a conjunction
    (x AND y) and the opinion about y, computes the opinion about x,
    such that opinion_xy == multiply(result, opinion_y).

    Preserves exact division of projected probabilities (Eq. 7.14):
    P(x) = P(x AND y) / P(y).

    Raises:
        ValueError: if the constraints of Eq. 7.13 are violated (the
        two input opinions are not consistent with a genuine
        multiplicative relationship), or on division by zero.
    """
    ax, bx, dx, ux = opinion_xy.base_rate, opinion_xy.belief, opinion_xy.disbelief, opinion_xy.uncertainty
    ay, by, dy, uy = opinion_y.base_rate, opinion_y.belief, opinion_y.disbelief, opinion_y.uncertainty

    if not (ax < ay):
        raise ValueError(f"Division requires ax < ay (Eq. 7.13); got ax={ax!r}, ay={ay!r}.")
    if dy == 1:
        raise ValueError("Division by zero: opinion_y has disbelief = 1 (Eq. 7.12).")

    denom_a = (ay - ax) * (by + ay * uy)
    denom_b = (ay - ax) * (1 - dy)
    if denom_a == 0 or denom_b == 0:
        raise ValueError("Division by zero encountered while dividing these opinions (Eq. 7.12).")

    belief = (ay * (bx + ax * ux)) / denom_a - (ax * (1 - dx)) / denom_b
    disbelief = (dx - dy) / (1 - dy)
    uncertainty = (ay * (1 - dx)) / denom_b - (ay * (bx + ax * ux)) / denom_a
    base_rate = ax / ay

    if belief < -_TOLERANCE or uncertainty < -_TOLERANCE:
        raise ValueError(
            f"Division produced a negative belief or uncertainty mass (b={belief!r}, u={uncertainty!r}). "
            "The input opinions are not consistent with a genuine multiplicative relationship (Eq. 7.13)."
        )
    belief = max(belief, 0.0)
    uncertainty = max(uncertainty, 0.0)

    return BinomialOpinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty, base_rate=base_rate)


def codivide(opinion_xy: BinomialOpinion, opinion_y: BinomialOpinion) -> BinomialOpinion:
    """
    Binomial codivision, Definition 7.4 / Eq. 7.15 (p. 112): the inverse
    of comultiplication. Given the opinion about a disjunction (x OR y)
    and the opinion about y, computes the opinion about x, such that
    opinion_xy == comultiply(result, opinion_y).

    Preserves exact codivision of projected probabilities (Eq. 7.17):
    P(x) = (P(x OR y) - P(y)) / (1 - P(y)).

    Raises:
        ValueError: if the constraints of Eq. 7.16 are violated, or on
        division by zero.
    """
    ax, bx, dx, ux = opinion_xy.base_rate, opinion_xy.belief, opinion_xy.disbelief, opinion_xy.uncertainty
    ay, by, dy, uy = opinion_y.base_rate, opinion_y.belief, opinion_y.disbelief, opinion_y.uncertainty

    if not (ax > ay):
        raise ValueError(f"Codivision requires ax > ay (Eq. 7.16); got ax={ax!r}, ay={ay!r}.")
    if by == 1:
        raise ValueError("Codivision by zero: opinion_y has belief = 1 (Eq. 7.15).")

    denom_ratio = ax - ay
    denom_d = denom_ratio * (dy + (1 - ay) * uy)
    denom_b = denom_ratio * (1 - by)
    if denom_ratio == 0 or denom_d == 0 or denom_b == 0:
        raise ValueError("Division by zero encountered while codividing these opinions (Eq. 7.15).")

    belief = (bx - by) / (1 - by)
    disbelief = ((1 - ay) * (dx + (1 - ax) * ux)) / denom_d - ((1 - ax) * (1 - bx)) / denom_b
    uncertainty = ((1 - ay) * (1 - bx)) / denom_b - ((1 - ay) * (dx + (1 - ax) * ux)) / denom_d
    base_rate = (ax - ay) / (1 - ay)

    if belief < -_TOLERANCE or disbelief < -_TOLERANCE or uncertainty < -_TOLERANCE:
        raise ValueError(
            f"Codivision produced a negative mass (b={belief!r}, d={disbelief!r}, u={uncertainty!r}). "
            "The input opinions are not consistent with a genuine disjunctive relationship (Eq. 7.16)."
        )
    belief = max(belief, 0.0)
    disbelief = max(disbelief, 0.0)
    uncertainty = max(uncertainty, 0.0)

    return BinomialOpinion(belief=belief, disbelief=disbelief, uncertainty=uncertainty, base_rate=base_rate)