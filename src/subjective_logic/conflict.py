"""
Conflict between opinions.

Implements Section 4.8 (Josang, 2016, pp. 79-82): a measure of how much
two agents' opinions about the same variable disagree, accounting for
how confident (low-uncertainty) each opinion is.

This is directly useful for detecting disagreement in a multi-agent
system before or instead of fusing opinions -- e.g. flagging when two
agents in a decision chain diverge sharply, which is a precondition for
the "epistemic contagion" failure mode described in the review paper.
"""

from __future__ import annotations

from typing import Mapping


def projected_distance(p_b: Mapping, p_c: Mapping) -> float:
    """
    Projected distance PD(omega_B, omega_C), Eq. 4.61 (p. 80): half the
    total absolute difference between two projected probability
    distributions over the same domain.

        PD = sum_x |P_B(x) - P_C(x)| / 2

    PD in [0, 1]. PD = 0 for identical projected probabilities (opinions
    may still differ in belief/uncertainty composition); PD = 1 only for
    absolute opinions with opposite projected probability.
    """
    if set(p_b.keys()) != set(p_c.keys()):
        raise ValueError("Both distributions must be defined over the same domain.")
    return sum(abs(p_b[x] - p_c[x]) for x in p_b) / 2.0


def conjunctive_certainty(uncertainty_b: float, uncertainty_c: float) -> float:
    """
    Conjunctive certainty CC(omega_B, omega_C), Eq. 4.62 (p. 80): the
    joint confidence of two opinions.

        CC = (1 - u_B) * (1 - u_C)

    CC in [0, 1]. CC = 0 if either opinion is vacuous (so disagreement
    is defused: an uncertain opinion "doesn't count" much in a
    disagreement); CC = 1 only if both opinions are dogmatic.
    """
    return (1.0 - uncertainty_b) * (1.0 - uncertainty_c)


def degree_of_conflict(
    p_b: Mapping, uncertainty_b: float, p_c: Mapping, uncertainty_c: float
) -> float:
    """
    Degree of conflict DC(omega_B, omega_C), Definition 4.20 / Eq. 4.63
    (p. 80): how much two opinions genuinely disagree, weighted by how
    confident both opinions are.

        DC = PD(omega_B, omega_C) * CC(omega_B, omega_C)

    A large projected distance does NOT by itself mean real conflict if
    one or both opinions carry high uncertainty (p. 80): "the more
    uncertain one or both opinions are, the more tolerance for a large
    PD should be given."
    """
    return projected_distance(p_b, p_c) * conjunctive_certainty(uncertainty_b, uncertainty_c)