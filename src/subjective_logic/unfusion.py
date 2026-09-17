"""
Unfusion and fission of subjective opinions.

Implements Chapter 13 (Josang, 2016, pp. 237-242): the inverse
operations of cumulative/averaging fusion (Chapter 12) -- removing a
known contributor's opinion from an already-fused result (unfusion),
and splitting a single opinion into two according to a proportion
(fission).

Use case for this project: "what would the combined opinion have been
without agent B's contribution?" -- isolating how much a specific
agent in a fusion pipeline is responsible for the group's current
belief or uncertainty.

NOT implemented: averaging fission (Section 13.2.3). The book states
this operator is trivial and not worth formalising: averaging fusion
of two equal opinions always reproduces the same opinion, so
"averaging fission" would just return the argument opinion twice,
unchanged (p. 242) -- not an omission on this module's part.
"""

from __future__ import annotations

from typing import Mapping


def cumulative_unfusion(
    belief_fused: Mapping, uncertainty_fused: float,
    belief_known: Mapping, uncertainty_known: float,
    domain,
) -> tuple:
    """
    Cumulative unfusion, Definition 13.1 / Eq. 13.1-13.2 (pp. 238-239):
    the inverse of cumulative fusion (Definition 12.5). Given a
    cumulatively fused opinion omega_C = omega_A (+) omega_B and the
    known contributor omega_B, recovers omega_A.

    Returns (belief_a, uncertainty_a). The base rate is unaffected
    (shared across all three opinions per the book's own notation) and
    is not computed here -- see each opinion class's wrapper method.

    Validated exactly against Section 13.1.3's worked example.
    """
    u_b, u_c = uncertainty_known, uncertainty_fused
    if u_b == 0.0 and u_c == 0.0:
        # Degenerate case: both zero, limit ratios are path-dependent.
        # Same pragmatic convention as fusion's Case II (Sec. 12.3.1, p. 227).
        belief = {x: 0.5 * belief_fused[x] - 0.5 * belief_known[x] for x in domain}
        uncertainty = 0.0
    else:
        denom = u_b - u_c + u_b * u_c
        if denom == 0:
            raise ValueError("Cannot cumulatively unfuse: division by zero in Eq. 13.1 for these arguments.")
        belief = {x: (belief_fused[x] * u_b - belief_known[x] * u_c) / denom for x in domain}
        uncertainty = (u_b * u_c) / denom

    return belief, uncertainty


def averaging_unfusion(
    belief_fused: Mapping, uncertainty_fused: float,
    belief_known: Mapping, uncertainty_known: float,
    domain,
) -> tuple:
    """
    Averaging unfusion, Definition 13.2 / Eq. 13.4-13.5 (p. 239): the
    inverse of averaging fusion (Definition 12.7).
    """
    u_b, u_c = uncertainty_known, uncertainty_fused
    if u_b == 0.0 and u_c == 0.0:
        belief = {x: 0.5 * belief_fused[x] - 0.5 * belief_known[x] for x in domain}
        uncertainty = 0.0
    else:
        denom = 2 * u_b - u_c
        if denom == 0:
            raise ValueError("Cannot averaging-unfuse: division by zero in Eq. 13.4 for these arguments.")
        belief = {x: (2 * belief_fused[x] * u_b - belief_known[x] * u_c) / denom for x in domain}
        uncertainty = (u_b * u_c) / denom

    return belief, uncertainty


def cumulative_fission(belief: Mapping, uncertainty: float, phi: float, domain) -> tuple:
    """
    Cumulative fission, Definition 13.3 / Eq. 13.9 (p. 241): splits an
    opinion into two, omega_C1 (getting proportion phi of the evidence)
    and omega_C2 (getting the rest), such that
    omega_C1 (+) omega_C2 == the original opinion (cumulative fusion,
    Chapter 12).

    Args:
        phi: fission parameter, 0 < phi < 1. phi close to 1 gives
            omega_C1 most of the evidence (least uncertainty).

    Returns:
        (belief_1, uncertainty_1, belief_2, uncertainty_2). Base rates
        are unaffected by fission (p. 241).

    Validated exactly against Table 13.1's worked example (phi=0.75).
    """
    if not (0.0 < phi < 1.0):
        raise ValueError(f"phi must be strictly between 0 and 1 (got {phi!r}).")

    total_belief = 1.0 - uncertainty  # sum_i b(x_i), by additivity -- always 1 - u.

    denom_1 = uncertainty + phi * total_belief
    denom_2 = uncertainty + (1.0 - phi) * total_belief

    belief_1 = {x: phi * belief[x] / denom_1 for x in domain}
    uncertainty_1 = uncertainty / denom_1

    belief_2 = {x: (1.0 - phi) * belief[x] / denom_2 for x in domain}
    uncertainty_2 = uncertainty / denom_2

    return belief_1, uncertainty_1, belief_2, uncertainty_2