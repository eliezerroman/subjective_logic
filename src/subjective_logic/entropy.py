"""
Entropy and surprisal in the opinion model.

Implements Section 4.7 (Josang, 2016, pp. 75-79): classical information-
theoretic surprisal and entropy, extended to opinions by decomposing them
into sharpness, vagueness and uncertainty components.

This is a complementary degradation metric to the ones already used in
the EQM (mean uncertainty, structural error, temporal instability):
entropy measures how hard an outcome distribution is to predict, which
is a different (though related) signal from raw uncertainty mass.
"""

from __future__ import annotations

import math
from typing import Mapping


def surprisal(probability: float) -> float:
    """
    Surprisal (self-information) of an outcome with the given probability,
    Eq. 4.48/4.49 (p. 76): I(x) = -log2(P(x)), measured in bits.

    Returns +inf for probability == 0 (an "impossible" outcome that
    happened would carry infinite information).
    """
    if probability < 0:
        raise ValueError(f"Probability must be non-negative (got {probability!r}).")
    if probability == 0:
        return math.inf
    return -math.log2(probability)


def opinion_entropy(projected_probabilities: Mapping) -> float:
    """
    Opinion entropy H_P(omega_X), Eq. 4.55 (p. 77): the expected surprisal
    over the projected probability distribution.

        H_P(omega_X) = -sum_x P_X(x) * log2(P_X(x))

    By convention, terms with P(x) = 0 contribute 0 (since x*log(x) -> 0
    as x -> 0). Note (Proposition 4.1, p. 78): this is insensitive to
    uncertainty mass as long as the projected probability distribution
    stays the same -- two very differently-confident opinions can have
    identical opinion entropy.
    """
    return -sum(p * math.log2(p) for p in projected_probabilities.values() if p > 0)


def sharpness_entropy(sharp_masses: Mapping, projected_probabilities: Mapping) -> float:
    """
    Sharpness entropy H_S(omega_X), Eq. 4.56 (p. 78): expected surprisal
    attributable specifically to sharp belief mass.

        H_S(omega_X) = -sum_x b^S_X(x) * log2(P_X(x))
    """
    return -sum(
        sharp_masses[x] * math.log2(p) for x, p in projected_probabilities.items() if p > 0
    )


def vagueness_entropy(vague_masses: Mapping, projected_probabilities: Mapping) -> float:
    """
    Vagueness entropy H_V(omega_X), Eq. 4.57 (p. 78): expected surprisal
    attributable to vague belief mass. Always 0 for binomial/multinomial
    opinions (they have no vagueness).
    """
    return -sum(
        vague_masses[x] * math.log2(p) for x, p in projected_probabilities.items() if p > 0
    )


def uncertainty_entropy(focal_masses: Mapping, projected_probabilities: Mapping) -> float:
    """
    Uncertainty entropy H_U(omega_X), Eq. 4.58 (p. 78): expected surprisal
    attributable to focal uncertainty mass.

    Note (Eq. 4.59): H_S + H_V + H_U == H_P exactly, so this decomposition
    tells you *why* an opinion is hard to predict -- from real
    discriminating evidence (sharpness), from vague evidence (vagueness),
    or from plain lack of evidence (uncertainty). This is a much richer
    diagnostic than raw entropy alone for distinguishing degradation
    mechanisms (e.g. miscalibration vs. compression-induced vacuity).
    """
    return -sum(
        focal_masses[x] * math.log2(p) for x, p in projected_probabilities.items() if p > 0
    )


def cross_entropy(base_rates: Mapping, projected_probabilities: Mapping) -> float:
    """
    Base-rate to projected-probability cross entropy H_BP(omega_X),
    Definition 4.19 / Eq. 4.60 (p. 79): measures how much the opinion's
    projected probability distribution has moved away from the prior
    (base rate) distribution.

        H_BP(omega_X) = -sum_x a_X(x) * log2(P_X(x))
    """
    return -sum(
        base_rates[x] * math.log2(p) for x, p in projected_probabilities.items() if p > 0
    )