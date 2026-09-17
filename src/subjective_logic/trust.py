"""
Computational trust: trust discounting and trust revision.

Implements Chapter 14 (Josang, 2016, pp. 243-270): deriving an opinion
from a transitive trust path (trust discounting), and revising referral
trust in the face of conflicting advice (trust revision).

Trust fusion (Section 14.4) needs no new code here: it is simply
discount() followed by one of Chapter 12's fusion operators -- see the
tests for a worked example composing the two.
"""

from __future__ import annotations

from typing import Mapping


def discount(trust_probability: float, belief: Mapping, uncertainty: float, domain) -> tuple:
    """
    Probability-sensitive trust discounting, Definition 14.6 / Eq. 14.6
    (pp. 255-256): discounts a source opinion by a trust probability.

        b^[A;B]_X(x) = P^A_B * b_X(x)
        u^[A;B]_X = 1 - P^A_B * sum(b_X)     [= 1 - P^A_B * (1 - u_X)]
        a^[A;B]_X(x) = a_X(x)                  (unchanged, not returned here)

    trust_probability = 1 (complete trust): result equals the source
    opinion exactly. trust_probability = 0 (complete distrust): result
    is fully vacuous.

    For multi-edge paths (Definition 14.7), pass the product of the
    chain's projected probabilities (see referral_trust_probability)
    as trust_probability -- the formula is otherwise identical.
    """
    total_belief = 1.0 - uncertainty
    belief_discounted = {x: trust_probability * belief[x] for x in domain}
    uncertainty_discounted = 1.0 - trust_probability * total_belief
    return belief_discounted, uncertainty_discounted


def referral_trust_probability(trust_chain) -> float:
    """
    Referral trust projected probability, Eq. 14.13 (p. 259): the
    product of projected probabilities along a chain of referral trust
    opinions (each a BinomialOpinion).
    """
    result = 1.0
    for trust in trust_chain:
        result *= trust.projected_probability
    return result


def uncertainty_differential(uncertainty_a: float, uncertainty_b: float) -> float:
    """
    Uncertainty differential UD(omega_A | omega_B), Eq. 14.23 (p. 266):
    the relative share of trust revision that omega_A should receive,
    given its uncertainty relative to omega_B's. UD in [0, 1]; the MORE
    uncertain opinion gets the HIGHER UD (and thus more revision).
    """
    denom = uncertainty_a + uncertainty_b
    return 0.5 if denom == 0 else uncertainty_a / denom


def revision_factor(uncertainty_a: float, uncertainty_b: float, degree_of_conflict: float) -> float:
    """Revision factor RF(omega_A), Eq. 14.24 (p. 266): UD(A|B) * DC."""
    return uncertainty_differential(uncertainty_a, uncertainty_b) * degree_of_conflict