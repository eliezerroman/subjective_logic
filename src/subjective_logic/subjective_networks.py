"""
Chain rules for subjective Bayesian networks.

Implements Section 17.2 (Josang, 2016, pp. 312-316): propagating an
opinion, a set of conditionals, or a joint opinion through a chain of
dependent variables X1 => X2 => ... => XK, by repeated application of
operators already implemented in earlier chapters (deduction, Ch. 9;
subjective Bayes' theorem, Ch. 10; joint opinions, Ch. 11).

Sections 17.1 (classical Bayesian network background), 17.3
(predictive/diagnostic/intercausal/combined reasoning) and 17.5-17.6
(subjective network modelling, combining STNs and SBNs) introduce no
new mathematical operators -- they are direct applications or
compositions of primitives already implemented (deduction, abduction,
multiplication, division, joint opinions, trust discounting/fusion).
See tests/test_subjective_networks.py for worked compositions of
those, rather than new functions duplicated here.
"""

from __future__ import annotations

from typing import Mapping

from .abduction import multinomial_invert
from .deduction import multinomial_deduce
from .multinomial import MultinomialOpinion


def chain_conditionals(conditionals_1: Mapping, conditionals_2: Mapping, base_rates_2: Mapping = None) -> dict:
    """
    Chains two sets of conditionals, Eq. 17.13 (p. 312): given
    omega_Y|X (conditionals_1) and omega_Z|Y (conditionals_2), derives
    omega_Z|X -- for each value x, deduces the opinion on Z implied by
    the conditional opinion on Y given x, via omega_Z|Y.
    """
    return {x: multinomial_deduce(cond, conditionals_2, base_rates_2) for x, cond in conditionals_1.items()}


def chain_conditionals_multi(*conditional_chains: Mapping) -> dict:
    """Chains three or more sets of conditionals by repeated application of chain_conditionals."""
    if len(conditional_chains) < 2:
        raise ValueError("chain_conditionals_multi requires at least two conditional sets.")
    result = conditional_chains[0]
    for next_conditionals in conditional_chains[1:]:
        result = chain_conditionals(result, next_conditionals)
    return result


def chain_deduce(opinion: MultinomialOpinion, *conditional_chains: Mapping) -> MultinomialOpinion:
    """
    Propagates an opinion through a chain of conditionals, Eq. 17.14
    (p. 313): omega_X1 deduced through omega_X2|X1, then through
    omega_X3|X2, and so on, in sequence.

    Note (p. 313): irrelevance between the first and last variable
    increases monotonically along the chain (unless variables are
    totally dependent) -- a long enough chain of not-fully-dependent
    variables produces a totally irrelevant/vacuous result. The same
    "the message decays after enough hops" intuition Chapter 14 gave
    for long referral trust paths, now for conditional chains.
    """
    result = opinion
    for conditionals in conditional_chains:
        result = multinomial_deduce(result, conditionals)
    return result


def chain_invert(conditionals_1: Mapping, conditionals_2: Mapping, base_rates_1: Mapping) -> dict:
    """
    Inverts a chain of conditionals, Eq. 17.16 (p. 314): the book's
    recommended method over Eq. 17.15's more complex per-link
    inversion -- chain forward with chain_conditionals, then invert the
    result ONCE with the subjective Bayes' theorem (Ch. 10).

    Section 17.2.3 validates numerically that this gives the exact same
    projected probability as inverting link by link and chaining the
    inverted conditionals, differing only slightly (approximately) in
    uncertainty mass, at much lower implementation complexity -- hence
    only this simpler method is implemented here.
    """
    chained = chain_conditionals(conditionals_1, conditionals_2)
    return multinomial_invert(chained, base_rates_1)


def chain_joint(opinion: MultinomialOpinion, *conditional_chains: Mapping) -> MultinomialOpinion:
    """
    Builds a chained joint opinion, Eq. 17.20 (p. 316): repeatedly
    applies joint_opinion (Chapter 11) along a chain of variables.

    The domain keys of the result nest with each step (after one step:
    (y, x) tuples; after two: ((z, y), x) tuples, and so on), reflecting
    the successive pairing joint_opinion performs at each stage.
    """
    from .joint import joint_opinion

    result = opinion
    for conditionals in conditional_chains:
        result = joint_opinion(result, conditionals)
    return result