"""
Joint and marginal opinions.

Implements Chapter 11 (Josang, 2016, pp. 199-206): combining an
evidence opinion on a parent variable with a set of conditionals into a
single joint opinion over the product domain, and the reverse operation
of marginalising a joint opinion back onto its factor variables.

As dependence between X and Y decreases toward independence, the joint
operation converges to proportional multiplication (Section 8.1.4,
p. 203) -- conceptually, joint opinion computation is what
proportional_multiply becomes once the factors are no longer assumed
independent.
"""

from __future__ import annotations

from typing import Mapping

from .deduction import _marginal_base_rate, multinomial_deduce
from .multinomial import MultinomialOpinion
from .multinomial_operators import _theoretical_max_uncertainty


def joint_opinion(
    opinion_x: MultinomialOpinion, conditionals: Mapping, base_rates_y: Mapping = None
) -> MultinomialOpinion:
    """
    Joint opinion omega_YX, Section 11.2 (Eq. 11.7-11.17): combines an
    evidence opinion on parent X with a set of conditional opinions
    omega_Y|X into a single opinion over the joint domain Y x X (note
    the book's ordering: Y first, then X, matching the conditional
    direction convention P(Y|X)). Domain keys of the result are (y, x)
    tuples.

    Args:
        opinion_x: omega_X, the evidence opinion on the parent.
        conditionals: {x: MultinomialOpinion on Y}, one per value of X.
        base_rates_y: optional explicit a_Y; defaults to the MBR
            (Eq. 9.68), same as multinomial_deduce.
    """
    domain_x = opinion_x.domain
    domain_y = next(iter(conditionals.values())).domain
    base_rates_x = opinion_x.base_rates

    if base_rates_y is None:
        base_rates_y = _marginal_base_rate(base_rates_x, conditionals)

    projected_conditionals = {
        x: {y: cond.belief_masses[y] + base_rates_y[y] * cond.uncertainty for y in domain_y}
        for x, cond in conditionals.items()
    }
    projected_x = opinion_x.projected_probabilities

    projected_joint = {
        (y, x): projected_conditionals[x][y] * projected_x[x] for x in domain_x for y in domain_y
    }  # Eq. 11.3
    base_rates_joint = {
        (y, x): projected_conditionals[x][y] * base_rates_x[x] for x in domain_x for y in domain_y
    }  # Eq. 11.8: a_YX(y,x) = P(y|x) * a_X(x)

    # Step 1 (Section 11.2.2): theoretical max joint uncertainty, via
    # uncertainty-maximisation of the dogmatic joint opinion.
    dogmatic_joint = MultinomialOpinion(
        belief_masses=dict(projected_joint), uncertainty=0.0, base_rates=base_rates_joint
    )
    max_uncertainty_joint = dogmatic_joint.uncertainty_maximized().uncertainty  # u-double-dot_YX

    # Step 2: marginal uncertainties and their theoretical maxima.
    marginal_x_uncertainty = opinion_x.uncertainty  # u_<X>, Eq. 11.12
    max_marginal_x_uncertainty = _theoretical_max_uncertainty(projected_x, base_rates_x)

    deduced_y = multinomial_deduce(opinion_x, conditionals, base_rates_y)
    marginal_y_uncertainty = deduced_y.uncertainty  # u_<Y> (approximate), Eq. 11.12
    max_marginal_y_uncertainty = _theoretical_max_uncertainty(deduced_y.projected_probabilities, base_rates_y)

    # Step 3 (Eq. 11.14-11.15): proportional joint uncertainty.
    denominator = max_marginal_x_uncertainty + max_marginal_y_uncertainty
    uncertainty_joint = (
        0.0
        if denominator == 0
        else max_uncertainty_joint * (marginal_x_uncertainty + marginal_y_uncertainty) / denominator
    )

    belief_joint = {
        key: projected_joint[key] - uncertainty_joint * base_rates_joint[key] for key in projected_joint
    }  # Eq. 11.16

    return MultinomialOpinion(belief_masses=belief_joint, uncertainty=uncertainty_joint, base_rates=base_rates_joint)


def marginalize(joint: MultinomialOpinion) -> tuple:
    """
    Marginalise a joint opinion (domain of (y, x) tuples) onto its two
    factor variables, Section 11.3.1 (Eq. 11.18-11.21).

    Returns:
        (marginal_y, marginal_x): two MultinomialOpinion instances.

    Note (p. 206): the marginal projected probabilities and base rates
    exactly match the original evidence/deduced opinions, but the
    marginal UNCERTAINTY generally does NOT exactly match omega_X's or
    the deduced omega_Y||X's uncertainty from Chapter 9 -- this is an
    acknowledged approximation in the book itself ("the exact nature of
    this approximation needs further investigation", p. 206), not a bug
    here.
    """
    domain_y, domain_x = [], []
    seen_y, seen_x = set(), set()
    for y, x in joint.domain:
        if y not in seen_y:
            seen_y.add(y)
            domain_y.append(y)
        if x not in seen_x:
            seen_x.add(x)
            domain_x.append(x)

    projected, belief, base_rates = joint.projected_probabilities, joint.belief_masses, joint.base_rates

    marginal_y = MultinomialOpinion(
        belief_masses={y: sum(belief[(y, x)] for x in domain_x) for y in domain_y},
        uncertainty=joint.uncertainty,  # Eq. 11.21
        base_rates={y: sum(base_rates[(y, x)] for x in domain_x) for y in domain_y},
    )
    marginal_x = MultinomialOpinion(
        belief_masses={x: sum(belief[(y, x)] for y in domain_y) for x in domain_x},
        uncertainty=joint.uncertainty,
        base_rates={x: sum(base_rates[(y, x)] for y in domain_y) for x in domain_x},
    )
    return marginal_y, marginal_x


def marginal_conditionals(joint: MultinomialOpinion) -> dict:
    """
    Marginal conditional opinions omega_<Y|x>, Eq. 11.22-11.23 (p. 204):
    derived FROM a joint opinion, as opposed to being the original input
    to joint_opinion(). Generally produces MORE uncertainty than the
    genuine conditionals that built the joint, since a marginal slice
    rests on less evidence than the whole joint.

    Note (p. 204, the book's own words): "the exact nature of this
    approximation needs further investigation" -- treat this function's
    output as a secondary/exploratory tool, not a load-bearing inverse
    of joint_opinion().
    """
    domain_y, domain_x = [], []
    seen_y, seen_x = set(), set()
    for y, x in joint.domain:
        if y not in seen_y:
            seen_y.add(y)
            domain_y.append(y)
        if x not in seen_x:
            seen_x.add(x)
            domain_x.append(x)

    projected = joint.projected_probabilities
    marginal_y, _ = marginalize(joint)
    base_rates_y = marginal_y.base_rates
    max_uncertainty_joint = joint.uncertainty_maximized().uncertainty

    result = {}
    for x in domain_x:
        marginal_prob_x = sum(projected[(y, x)] for y in domain_y)
        if marginal_prob_x == 0:
            raise ValueError(f"Cannot derive a marginal conditional for x={x!r}: its marginal probability is zero.")

        projected_conditional = {y: projected[(y, x)] / marginal_prob_x for y in domain_y}  # Eq. 11.22
        max_uncertainty_conditional = _theoretical_max_uncertainty(projected_conditional, base_rates_y)

        uncertainty_conditional = (
            0.0
            if max_uncertainty_joint == 0
            else joint.uncertainty * max_uncertainty_conditional / max_uncertainty_joint
        )  # Eq. 11.23

        belief_conditional = {
            y: projected_conditional[y] - base_rates_y[y] * uncertainty_conditional for y in domain_y
        }
        result[x] = MultinomialOpinion(
            belief_masses=belief_conditional, uncertainty=uncertainty_conditional, base_rates=base_rates_y
        )

    return result