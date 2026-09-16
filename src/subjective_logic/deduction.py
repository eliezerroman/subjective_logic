"""
Conditional deduction in Subjective Logic.

Implements Chapter 9 (Josang, 2016, pp. 133-170): given a parent
(evidence) opinion and a set of conditional opinions on a child
variable, deduces the opinion on the child.

DESIGN NOTE on binomial vs. multinomial: the book gives a closed-form
expression for binomial deduction (Definition 9.1, Eq. 9.40-9.51) with
nine case distinctions comparing a conditional's belief/disbelief mass
against its complement's. The source PDF's text extraction collapses
the bar diacritic that distinguishes a value from its complement in
several of those case conditions and K-factor formulas, making a
literal transcription an unacceptable risk of a silent sign/term error
in a chapter this project depends on heavily.

Instead, multinomial_deduce() implements the unambiguous three-step
algorithm of Definition 9.2 (Section 9.5.4) as the single source of
truth, and binomial_deduce() is a thin wrapper around it -- justified
directly by the book itself (p. 151): "The closed form expression of
Definition 9.1 is equivalent to the special case of 2x2 deduction using
the multinomial method for conditional deduction of Definition 9.2."
Both paths are validated exactly against the book's own worked examples
(see tests): Figure 9.8 and Section 9.4.4's example (both 2x2 binomial
cases) and the Match-Fixing example (Section 9.6, a 2x3 multinomial
case) -- all match to full precision, not just rounded agreement.
"""

from __future__ import annotations

from typing import Mapping

from .binomial import BinomialOpinion
from .multinomial import MultinomialOpinion

_TOLERANCE = 1e-9


def _marginal_base_rate(base_rates_x: Mapping, conditionals: Mapping) -> dict:
    """
    Marginal base rate (MBR) distribution a_Y, Eq. 9.57/9.68 (p. 155-156):
    the base rate distribution for the child variable Y, required for
    consistency so that repeated conditional inversions preserve
    projected probabilities.

    Raises:
        ValueError: if the conditionals are collectively fully vacuous
        (Eq. 9.68's denominator <= 0) -- in that case there is no
        constraint on a_Y (p. 156), and the caller must supply one
        explicitly via multinomial_deduce's base_rates_y argument.
    """
    domain_y = next(iter(conditionals.values())).domain
    denominator = 1.0 - sum(base_rates_x[x] * cond.uncertainty for x, cond in conditionals.items())
    if denominator <= 0:
        raise ValueError(
            "Cannot compute the marginal base rate: the conditionals are collectively too "
            "vacuous (Eq. 9.68's denominator is non-positive). Supply base_rates_y explicitly "
            "(Section 9.5.1: with fully vacuous conditionals, a_Y is unconstrained)."
        )
    return {
        y: sum(base_rates_x[x] * conditionals[x].belief_masses[y] for x in conditionals) / denominator
        for y in domain_y
    }


def free_base_rate_interval(base_rates_x: Mapping, conditionals: Mapping) -> dict:
    """
    Free base-rate interval [a_Y-(y), a_Y+(y)] for each y, Eq. 9.58-9.59
    (p. 156, generalising the binomial Eq. 9.37-9.38): the range of base
    rates for Y consistent with the given conditionals, as an
    alternative to committing to the single MBR value.

    Since the sub-simplex apex projected probability is linear (non-
    decreasing) in a_Y(y) for fixed y, the extremes are reached exactly
    at a_Y(y) = 0 and a_Y(y) = 1, giving closed forms:

        a_Y-(y) = sum_x a_X(x) * b_Y|x(y)
        a_Y+(y) = sum_x a_X(x) * (b_Y|x(y) + u_Y|x)

    Validated exactly against the book's screenshots (Figures 9.6-9.7,
    p. 149): a_y- = 0.32, a_y+ = 0.52 for that example's conditionals.
    """
    domain_y = next(iter(conditionals.values())).domain
    return {
        y: (
            sum(base_rates_x[x] * cond.belief_masses[y] for x, cond in conditionals.items()),
            sum(base_rates_x[x] * (cond.belief_masses[y] + cond.uncertainty) for x, cond in conditionals.items()),
        )
        for y in domain_y
    }


def multinomial_deduce(
    opinion_x: MultinomialOpinion, conditionals: Mapping, base_rates_y: Mapping = None
) -> MultinomialOpinion:
    """
    Multinomial conditional deduction, Definition 9.2 / Eq. 9.64-9.79
    (pp. 154-161): given an opinion on a parent variable X and a set of
    conditional opinions omega_Y|xi (one per value of X), deduces the
    opinion on the child variable Y.

    Args:
        opinion_x: the evidence opinion on the parent, omega_X.
        conditionals: a mapping {x: MultinomialOpinion on Y}, one entry
            for every x in opinion_x.domain, all sharing the same
            domain for Y. Only each conditional's belief_masses and
            uncertainty are used -- their own base_rates fields are
            IGNORED, since the book requires a single shared a_Y (the
            MBR) across all conditionals and the deduced opinion; this
            function is the sole authority for computing it (Step 1),
            unless base_rates_y is supplied explicitly.
        base_rates_y: optional explicit base rate distribution for Y
            (e.g. from free_base_rate_interval). If omitted, the MBR is
            computed automatically -- the recommended default.

    Three steps (Section 9.5.4):
        1. Marginal base rate a_Y (Eq. 9.68).
        2. Sub-simplex apex opinion omega_{Y||X~}: the deduced opinion
           for a vacuous antecedent, with maximum uncertainty consistent
           with the conditionals (Eq. 9.69-9.74).
        3. Linear projection of omega_X into the resulting sub-simplex
           (Eq. 9.75-9.77).
    """
    if set(conditionals.keys()) != set(opinion_x.domain):
        raise ValueError("Must supply exactly one conditional opinion per value of opinion_x's domain.")

    domain_y = next(iter(conditionals.values())).domain
    for x, cond in conditionals.items():
        if cond.domain != domain_y:
            raise ValueError("All conditional opinions must share the same domain for Y.")

    base_rates_x = opinion_x.base_rates

    # Step 1.
    if base_rates_y is None:
        base_rates_y = _marginal_base_rate(base_rates_x, conditionals)

    projected_conditionals = {
        x: {y: cond.belief_masses[y] + base_rates_y[y] * cond.uncertainty for y in domain_y}
        for x, cond in conditionals.items()
    }

    # Step 2.
    projected_apex = {
        y: sum(base_rates_x[x] * projected_conditionals[x][y] for x in conditionals) for y in domain_y
    }
    candidate_apex_uncertainties = [
        (projected_apex[y] - min(conditionals[x].belief_masses[y] for x in conditionals)) / base_rates_y[y]
        for y in domain_y
        if base_rates_y[y] > 0
    ]
    if not candidate_apex_uncertainties:
        raise ValueError("Cannot determine the sub-simplex apex uncertainty: all base rates for Y are zero.")
    apex_uncertainty = max(min(candidate_apex_uncertainties), 0.0)  # Eq. 9.74, clamped for float noise

    # Step 3.
    uncertainty_yx = opinion_x.uncertainty * apex_uncertainty + sum(
        conditionals[x].uncertainty * opinion_x.belief_masses[x] for x in conditionals
    )  # Eq. 9.75

    projected_x = opinion_x.projected_probabilities
    projected_yx = {
        y: sum(projected_x[x] * projected_conditionals[x][y] for x in conditionals) for y in domain_y
    }  # Eq. 9.61
    belief_yx = {y: projected_yx[y] - base_rates_y[y] * uncertainty_yx for y in domain_y}  # Eq. 9.77

    return MultinomialOpinion(belief_masses=belief_yx, uncertainty=uncertainty_yx, base_rates=base_rates_y)


def binomial_deduce(
    opinion_x: BinomialOpinion, conditional_given_x: BinomialOpinion, conditional_given_not_x: BinomialOpinion
) -> BinomialOpinion:
    """
    Binomial conditional deduction, Definition 9.1 (pp. 150-152), by way
    of the multinomial algorithm -- see this module's docstring for why.

    Args:
        opinion_x: omega_x, the evidence opinion on the antecedent.
        conditional_given_x: omega_y|x, opinion on the consequent given
            the antecedent is TRUE.
        conditional_given_not_x: omega_y|not_x, opinion on the
            consequent given the antecedent is FALSE.
    """
    opinion_x_multi = MultinomialOpinion(
        belief_masses={"x": opinion_x.belief, "not_x": opinion_x.disbelief},
        uncertainty=opinion_x.uncertainty,
        base_rates={"x": opinion_x.base_rate, "not_x": 1.0 - opinion_x.base_rate},
    )
    conditionals = {
        "x": MultinomialOpinion(
            belief_masses={"y": conditional_given_x.belief, "not_y": conditional_given_x.disbelief},
            uncertainty=conditional_given_x.uncertainty,
            base_rates={"y": conditional_given_x.base_rate, "not_y": 1.0 - conditional_given_x.base_rate},
        ),
        "not_x": MultinomialOpinion(
            belief_masses={"y": conditional_given_not_x.belief, "not_y": conditional_given_not_x.disbelief},
            uncertainty=conditional_given_not_x.uncertainty,
            base_rates={"y": conditional_given_not_x.base_rate, "not_y": 1.0 - conditional_given_not_x.base_rate},
        ),
    }

    result = multinomial_deduce(opinion_x_multi, conditionals)

    return BinomialOpinion(
        belief=result.belief_masses["y"],
        disbelief=result.belief_masses["not_y"],
        uncertainty=result.uncertainty,
        base_rate=result.base_rates["y"],
    )


def material_implication(x_is_true: bool, y_is_true: bool, base_rate_y: float = 0.5) -> BinomialOpinion:
    """
    Subjective material implication, Table 9.4 / Section 9.7.4 (p. 168):
    a reinterpretation of (x -> y) as a binomial opinion rather than a
    Boolean, correctly representing the 'antecedent FALSE' cases as a
    vacuous opinion instead of the traditional (and, per Section 9.7.2,
    inconsistent-with-probability-calculus) Boolean TRUE.

    Mostly of pedagogical value (Section 9.7) rather than something the
    rest of this package depends on.
    """
    if not x_is_true:
        return BinomialOpinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
    if y_is_true:
        return BinomialOpinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=base_rate_y)
    return BinomialOpinion(belief=0.0, disbelief=1.0, uncertainty=0.0, base_rate=base_rate_y)