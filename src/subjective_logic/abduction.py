"""
Subjective abduction and Bayes' theorem in Subjective Logic.

Implements Chapter 10 (Josang, 2016, pp. 171-198): inverting conditional
opinions (the subjective Bayes' theorem) and using the inverted
conditionals for deduction (abduction).

DESIGN NOTE: unlike Chapter 9's binomial deduction (Definition 9.1),
Definition 10.6 (multinomial subjective Bayes' theorem) does NOT have
the bar/complement transcription ambiguity that forced a workaround
last chapter -- its formulas are indexed sums and mins over well-defined
sets, not paired x/x-bar comparisons. So multinomial_invert() below is
implemented directly from the book, validated exactly against the
Military Intelligence example (Section 10.8.2) and Table 10.1's clean
in-text equation for a single inversion. binomial_invert() still wraps
it (same architectural pattern as Chapter 9's binomial_deduce), for
consistency and code reuse rather than out of necessity this time.
"""

from __future__ import annotations

from typing import Mapping

from .binomial import BinomialOpinion
from .deduction import _marginal_base_rate, multinomial_deduce
from .multinomial import MultinomialOpinion
from .multinomial_operators import _theoretical_max_uncertainty


def relevance(conditionals: Mapping, y) -> float:
    """
    Relevance Psi(y|X) of parent variable X to a specific child value y,
    Definition 10.2 / Eq. 10.5 (p. 174): how much the conditionals'
    projected probability for y varies across values of X.

    Args:
        conditionals: {x: MultinomialOpinion on Y}, one per parent value.
        y: the specific child value to compute relevance for.
    """
    projected_values = [cond.projected_probabilities[y] for cond in conditionals.values()]
    return max(projected_values) - min(projected_values)


def irrelevance(conditionals: Mapping, y) -> float:
    """Irrelevance, the complement of relevance, Eq. 10.6 (p. 174)."""
    return 1.0 - relevance(conditionals, y)


def dependence(conditionals: Mapping) -> float:
    """
    Dependence delta(Y|X) of child Y on parent X, Definition 10.3 / Eq.
    10.7 (p. 175): the greatest relevance across all values of Y.
    """
    domain_y = next(iter(conditionals.values())).domain
    return max(relevance(conditionals, y) for y in domain_y)


def independence(conditionals: Mapping) -> float:
    """Independence, the complement of dependence, Eq. 10.8 (p. 175)."""
    return 1.0 - dependence(conditionals)


def multinomial_invert(
    conditionals: Mapping, base_rates_x: Mapping, base_rates_y: Mapping = None
) -> dict:
    """
    Multinomial subjective Bayes' theorem, Definition 10.6 / Eq.
    10.36-10.51 (pp. 187-193): inverts a set of conditional opinions
    omega_Y|xi into the opposite-direction set omega_X|yj.

    Args:
        conditionals: {x: MultinomialOpinion on Y}, one per value of X
            (same shape as multinomial_deduce's `conditionals`).
        base_rates_x: a_X, base rate distribution over the parent.
        base_rates_y: optional explicit a_Y; if omitted, computed as the
            MBR (Eq. 9.68), same logic as multinomial_deduce.

    Returns:
        {y: MultinomialOpinion on X}, one inverted conditional per value
        of Y, sharing base_rates_x.

    Validated exactly against the Military Intelligence example's y2
    column (Section 10.8.2, Table 10.6, p. 197) and Table 10.1's clean
    single-inversion equation (p. 182).
    """
    domain_x = list(base_rates_x.keys())
    domain_y = next(iter(conditionals.values())).domain

    if base_rates_y is None:
        base_rates_y = _marginal_base_rate(base_rates_x, conditionals)

    projected = {
        x: {y: cond.belief_masses[y] + base_rates_y[y] * cond.uncertainty for y in domain_y}
        for x, cond in conditionals.items()
    }

    # Step 2 (Eq. 9.42-9.48 pattern reused here): weighted proportional
    # uncertainty u_Y||X, shared across all inverted conditionals.
    total_uncertainty = sum(cond.uncertainty for cond in conditionals.values())  # u^S_Y|X, Eq. 10.42
    weighted_uncertainty = 0.0  # u^w_Y|X, Eq. 10.48
    for x, cond in conditionals.items():
        weight = cond.uncertainty / total_uncertainty if total_uncertainty > 0 else 0.0  # Eq. 10.43
        max_uncertainty_x = _theoretical_max_uncertainty(projected[x], base_rates_y)  # ü_Y|x, Eq. 10.46
        contribution = (weight * cond.uncertainty / max_uncertainty_x) if max_uncertainty_x > 0 else 0.0  # Eq. 10.47
        weighted_uncertainty += contribution

    inverted = {}
    for y in domain_y:
        # Step 1: theoretical max uncertainty of the inverted conditional (Eq. 10.41).
        denominator = sum(base_rates_x[x] * projected[x][y] for x in domain_x)
        if denominator == 0:
            raise ValueError(f"Cannot invert: the marginal probability of {y!r} is zero.")
        max_uncertainty_y = min(projected[x][y] / denominator for x in domain_x)

        # Step 3: relevance/irrelevance and relative uncertainty (coproduct, Eq. 10.49).
        values = [projected[x][y] for x in domain_x]
        this_relevance = max(values) - min(values)
        this_irrelevance = 1.0 - this_relevance
        relative_uncertainty = (
            weighted_uncertainty + this_irrelevance - weighted_uncertainty * this_irrelevance
        )

        # Step 4 (Eq. 10.50).
        uncertainty_y = max_uncertainty_y * relative_uncertainty

        # Belief masses (Eq. 10.36 for P(x|y), then Eq. 10.51 for belief).
        belief = {}
        for x in domain_x:
            projected_x_given_y = base_rates_x[x] * projected[x][y] / denominator
            belief[x] = projected_x_given_y - uncertainty_y * base_rates_x[x]

        inverted[y] = MultinomialOpinion(belief_masses=belief, uncertainty=uncertainty_y, base_rates=dict(base_rates_x))

    return inverted


def binomial_invert(
    conditional_given_x: BinomialOpinion, conditional_given_not_x: BinomialOpinion, base_rate_x: float
) -> tuple:
    """
    Binomial subjective Bayes' theorem, Definition 10.4 (pp. 175-181),
    via the multinomial algorithm -- see this module's docstring.

    Returns:
        (inverted_given_y, inverted_given_not_y): the pair (omega_x'|y,
        omega_x'|not_y).

    Validated exactly against Table 10.1's single-inversion equation:
    inverting omega_y|x=(0.80,0.20,0,0.50), omega_y|not_x=(0.20,0.80,0,
    0.50) with a_x=0.50 gives omega_x'|y=(0.72,0.12,0.16,0.50).
    """
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
    base_rates_x = {"x": base_rate_x, "not_x": 1.0 - base_rate_x}

    inverted = multinomial_invert(conditionals, base_rates_x)

    inverted_given_y = BinomialOpinion(
        belief=inverted["y"].belief_masses["x"],
        disbelief=inverted["y"].belief_masses["not_x"],
        uncertainty=inverted["y"].uncertainty,
        base_rate=base_rate_x,
    )
    inverted_given_not_y = BinomialOpinion(
        belief=inverted["not_y"].belief_masses["x"],
        disbelief=inverted["not_y"].belief_masses["not_x"],
        uncertainty=inverted["not_y"].uncertainty,
        base_rate=base_rate_x,
    )
    return inverted_given_y, inverted_given_not_y


def multinomial_abduce(
    opinion_y: MultinomialOpinion, conditionals: Mapping, base_rates_x: Mapping, base_rates_y: Mapping = None
) -> MultinomialOpinion:
    """
    Multinomial abduction, Definition 10.7 / Eq. 10.53-10.56 (pp.
    193-194): invert conditionals then deduce, using the evidence
    opinion on Y to derive an opinion on X.

    Validated exactly against the Military Intelligence example
    (Section 10.8.2, Eq. 10.67).
    """
    inverted = multinomial_invert(conditionals, base_rates_x, base_rates_y)
    return multinomial_deduce(opinion_y, inverted, base_rates_y=base_rates_x)


def binomial_abduce(
    opinion_y: BinomialOpinion,
    conditional_given_x: BinomialOpinion,
    conditional_given_not_x: BinomialOpinion,
    base_rate_x: float,
) -> BinomialOpinion:
    """
    Binomial abduction, Definition 10.5 (pp. 183-184): invert the pair
    of conditionals, then apply binomial deduction with the evidence
    opinion on y.
    """
    from .deduction import binomial_deduce

    inverted_y, inverted_not_y = binomial_invert(conditional_given_x, conditional_given_not_x, base_rate_x)
    return binomial_deduce(opinion_y, inverted_y, inverted_not_y)