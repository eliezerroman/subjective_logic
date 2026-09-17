"""
Tests for chain rules (Chapter 17). Validates chain_invert against the
Bayes'-rule identity it must satisfy (Eq. 10.36 applied to the chained
conditional), and demonstrates -- without new library code -- the
composed reasoning patterns of Sections 17.3 and 17.5-17.6.
"""

import math

from subjective_logic import (
    BinomialOpinion,
    MultinomialOpinion,
    chain_conditionals,
    chain_deduce,
    chain_invert,
)


def _chained_example():
    base_rates_x = {"x1": 0.700, "x2": 0.300}
    base_rates_y = {"y1": 0.732, "y2": 0.268}
    base_rates_z = {"z1": 0.717, "z2": 0.282}

    y_given_x = {
        "x1": MultinomialOpinion(belief_masses={"y1": 0.950, "y2": 0.000}, uncertainty=0.050, base_rates=base_rates_y),
        "x2": MultinomialOpinion(belief_masses={"y1": 0.100, "y2": 0.850}, uncertainty=0.050, base_rates=base_rates_y),
    }
    z_given_y = {
        "y1": MultinomialOpinion(belief_masses={"z1": 0.900, "z2": 0.050}, uncertainty=0.050, base_rates=base_rates_z),
        "y2": MultinomialOpinion(belief_masses={"z1": 0.050, "z2": 0.850}, uncertainty=0.100, base_rates=base_rates_z),
    }
    return base_rates_x, y_given_x, z_given_y


def test_chain_conditionals_and_deduce_matches_table_17_2_setup():
    """Table 17.2 (p. 315): chaining omega_Y|X and omega_Z|Y should produce a well-formed omega_Z|X."""
    base_rates_x, y_given_x, z_given_y = _chained_example()
    chained_z_given_x = chain_conditionals(y_given_x, z_given_y)

    for x in base_rates_x:
        opinion = chained_z_given_x[x]
        total = sum(opinion.belief_masses.values()) + opinion.uncertainty
        assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_chain_invert_satisfies_bayes_rule():
    """
    Cross-check chain_invert's result against Bayes' rule (Eq. 10.36)
    computed independently from the chained conditional's own P(z|x)
    values -- not by calling multinomial_invert a second time, so this
    is a genuine check, not a circular one.
    """
    base_rates_x, y_given_x, z_given_y = _chained_example()
    chained_z_given_x = chain_conditionals(y_given_x, z_given_y)
    inverted = chain_invert(y_given_x, z_given_y, base_rates_x)

    projected_z1_given_x = {x: chained_z_given_x[x].projected_probabilities["z1"] for x in base_rates_x}
    numerator = {x: base_rates_x[x] * projected_z1_given_x[x] for x in base_rates_x}
    denom = sum(numerator.values())
    expected = {x: numerator[x] / denom for x in base_rates_x}

    actual = inverted["z1"].projected_probabilities
    for x in base_rates_x:
        assert math.isclose(actual[x], expected[x], abs_tol=1e-9)


def test_chain_deduce_uncertainty_does_not_decrease_along_a_chain():
    """p. 313: irrelevance (and thus uncertainty) generally grows along a chain of non-fully-dependent variables."""
    base_rates_x, y_given_x, z_given_y = _chained_example()
    opinion_x = MultinomialOpinion(belief_masses={"x1": 0.9, "x2": 0.0}, uncertainty=0.1, base_rates=base_rates_x)

    one_hop = multinomial_deduce_result = chain_deduce(opinion_x, y_given_x)
    two_hops = chain_deduce(opinion_x, y_given_x, z_given_y)

    assert two_hops.uncertainty >= one_hop.uncertainty - 1e-9


def test_composed_trust_and_deduction_chain():
    """
    Sections 17.5-17.6 (Eq. 17.35): a subjective network is just trust
    discounting (Ch. 14) composed with a conditional deduction chain
    (this chapter) -- no new operator needed.
    """
    base_rates_x, y_given_x, z_given_y = _chained_example()

    trust_in_b = BinomialOpinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
    opinion_x_from_b = BinomialOpinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
    discounted = opinion_x_from_b.discount_by(trust_in_b)

    opinion_x_multi = MultinomialOpinion(
        belief_masses={"x1": discounted.belief, "x2": discounted.disbelief},
        uncertainty=discounted.uncertainty,
        base_rates=base_rates_x,
    )
    result = chain_deduce(opinion_x_multi, y_given_x, z_given_y)

    total = sum(result.belief_masses.values()) + result.uncertainty
    assert math.isclose(total, 1.0, abs_tol=1e-9)