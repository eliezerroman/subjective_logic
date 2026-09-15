"""
Tests for the sharp/vague/focal mass decomposition (Section 4.1-4.2),
validated against the worked example in Table 4.1 (Josang, 2016, p. 57).
"""

import math

from subjective_logic import HyperOpinion


def _table_4_1_opinion() -> HyperOpinion:
    base_rates = {"x1": 0.20, "x2": 0.30, "x3": 0.50}
    belief_masses = {
        frozenset({"x1"}): 0.10,
        frozenset({"x2"}): 0.10,
        frozenset({"x3"}): 0.00,
        frozenset({"x1", "x2"}): 0.20,
        frozenset({"x1", "x3"}): 0.30,
        frozenset({"x2", "x3"}): 0.10,
    }
    return HyperOpinion(belief_masses=belief_masses, uncertainty=0.20, base_rates=base_rates)


def test_table_4_1_mass_sum_for_x1():
    opinion = _table_4_1_opinion()
    sharp, vague, focal = opinion.mass_sum(frozenset({"x1"}))

    assert math.isclose(sharp, 0.10, abs_tol=1e-6)
    assert math.isclose(vague, 0.1657, abs_tol=0.01)
    assert math.isclose(focal, 0.04, abs_tol=1e-6)


def test_table_4_1_mass_sum_for_x2():
    opinion = _table_4_1_opinion()
    sharp, vague, focal = opinion.mass_sum(frozenset({"x2"}))

    assert math.isclose(sharp, 0.10, abs_tol=1e-6)
    assert math.isclose(vague, 0.1575, abs_tol=0.01)
    assert math.isclose(focal, 0.06, abs_tol=1e-6)


def test_table_4_1_mass_sum_for_composite_x4():
    """x4 = {x1, x2}: verifies sharp/vague also work correctly for composite values."""
    opinion = _table_4_1_opinion()
    sharp, vague, focal = opinion.mass_sum(frozenset({"x1", "x2"}))

    assert math.isclose(sharp, 0.40, abs_tol=1e-6)
    assert math.isclose(vague, 0.1232, abs_tol=0.01)
    assert math.isclose(focal, 0.10, abs_tol=1e-6)


def test_mass_sum_additivity_matches_projected_probability():
    """Eq. 4.9: sharp + vague + focal == projected probability, for every value in R(X)."""
    opinion = _table_4_1_opinion()
    for value in opinion.hyperdomain_values:
        sharp, vague, focal = opinion.mass_sum(value)
        assert math.isclose(sharp + vague + focal, opinion.projected_probability_of(value), abs_tol=1e-9)


def test_total_mass_sum_additivity():
    """Eq. 4.11: total sharp + total vague + uncertainty == 1."""
    opinion = _table_4_1_opinion()
    sharp, vague, uncertainty = opinion.total_mass_sum
    assert math.isclose(sharp + vague + uncertainty, 1.0, abs_tol=1e-9)


def test_binomial_and_multinomial_have_zero_vagueness():
    """Section 4.1.2: binomial and multinomial opinions never contain vagueness."""
    from subjective_logic import BinomialOpinion, MultinomialOpinion

    binomial = BinomialOpinion.from_evidence(r=2.0, s=1.0, base_rate=0.9)
    assert binomial.vague_belief_mass == 0.0

    base_rates = MultinomialOpinion.default_base_rates(["x1", "x2", "x3"])
    multinomial = MultinomialOpinion.from_evidence(evidence={"x1": 6.0, "x2": 1.0, "x3": 1.0}, base_rates=base_rates)
    assert multinomial.vague_belief_mass("x1") == 0.0