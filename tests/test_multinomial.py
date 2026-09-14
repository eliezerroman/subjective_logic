"""
Tests for MultinomialOpinion, validated against the worked examples from
Josang (2016), Sections 3.5.1, 3.5.3 and 3.5.6.
"""

import math

import pytest

from subjective_logic import MultinomialOpinion


def test_book_example_projected_probability():
    """
    Section 3.5.1 example: ternary domain X = {x1, x2, x3},
    b_X = {0.20, 0.20, 0.20}, u_X = 0.40, a_X = {0.750, 0.125, 0.125}
    should project to P_X = {0.50, 0.25, 0.25}.
    """
    opinion = MultinomialOpinion(
        belief_masses={"x1": 0.20, "x2": 0.20, "x3": 0.20},
        uncertainty=0.40,
        base_rates={"x1": 0.750, "x2": 0.125, "x3": 0.125},
    )

    projected = opinion.projected_probabilities
    assert math.isclose(projected["x1"], 0.50, abs_tol=1e-6)
    assert math.isclose(projected["x2"], 0.25, abs_tol=1e-6)
    assert math.isclose(projected["x3"], 0.25, abs_tol=1e-6)


def test_coarsening_example_expected_probability():
    """
    Section 3.5.3 urn example: default base rate 1/3 for each of x1, x2, x3.
    After observing r(x1)=6, r(x2)=1, r(x3)=1, the projected probability of
    x1 should be 2/3 (Eq. 3.17).
    """
    base_rates = MultinomialOpinion.default_base_rates(["x1", "x2", "x3"])
    opinion = MultinomialOpinion.from_evidence(
        evidence={"x1": 6.0, "x2": 1.0, "x3": 1.0}, base_rates=base_rates
    )

    projected = opinion.projected_probabilities
    assert math.isclose(projected["x1"], 2 / 3, abs_tol=1e-6)


def test_evidence_round_trip():
    """Converting opinion -> evidence -> opinion should be the identity."""
    base_rates = MultinomialOpinion.default_base_rates(["x1", "x2", "x3"])
    original = MultinomialOpinion.from_evidence(
        evidence={"x1": 6.0, "x2": 1.0, "x3": 1.0}, base_rates=base_rates
    )
    evidence = original.to_evidence()
    rebuilt = MultinomialOpinion.from_evidence(evidence=evidence, base_rates=base_rates)

    for x in original.domain:
        assert math.isclose(rebuilt.belief_masses[x], original.belief_masses[x], abs_tol=1e-9)
    assert math.isclose(rebuilt.uncertainty, original.uncertainty, abs_tol=1e-9)


def test_uncertainty_maximisation_preserves_projected_probability():
    """
    Section 3.5.6: uncertainty-maximisation (Eq. 3.25-3.27) must preserve
    the projected probability distribution while pushing at least one
    belief mass to zero.
    """
    opinion = MultinomialOpinion(
        belief_masses={"x1": 0.30, "x2": 0.30, "x3": 0.20},
        uncertainty=0.20,
        base_rates={"x1": 0.40, "x2": 0.35, "x3": 0.25},
    )

    maximised = opinion.uncertainty_maximized()

    # u_max = min(P(xi)/a(xi)) = min(0.38/0.40, 0.37/0.35, 0.25/0.25) = 0.95
    assert math.isclose(maximised.uncertainty, 0.95, abs_tol=1e-6)

    # At least one belief mass should be (numerically) zero.
    assert any(math.isclose(b, 0.0, abs_tol=1e-9) for b in maximised.belief_masses.values())

    # Projected probability distribution must be unchanged.
    original_projection = opinion.projected_probabilities
    new_projection = maximised.projected_probabilities
    for x in opinion.domain:
        assert math.isclose(original_projection[x], new_projection[x], abs_tol=1e-9)


def test_vacuous_multinomial_opinion():
    base_rates = MultinomialOpinion.default_base_rates(["x1", "x2", "x3"])
    opinion = MultinomialOpinion.vacuous(base_rates=base_rates)

    assert opinion.is_vacuous
    for x in opinion.domain:
        assert math.isclose(opinion.belief_masses[x], 0.0, abs_tol=1e-9)


def test_binary_domain_is_rejected():
    """Definition 3.4 requires k > 2; binary domains should use BinomialOpinion."""
    with pytest.raises(ValueError):
        MultinomialOpinion(
            belief_masses={"x1": 0.5, "x2": 0.3},
            uncertainty=0.2,
            base_rates={"x1": 0.5, "x2": 0.5},
        )


def test_mismatched_domains_are_rejected():
    with pytest.raises(ValueError):
        MultinomialOpinion(
            belief_masses={"x1": 0.3, "x2": 0.3, "x3": 0.2},
            uncertainty=0.2,
            base_rates={"x1": 0.5, "x2": 0.3, "x4": 0.2},
        )


def test_additivity_constraint_is_enforced():
    with pytest.raises(ValueError):
        MultinomialOpinion(
            belief_masses={"x1": 0.5, "x2": 0.5, "x3": 0.5},
            uncertainty=0.5,
            base_rates={"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3},
        )