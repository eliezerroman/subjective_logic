"""
Tests for multinomial multiplication and division (Chapter 8),
validated against the egg gender/mutation example (Josang, 2016,
Section 8.2, Tables 8.2-8.3, pp. 125-127).
"""

import math

from subjective_logic import MultinomialOpinion, averaging_proportional_divide, normal_multiply, proportional_multiply, selective_divide


def _gender_opinion() -> MultinomialOpinion:
    return MultinomialOpinion(
        belief_masses={"M": 0.60, "F": 0.30}, uncertainty=0.10, base_rates={"M": 0.50, "F": 0.50}
    )


def _mutation_opinion() -> MultinomialOpinion:
    return MultinomialOpinion(
        belief_masses={"y1": 0.70, "y2": 0.20}, uncertainty=0.10, base_rates={"y1": 0.50, "y2": 0.50}
    )


def test_normal_product_matches_book_exactly():
    """Table 8.3 'Normal product' row -- exact match, no rounding needed."""
    result = normal_multiply(_gender_opinion(), _mutation_opinion())

    assert math.isclose(result.belief_masses[("M", "y1")], 0.460, abs_tol=1e-9)
    assert math.isclose(result.belief_masses[("M", "y2")], 0.135, abs_tol=1e-9)
    assert math.isclose(result.belief_masses[("F", "y1")], 0.235, abs_tol=1e-9)
    assert math.isclose(result.belief_masses[("F", "y2")], 0.060, abs_tol=1e-9)
    assert math.isclose(result.uncertainty, 0.110, abs_tol=1e-9)


def test_proportional_product_matches_book_exactly():
    """Table 8.3 'Proportional product' row -- exact match."""
    result = proportional_multiply(_gender_opinion(), _mutation_opinion())

    assert math.isclose(result.belief_masses[("M", "y1")], 0.473, abs_tol=0.001)
    assert math.isclose(result.belief_masses[("M", "y2")], 0.148, abs_tol=0.001)
    assert math.isclose(result.belief_masses[("F", "y1")], 0.248, abs_tol=0.001)
    assert math.isclose(result.belief_masses[("F", "y2")], 0.073, abs_tol=0.001)
    assert math.isclose(result.uncertainty, 0.058, abs_tol=0.001)


def test_both_methods_preserve_exact_projected_probability():
    """Eq. 8.8: P_XY(x,y) = P_X(x) * P_Y(y), exact regardless of method."""
    gender, mutation = _gender_opinion(), _mutation_opinion()
    pg, pm = gender.projected_probabilities, mutation.projected_probabilities

    for product in (normal_multiply(gender, mutation), proportional_multiply(gender, mutation)):
        projected = product.projected_probabilities
        for g in gender.domain:
            for m in mutation.domain:
                assert math.isclose(projected[(g, m)], pg[g] * pm[m], abs_tol=1e-9)


def test_normal_product_preserves_more_uncertainty_than_proportional():
    """p. 127: 'The normal product preserves the most uncertainty, the proportional product... about 50% less.'"""
    gender, mutation = _gender_opinion(), _mutation_opinion()
    normal = normal_multiply(gender, mutation)
    proportional = proportional_multiply(gender, mutation)
    assert normal.uncertainty > proportional.uncertainty


def test_averaging_division_recovers_factor_exactly_when_product_is_genuine():
    """
    When opinion_xy really is the product of opinion_x and opinion_y,
    P_XY(x,y)/P_Y(y) = P_X(x) for every y (constant), so averaging
    identical values recovers P_X exactly (Eq. 8.49).
    """
    gender, mutation = _gender_opinion(), _mutation_opinion()
    product = normal_multiply(gender, mutation)

    recovered = averaging_proportional_divide(product, mutation)
    recovered_projected = recovered.projected_probabilities
    gender_projected = gender.projected_probabilities

    for g in gender.domain:
        assert math.isclose(recovered_projected[g], gender_projected[g], abs_tol=1e-9)


def test_selective_division_extracts_joint_column():
    """
    Eq. 8.58: with a joint opinion whose entire mass sits on the
    observed column (consistent with that value being certain),
    selective division extracts it exactly.
    """
    joint = MultinomialOpinion(
        belief_masses={("x1", "y1"): 0.6, ("x1", "y2"): 0.0, ("x2", "y1"): 0.4, ("x2", "y2"): 0.0},
        uncertainty=0.0,
        base_rates={("x1", "y1"): 0.4, ("x1", "y2"): 0.1, ("x2", "y1"): 0.4, ("x2", "y2"): 0.1},
    )
    y_observed = MultinomialOpinion(
        belief_masses={"y1": 1.0, "y2": 0.0}, uncertainty=0.0, base_rates={"y1": 0.8, "y2": 0.2}
    )

    result = selective_divide(joint, y_observed, observed_value="y1")

    assert math.isclose(result.belief_masses["x1"], 0.6, abs_tol=1e-9)
    assert math.isclose(result.belief_masses["x2"], 0.4, abs_tol=1e-9)
    assert math.isclose(result.uncertainty, 0.0, abs_tol=1e-9)


def test_selective_division_rejects_inconsistent_joint():
    """
    If the joint opinion's marginal for observed_value isn't 1, the
    joint is not consistent with that value being certain -- must raise
    rather than return an invalid (non-additive) opinion.
    """
    import pytest

    joint = normal_multiply(_gender_opinion(), _mutation_opinion())  # marginal for y1 is 0.75, not 1
    y_observed = MultinomialOpinion(
        belief_masses={"y1": 1.0, "y2": 0.0}, uncertainty=0.0, base_rates={"y1": 0.5, "y2": 0.5}
    )
    with pytest.raises(ValueError):
        selective_divide(joint, y_observed, observed_value="y1")