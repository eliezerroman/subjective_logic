"""
Tests for HyperOpinion, validated against the "Scenario A" evidence
example in Josang (2016), Section 3.6.5, Table 3.3 (ternary domain,
100 observations): x3 observed 20 times, {x1, x2} observed 80 times,
everything else 0.
"""

import math

from subjective_logic import HyperOpinion, MultinomialOpinion


def _scenario_a_opinion() -> HyperOpinion:
    base_rates = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3}
    evidence = {
        frozenset({"x1"}): 0.0,
        frozenset({"x2"}): 0.0,
        frozenset({"x3"}): 20.0,
        frozenset({"x1", "x2"}): 80.0,
        frozenset({"x1", "x3"}): 0.0,
        frozenset({"x2", "x3"}): 0.0,
    }
    return HyperOpinion.from_evidence(evidence=evidence, base_rates=base_rates)


def test_scenario_a_projected_probabilities():
    """
    Hand-derived from Eq. 3.28 for Table 3.3's Scenario A:
    P(x1) = P(x2) = 61/153 ~ 0.39869, P(x3) = 31/153 ~ 0.20261.
    """
    opinion = _scenario_a_opinion()
    projected = opinion.projected_probabilities

    assert math.isclose(projected["x1"], 61 / 153, abs_tol=1e-6)
    assert math.isclose(projected["x2"], 61 / 153, abs_tol=1e-6)
    assert math.isclose(projected["x3"], 31 / 153, abs_tol=1e-6)


def test_projected_probabilities_sum_to_one():
    """Eq. 3.29.a: projected probability over the underlying domain X is additive."""
    opinion = _scenario_a_opinion()
    total = sum(opinion.projected_probabilities.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_projection_to_multinomial_preserves_projected_probability():
    """
    Book claim (p. 40, right after Eq. 3.30): P(omega_X) == P(omega'_X).
    Projecting a hyper-opinion to a multinomial one must not change the
    projected probability distribution.
    """
    hyper_opinion = _scenario_a_opinion()
    multinomial = hyper_opinion.to_multinomial()

    assert isinstance(multinomial, MultinomialOpinion)
    hyper_projection = hyper_opinion.projected_probabilities
    multinomial_projection = multinomial.projected_probabilities

    for x in hyper_opinion.domain:
        assert math.isclose(hyper_projection[x], multinomial_projection[x], abs_tol=1e-9)


def test_evidence_round_trip():
    original = _scenario_a_opinion()
    evidence = original.to_evidence()
    rebuilt = HyperOpinion.from_evidence(evidence=evidence, base_rates=dict(original.base_rates))

    for value in original.hyperdomain_values:
        assert math.isclose(rebuilt.belief_masses[value], original.belief_masses[value], abs_tol=1e-9)


def test_vagueness_is_belief_on_composites():
    opinion = _scenario_a_opinion()
    # All the belief mass on {x1, x2} (80/102) is vague, per Table 3.3.
    assert math.isclose(opinion.vagueness, 80 / 102, abs_tol=1e-9)


def test_binary_and_ternary_domains_rejected():
    """Definition 3.7 requires k > 2, same restriction as MultinomialOpinion."""
    import pytest

    base_rates = {"x1": 0.5, "x2": 0.5}
    with pytest.raises(ValueError):
        HyperOpinion.vacuous(base_rates=base_rates)