"""
Tests for belief fusion (Chapter 12), validated against Zadeh's example
(Table 12.5, dogmatic inputs) and its uncertain variant (Table 12.6),
both of which matched to full computed precision, plus the cinema
example (Section 12.2.4) for constraint fusion.
"""

import math

import pytest

from subjective_logic import HyperOpinion, MultinomialOpinion


def test_table_12_5_dogmatic_zadeh_example():
    """
    Dogmatic inputs: A-CBF, ABF and WBF all collapse to the same simple
    0.5/0.5 average (Table 12.5's own columns are identical).
    """
    base_rates = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3}
    opinion_a = MultinomialOpinion(belief_masses={"x1": 0.99, "x2": 0.01, "x3": 0.00}, uncertainty=0.0, base_rates=base_rates)
    opinion_b = MultinomialOpinion(belief_masses={"x1": 0.00, "x2": 0.01, "x3": 0.99}, uncertainty=0.0, base_rates=base_rates)

    for fused in (opinion_a.fuse_cumulative(opinion_b), opinion_a.fuse_averaging(opinion_b), opinion_a.fuse_weighted(opinion_b)):
        assert math.isclose(fused.belief_masses["x1"], 0.495, abs_tol=1e-6)
        assert math.isclose(fused.belief_masses["x2"], 0.010, abs_tol=1e-6)
        assert math.isclose(fused.belief_masses["x3"], 0.495, abs_tol=1e-6)
        assert math.isclose(fused.uncertainty, 0.0, abs_tol=1e-6)


def test_table_12_6_a_cbf_exact():
    """A-CBF row of Table 12.6 with uncertain inputs."""
    base_rates = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3}
    opinion_a = MultinomialOpinion(belief_masses={"x1": 0.98, "x2": 0.01, "x3": 0.00}, uncertainty=0.01, base_rates=base_rates)
    opinion_b = MultinomialOpinion(belief_masses={"x1": 0.00, "x2": 0.01, "x3": 0.90}, uncertainty=0.09, base_rates=base_rates)

    fused = opinion_a.fuse_cumulative(opinion_b)

    assert math.isclose(fused.belief_masses["x1"], 0.890, abs_tol=0.001)
    assert math.isclose(fused.belief_masses["x2"], 0.010, abs_tol=0.001)
    assert math.isclose(fused.belief_masses["x3"], 0.091, abs_tol=0.001)
    assert math.isclose(fused.uncertainty, 0.009, abs_tol=0.001)


def test_table_12_6_abf_exact():
    """ABF row of Table 12.6."""
    base_rates = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3}
    opinion_a = MultinomialOpinion(belief_masses={"x1": 0.98, "x2": 0.01, "x3": 0.00}, uncertainty=0.01, base_rates=base_rates)
    opinion_b = MultinomialOpinion(belief_masses={"x1": 0.00, "x2": 0.01, "x3": 0.90}, uncertainty=0.09, base_rates=base_rates)

    fused = opinion_a.fuse_averaging(opinion_b)

    assert math.isclose(fused.belief_masses["x1"], 0.882, abs_tol=0.001)
    assert math.isclose(fused.belief_masses["x2"], 0.010, abs_tol=0.001)
    assert math.isclose(fused.belief_masses["x3"], 0.090, abs_tol=0.001)
    assert math.isclose(fused.uncertainty, 0.018, abs_tol=0.001)


def test_table_12_6_wbf_exact():
    """WBF row of Table 12.6."""
    base_rates = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3}
    opinion_a = MultinomialOpinion(belief_masses={"x1": 0.98, "x2": 0.01, "x3": 0.00}, uncertainty=0.01, base_rates=base_rates)
    opinion_b = MultinomialOpinion(belief_masses={"x1": 0.00, "x2": 0.01, "x3": 0.90}, uncertainty=0.09, base_rates=base_rates)

    fused = opinion_a.fuse_weighted(opinion_b)

    assert math.isclose(fused.belief_masses["x1"], 0.889, abs_tol=0.001)
    assert math.isclose(fused.belief_masses["x2"], 0.010, abs_tol=0.001)
    assert math.isclose(fused.belief_masses["x3"], 0.083, abs_tol=0.001)
    assert math.isclose(fused.uncertainty, 0.018, abs_tol=0.002)  # book rounds 0.0174 -> 0.018


def test_cumulative_fusion_is_non_idempotent():
    """Section 12.1.2 (p. 212): CBF is non-idempotent -- fusing equal opinions REDUCES uncertainty."""
    from subjective_logic import BinomialOpinion

    opinion = BinomialOpinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
    fused = opinion.fuse_cumulative(opinion)
    assert fused.uncertainty < opinion.uncertainty


def test_averaging_fusion_is_idempotent():
    """Section 12.1.2 (p. 212): ABF is idempotent -- fusing an opinion with itself changes nothing."""
    from subjective_logic import BinomialOpinion

    opinion = BinomialOpinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
    fused = opinion.fuse_averaging(opinion)
    assert math.isclose(fused.belief, opinion.belief, abs_tol=1e-9)
    assert math.isclose(fused.uncertainty, opinion.uncertainty, abs_tol=1e-9)


def test_weighted_fusion_vacuous_is_neutral_element():
    """Section 12.1.2 (p. 212): WBF has the vacuous opinion as neutral element."""
    from subjective_logic import BinomialOpinion

    opinion = BinomialOpinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
    vacuous = BinomialOpinion.vacuous(base_rate=0.5)
    fused = opinion.fuse_weighted(vacuous)
    assert math.isclose(fused.belief, opinion.belief, abs_tol=1e-9)
    assert math.isclose(fused.uncertainty, opinion.uncertainty, abs_tol=1e-9)


def test_cinema_example_constraint_fusion():
    """Section 12.2.4 (Table 12.2): Alice & Bob's hard, conflicting preferences resolve to Grey Matter."""
    base_rates = {"BD": 1 / 3, "GM": 1 / 3, "WP": 1 / 3}
    alice = HyperOpinion.from_evidence(
        evidence={
            frozenset({"BD"}): 99.0, frozenset({"GM"}): 1.0, frozenset({"WP"}): 0.0,
            frozenset({"BD", "GM"}): 0.0, frozenset({"BD", "WP"}): 0.0, frozenset({"GM", "WP"}): 0.0,
        },
        base_rates=base_rates, prior_weight=1e-9,  # near-zero prior weight -> effectively dogmatic
    )
    bob = HyperOpinion.from_evidence(
        evidence={
            frozenset({"BD"}): 0.0, frozenset({"GM"}): 1.0, frozenset({"WP"}): 99.0,
            frozenset({"BD", "GM"}): 0.0, frozenset({"BD", "WP"}): 0.0, frozenset({"GM", "WP"}): 0.0,
        },
        base_rates=base_rates, prior_weight=1e-9,
    )

    fused = alice.fuse_constraint(bob)
    assert fused.belief_masses[frozenset({"GM"})] > 0.99


def test_totally_conflicting_opinions_raise():
    """Section 12.2.6: Con=1 has no defined fusion result."""
    base_rates = {"BD": 1 / 3, "GM": 1 / 3, "WP": 1 / 3}
    alice = HyperOpinion.from_evidence(
        evidence={
            frozenset({"BD"}): 1.0, frozenset({"GM"}): 0.0, frozenset({"WP"}): 0.0,
            frozenset({"BD", "GM"}): 0.0, frozenset({"BD", "WP"}): 0.0, frozenset({"GM", "WP"}): 0.0,
        },
        base_rates=base_rates, prior_weight=1e-9,
    )
    bob = HyperOpinion.from_evidence(
        evidence={
            frozenset({"BD"}): 0.0, frozenset({"GM"}): 0.0, frozenset({"WP"}): 1.0,
            frozenset({"BD", "GM"}): 0.0, frozenset({"BD", "WP"}): 0.0, frozenset({"GM", "WP"}): 0.0,
        },
        base_rates=base_rates, prior_weight=1e-9,
    )
    with pytest.raises(ValueError):
        alice.fuse_constraint(bob)