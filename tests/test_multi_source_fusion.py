"""
Tests for N-source averaging fusion (multi_source_averaging_fusion),
validated against: (1) exact agreement with pairwise fuse_averaging at
N=2, (2) order-independence for N=3 and N=4, and (3) reproducing the
order-dependence BUG this operator was built to fix, so the bug itself
stays covered by a regression test.
"""

import math

from subjective_logic import BinomialOpinion, MultinomialOpinion


def test_matches_pairwise_at_n_equals_2():
    a = BinomialOpinion(belief=0.4, disbelief=0.3, uncertainty=0.3, base_rate=0.5)
    b = BinomialOpinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)

    pairwise = a.fuse_averaging(b)
    multi = a.fuse_averaging_multi(b)

    assert math.isclose(pairwise.belief, multi.belief, abs_tol=1e-9)
    assert math.isclose(pairwise.disbelief, multi.disbelief, abs_tol=1e-9)
    assert math.isclose(pairwise.uncertainty, multi.uncertainty, abs_tol=1e-9)


def test_order_independent_for_four_sources():
    base_rates = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3}
    ops = [
        MultinomialOpinion.from_evidence({"x1": 7.0, "x2": 2.0, "x3": 1.0}, base_rates),
        MultinomialOpinion.from_evidence({"x1": 1.0, "x2": 8.0, "x3": 1.0}, base_rates),
        MultinomialOpinion.from_evidence({"x1": 2.0, "x2": 2.0, "x3": 6.0}, base_rates),
        MultinomialOpinion.from_evidence({"x1": 0.0, "x2": 0.0, "x3": 10.0}, base_rates),
    ]

    forward = ops[0].fuse_averaging_multi(*ops[1:])
    reversed_order = ops[3].fuse_averaging_multi(*ops[2::-1])

    for x in base_rates:
        assert math.isclose(forward.belief_masses[x], reversed_order.belief_masses[x], abs_tol=1e-9)
    assert math.isclose(forward.uncertainty, reversed_order.uncertainty, abs_tol=1e-9)


def test_chained_pairwise_averaging_is_order_dependent_regression():
    """
    Regression test documenting the bug that motivated this module
    (Phase 2, Experiment 2.1): chaining the pairwise fuse_averaging
    operator gives DIFFERENT results depending on order, unlike
    fuse_averaging_multi.
    """
    base_rates = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3}
    ops = [
        MultinomialOpinion.from_evidence({"x1": 7.0, "x2": 2.0, "x3": 1.0}, base_rates),
        MultinomialOpinion.from_evidence({"x1": 1.0, "x2": 8.0, "x3": 1.0}, base_rates),
        MultinomialOpinion.from_evidence({"x1": 0.0, "x2": 0.0, "x3": 10.0}, base_rates),
    ]

    chained_forward = ops[0].fuse_averaging(ops[1]).fuse_averaging(ops[2])
    chained_reverse = ops[2].fuse_averaging(ops[1]).fuse_averaging(ops[0])

    assert not math.isclose(chained_forward.belief_masses["x3"], chained_reverse.belief_masses["x3"], abs_tol=1e-6)


def test_equal_uncertainty_sources_reduce_to_plain_average():
    """When every source shares the same uncertainty, the result is a plain arithmetic mean of belief masses."""
    base_rates = {"x1": 0.5, "x2": 0.5}
    a = MultinomialOpinion.from_evidence({"x1": 7.0, "x2": 3.0}, base_rates)   # u = 2/12
    b = MultinomialOpinion.from_evidence({"x1": 1.0, "x2": 9.0}, base_rates)   # u = 2/12
    c = MultinomialOpinion.from_evidence({"x1": 5.0, "x2": 5.0}, base_rates)   # u = 2/12

    fused = a.fuse_averaging_multi(b, c)
    expected = {x: (a.belief_masses[x] + b.belief_masses[x] + c.belief_masses[x]) / 3 for x in base_rates}

    for x in base_rates:
        assert math.isclose(fused.belief_masses[x], expected[x], abs_tol=1e-9)
    assert math.isclose(fused.uncertainty, a.uncertainty, abs_tol=1e-9)


def test_dogmatic_source_dominates():
    """A single dogmatic (u=0) source among several uncertain ones fully determines the result."""
    base_rates = {"x1": 0.5, "x2": 0.5}
    dogmatic = MultinomialOpinion(belief_masses={"x1": 0.9, "x2": 0.1}, uncertainty=0.0, base_rates=base_rates)
    uncertain_a = MultinomialOpinion.from_evidence({"x1": 1.0, "x2": 1.0}, base_rates)
    uncertain_b = MultinomialOpinion.from_evidence({"x1": 5.0, "x2": 0.0}, base_rates)

    fused = dogmatic.fuse_averaging_multi(uncertain_a, uncertain_b)

    assert math.isclose(fused.belief_masses["x1"], 0.9, abs_tol=1e-9)
    assert math.isclose(fused.uncertainty, 0.0, abs_tol=1e-9)