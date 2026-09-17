"""
Tests for unfusion and fission (Chapter 13), validated against Section
13.1.3's cumulative unfusion example and Table 13.1's cumulative
fission example (phi=0.75).
"""

import math

from subjective_logic import BinomialOpinion, MultinomialOpinion


def test_cumulative_unfusion_matches_book_example():
    """
    omega_x^(A CBF B) = (0.90, 0.05, 0.05, 0.50), omega_x^B = (0.70,
    0.10, 0.20, 0.50) -> omega_x^A = (0.91, 0.03, 0.06, 0.50).
    """
    fused = BinomialOpinion(belief=0.90, disbelief=0.05, uncertainty=0.05, base_rate=0.50)
    known_b = BinomialOpinion(belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.50)

    recovered_a = fused.unfuse_cumulative(known_b)

    assert math.isclose(recovered_a.belief, 0.91, abs_tol=0.005)
    assert math.isclose(recovered_a.disbelief, 0.03, abs_tol=0.005)
    assert math.isclose(recovered_a.uncertainty, 0.06, abs_tol=0.005)


def test_cumulative_unfusion_is_inverse_of_cumulative_fusion():
    """Round-trip property: fusing then unfusing with one contributor recovers the other exactly."""
    opinion_a = BinomialOpinion(belief=0.4, disbelief=0.3, uncertainty=0.3, base_rate=0.5)
    opinion_b = BinomialOpinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)

    fused = opinion_a.fuse_cumulative(opinion_b)
    recovered_a = fused.unfuse_cumulative(opinion_b)

    assert math.isclose(recovered_a.belief, opinion_a.belief, abs_tol=1e-9)
    assert math.isclose(recovered_a.disbelief, opinion_a.disbelief, abs_tol=1e-9)
    assert math.isclose(recovered_a.uncertainty, opinion_a.uncertainty, abs_tol=1e-9)


def test_averaging_unfusion_is_inverse_of_averaging_fusion():
    opinion_a = BinomialOpinion(belief=0.4, disbelief=0.3, uncertainty=0.3, base_rate=0.5)
    opinion_b = BinomialOpinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)

    fused = opinion_a.fuse_averaging(opinion_b)
    recovered_a = fused.unfuse_averaging(opinion_b)

    assert math.isclose(recovered_a.belief, opinion_a.belief, abs_tol=1e-9)
    assert math.isclose(recovered_a.uncertainty, opinion_a.uncertainty, abs_tol=1e-9)


def test_cumulative_fission_matches_table_13_1():
    """Table 13.1: ternary opinion, phi=0.75."""
    opinion = MultinomialOpinion(
        belief_masses={"x1": 0.20, "x2": 0.30, "x3": 0.40},
        uncertainty=0.10,
        base_rates={"x1": 0.10, "x2": 0.20, "x3": 0.70},
    )

    opinion_1, opinion_2 = opinion.split_cumulative(phi=0.75)

    assert math.isclose(opinion_1.belief_masses["x1"], 0.194, abs_tol=0.001)
    assert math.isclose(opinion_1.belief_masses["x2"], 0.290, abs_tol=0.001)
    assert math.isclose(opinion_1.belief_masses["x3"], 0.387, abs_tol=0.001)
    assert math.isclose(opinion_1.uncertainty, 0.129, abs_tol=0.001)

    assert math.isclose(opinion_2.belief_masses["x1"], 0.154, abs_tol=0.001)
    assert math.isclose(opinion_2.belief_masses["x2"], 0.230, abs_tol=0.001)
    assert math.isclose(opinion_2.belief_masses["x3"], 0.308, abs_tol=0.001)
    assert math.isclose(opinion_2.uncertainty, 0.308, abs_tol=0.001)


def test_fission_then_cumulative_fusion_recovers_original():
    """p. 241: omega_C1 (+) omega_C2 == omega_C, as expected."""
    opinion = MultinomialOpinion(
        belief_masses={"x1": 0.20, "x2": 0.30, "x3": 0.40},
        uncertainty=0.10,
        base_rates={"x1": 0.10, "x2": 0.20, "x3": 0.70},
    )

    opinion_1, opinion_2 = opinion.split_cumulative(phi=0.75)
    recombined = opinion_1.fuse_cumulative(opinion_2)

    for x in opinion.domain:
        assert math.isclose(recombined.belief_masses[x], opinion.belief_masses[x], abs_tol=1e-9)
    assert math.isclose(recombined.uncertainty, opinion.uncertainty, abs_tol=1e-9)


def test_fission_requires_phi_strictly_between_zero_and_one():
    import pytest

    opinion = MultinomialOpinion(belief_masses={"x1": 0.5, "x2": 0.3}, uncertainty=0.2, base_rates={"x1": 0.5, "x2": 0.5})
    with pytest.raises(ValueError):
        opinion.split_cumulative(phi=0.0)
    with pytest.raises(ValueError):
        opinion.split_cumulative(phi=1.0)