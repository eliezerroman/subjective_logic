"""
Tests for the probabilistic opinion notation (Section 3.7.1).
"""

import math

from subjective_logic import BinomialOpinion, MultinomialOpinion


def test_binomial_probabilistic_notation_round_trip():
    original = BinomialOpinion.from_evidence(r=2.0, s=1.0, base_rate=0.9)
    p, u, a = original.to_probabilistic_notation()
    rebuilt = BinomialOpinion.from_probabilistic_notation(p, u, a)

    assert math.isclose(rebuilt.belief, original.belief, abs_tol=1e-9)
    assert math.isclose(rebuilt.disbelief, original.disbelief, abs_tol=1e-9)
    assert math.isclose(rebuilt.uncertainty, original.uncertainty, abs_tol=1e-9)


def test_multinomial_probabilistic_notation_round_trip():
    base_rates = MultinomialOpinion.default_base_rates(["x1", "x2", "x3"])
    original = MultinomialOpinion.from_evidence(
        evidence={"x1": 6.0, "x2": 1.0, "x3": 1.0}, base_rates=base_rates
    )
    p, u, a = original.to_probabilistic_notation()
    rebuilt = MultinomialOpinion.from_probabilistic_notation(p, u, a)

    for x in original.domain:
        assert math.isclose(rebuilt.belief_masses[x], original.belief_masses[x], abs_tol=1e-9)