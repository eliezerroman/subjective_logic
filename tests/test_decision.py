"""
Tests for utility normalisation and decision criteria (Sections 4.3-4.4),
validated against the urn betting example in Josang (2016), pp. 61-62.
"""

import math

from subjective_logic import BinomialOpinion, DecisionOption, choose_best_option, utility_normalized_probability


def test_urn_example_utility_normalized_probability_is_equal():
    """
    Urn X: P(black) = 0.2, bet pays $1000. Urn Y: P(black) = 0.4, bet
    pays $500. Both should have utility-normalised probability 0.2.
    """
    urn_x = BinomialOpinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
    urn_y = BinomialOpinion(belief=0.4, disbelief=0.2, uncertainty=0.4, base_rate=0.5)

    # "Black" is the disbelief side of each "red" opinion.
    option_x = DecisionOption(
        label="X-black", sharp_belief_mass=urn_x.disbelief, vague_belief_mass=0.0,
        focal_uncertainty_mass=(1 - urn_x.base_rate) * urn_x.uncertainty, utility=1000.0,
    )
    option_y = DecisionOption(
        label="Y-black", sharp_belief_mass=urn_y.disbelief, vague_belief_mass=0.0,
        focal_uncertainty_mass=(1 - urn_y.base_rate) * urn_y.uncertainty, utility=500.0,
    )

    assert math.isclose(option_x.projected_probability, 0.2, abs_tol=1e-9)
    assert math.isclose(option_y.projected_probability, 0.4, abs_tol=1e-9)

    max_abs_utility = 1000.0
    assert math.isclose(utility_normalized_probability(option_x, max_abs_utility), 0.2, abs_tol=1e-9)
    assert math.isclose(utility_normalized_probability(option_y, max_abs_utility), 0.2, abs_tol=1e-9)


def test_urn_example_tie_broken_by_sharp_belief_mass():
    """With equal utility-normalised probability, option Y wins on greater raw sharp belief mass (p. 62)."""
    option_x = DecisionOption(label="X-black", sharp_belief_mass=0.1, vague_belief_mass=0.0, focal_uncertainty_mass=0.1, utility=1000.0)
    option_y = DecisionOption(label="Y-black", sharp_belief_mass=0.2, vague_belief_mass=0.0, focal_uncertainty_mass=0.2, utility=500.0)

    best = choose_best_option([option_x, option_y])
    assert best is not None
    assert best.label == "Y-black"


def test_choose_best_option_clear_winner():
    strong = DecisionOption(label="strong", sharp_belief_mass=0.8, vague_belief_mass=0.0, focal_uncertainty_mass=0.1)
    weak = DecisionOption(label="weak", sharp_belief_mass=0.2, vague_belief_mass=0.0, focal_uncertainty_mass=0.1)

    assert choose_best_option([strong, weak]).label == "strong"


def test_choose_best_option_genuine_tie_returns_none():
    a = DecisionOption(label="a", sharp_belief_mass=0.5, vague_belief_mass=0.0, focal_uncertainty_mass=0.2)
    b = DecisionOption(label="b", sharp_belief_mass=0.5, vague_belief_mass=0.0, focal_uncertainty_mass=0.2)

    assert choose_best_option([a, b]) is None