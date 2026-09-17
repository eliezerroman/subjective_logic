"""
Tests for Bayesian reputation systems (Chapter 16), validated against
Figure 16.2's worked example (point estimates of average vs. polarised
ratings, both 0.5, p. 298) and the geometric-series convergence
identity (Eq. 16.7-16.8).
"""

import math

import pytest

from subjective_logic import (
    MultinomialOpinion,
    aggregate_with_decay,
    community_base_rate,
    convergence_value,
    point_estimate,
)

LEVELS = ["L1", "L2", "L3", "L4", "L5"]  # Bad, Mediocre, Average, Good, Excellent


def _score(ratings):
    base_rate = {level: 0.2 for level in LEVELS}
    return MultinomialOpinion.from_evidence(ratings, base_rate).projected_probabilities


def test_average_ratings_point_estimate_matches_figure_16_2a():
    """10 average (L3) ratings: point estimate = 0.5, Figure 16.2.a."""
    score = _score({"L1": 0.0, "L2": 0.0, "L3": 10.0, "L4": 0.0, "L5": 0.0})
    sigma = point_estimate(score, LEVELS)
    assert math.isclose(sigma, 0.5, abs_tol=1e-6)


def test_polarised_ratings_point_estimate_matches_figure_16_2b():
    """5 bad + 5 excellent ratings: point estimate is ALSO 0.5, despite very different distribution (p. 298)."""
    score = _score({"L1": 5.0, "L2": 0.0, "L3": 0.0, "L4": 0.0, "L5": 5.0})
    sigma = point_estimate(score, LEVELS)
    assert math.isclose(sigma, 0.5, abs_tol=1e-6)


def test_point_estimate_loses_information_between_the_two_scenarios():
    """The whole point of Section 16.4.2's caveat: identical point estimate, very different score vectors."""
    score_average = _score({"L1": 0.0, "L2": 0.0, "L3": 10.0, "L4": 0.0, "L5": 0.0})
    score_polarised = _score({"L1": 5.0, "L2": 0.0, "L3": 0.0, "L4": 0.0, "L5": 5.0})

    assert math.isclose(point_estimate(score_average, LEVELS), point_estimate(score_polarised, LEVELS), abs_tol=1e-6)
    assert not math.isclose(score_average["L3"], score_polarised["L3"], abs_tol=0.1)


def test_decay_converges_to_geometric_series_value():
    """Eq. 16.8: R_infinity = e/(1-lambda) for a constant per-period rating."""
    constant_rating = {"L1": 1.0, "L2": 0.0}
    longevity = 0.9

    accumulated = {"L1": 0.0, "L2": 0.0}
    for _ in range(2000):  # enough periods to be well within float precision of the limit
        accumulated = aggregate_with_decay(accumulated, constant_rating, longevity)

    expected = convergence_value(constant_rating, longevity)
    assert math.isclose(accumulated["L1"], expected["L1"], rel_tol=1e-6)
    assert math.isclose(expected["L1"], 10.0, abs_tol=1e-6)  # 1 / (1 - 0.9) = 10


def test_convergence_value_requires_longevity_below_one():
    with pytest.raises(ValueError):
        convergence_value({"L1": 1.0}, longevity=1.0)


def test_community_base_rate_reflects_aggregate_quality():
    """Definition 16.1: the community base rate is the aggregate community's own score."""
    service_y = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "L4": 0.0, "L5": 10.0}  # excellent
    service_z = {"L1": 10.0, "L2": 0.0, "L3": 0.0, "L4": 0.0, "L5": 0.0}  # bad

    base_rate = community_base_rate([service_y, service_z])

    # The community is evenly split between excellent and bad -> base rate should favour those two levels equally.
    assert math.isclose(base_rate["L1"], base_rate["L5"], abs_tol=1e-6)
    assert base_rate["L1"] > base_rate["L3"]