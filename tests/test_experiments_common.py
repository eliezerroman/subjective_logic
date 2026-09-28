"""
Tests for experiments/common.py, re-deriving Phase 0's key numeric
checks so this helper module stays covered by the same pytest suite as
the library itself.
"""

import math
import os
import sys

import pytest

# experiments/ is not an installable package; make it importable for tests only.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from common import credible_interval, standardized_sq_error  # noqa: E402

from subjective_logic import BinomialOpinion


def test_credible_interval_matches_direct_from_evidence():
    """For an opinion built directly by from_evidence, to_evidence() should round-trip exactly (Ch. 3),
    so the interval computed from it must match one computed from the original (r, s) directly."""
    opinion = BinomialOpinion.from_evidence(r=8.0, s=2.0, base_rate=0.5)
    r_eq, s_eq = opinion.to_evidence()

    lo, hi = credible_interval(r_eq, s_eq, base_rate=0.5, level=0.90)
    lo_direct, hi_direct = credible_interval(8.0, 2.0, base_rate=0.5, level=0.90)

    assert math.isclose(lo, lo_direct, abs_tol=1e-9)
    assert math.isclose(hi, hi_direct, abs_tol=1e-9)
    assert 0.0 <= lo < hi <= 1.0


def test_credible_interval_narrows_with_more_evidence():
    lo_small, hi_small = credible_interval(4.0, 1.0)
    lo_large, hi_large = credible_interval(40.0, 10.0)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_standardized_sq_error_is_zero_at_the_mean():
    """When p_true equals the opinion's own projected probability, the standardized error is 0."""
    opinion = BinomialOpinion.from_evidence(r=8.0, s=2.0, base_rate=0.5)
    assert math.isclose(standardized_sq_error(opinion, opinion.projected_probability), 0.0, abs_tol=1e-9)


def test_standardized_sq_error_grows_further_from_the_mean():
    opinion = BinomialOpinion.from_evidence(r=8.0, s=2.0, base_rate=0.5)
    near = standardized_sq_error(opinion, opinion.projected_probability + 0.01)
    far = standardized_sq_error(opinion, opinion.projected_probability + 0.10)
    assert far > near


def test_wrong_base_rate_is_overconfident_at_small_n():
    """
    Reproduces the core finding of Phase 0, Experiment 0.1: a wrong base
    rate (a=0.9) at small N produces a high MSSE (overconfidence),
    because the opinion trusts its prior more than the little evidence
    it has actually seen.
    """
    p_true = 0.7
    # N=2, both observations happen to be positive: r=2, s=0.
    wrong_prior = BinomialOpinion.from_evidence(r=2.0, s=0.0, base_rate=0.9)
    neutral_prior = BinomialOpinion.from_evidence(r=2.0, s=0.0, base_rate=0.5)

    mse_wrong = standardized_sq_error(wrong_prior, p_true)
    mse_neutral = standardized_sq_error(neutral_prior, p_true)

    assert mse_wrong > mse_neutral