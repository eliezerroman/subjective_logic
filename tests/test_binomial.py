"""
Tests for BinomialOpinion, validated against the worked example from
Josang (2016), Section 3.4.3, Figures 3.1 and 3.2:

    omega_x = (0.40, 0.20, 0.40, 0.90)  <=>  Beta_e(px, 2.0, 1.0, 0.9)

with projected probability P(x) = 0.76.
"""

import math

import pytest

from subjective_logic import BinomialOpinion


def test_book_example_from_evidence():
    """Evidence (r=2.0, s=1.0, a=0.9) should map to the book's example opinion."""
    opinion = BinomialOpinion.from_evidence(r=2.0, s=1.0, base_rate=0.9)

    assert math.isclose(opinion.belief, 0.40, abs_tol=1e-6)
    assert math.isclose(opinion.disbelief, 0.20, abs_tol=1e-6)
    assert math.isclose(opinion.uncertainty, 0.40, abs_tol=1e-6)
    assert opinion.base_rate == 0.9


def test_book_example_projected_probability():
    """P(x) = b + a*u should match the book's stated result of 0.76."""
    opinion = BinomialOpinion(belief=0.40, disbelief=0.20, uncertainty=0.40, base_rate=0.90)
    assert math.isclose(opinion.projected_probability, 0.76, abs_tol=1e-6)


def test_evidence_round_trip():
    """Converting opinion -> evidence -> opinion should be the identity."""
    original = BinomialOpinion.from_evidence(r=2.0, s=1.0, base_rate=0.9)
    r, s = original.to_evidence()
    rebuilt = BinomialOpinion.from_evidence(r=r, s=s, base_rate=0.9)

    assert math.isclose(rebuilt.belief, original.belief, abs_tol=1e-9)
    assert math.isclose(rebuilt.disbelief, original.disbelief, abs_tol=1e-9)
    assert math.isclose(rebuilt.uncertainty, original.uncertainty, abs_tol=1e-9)


def test_vacuous_opinion_has_default_evidence_zero():
    """A vacuous opinion should correspond to zero evidence (r = s = 0)."""
    opinion = BinomialOpinion.vacuous(base_rate=0.5)
    r, s = opinion.to_evidence()

    assert math.isclose(r, 0.0, abs_tol=1e-9)
    assert math.isclose(s, 0.0, abs_tol=1e-9)
    assert opinion.is_vacuous


def test_dogmatic_opinion_cannot_be_converted_to_finite_evidence():
    """Dogmatic opinions (u=0) require infinite evidence (Section 3.2)."""
    opinion = BinomialOpinion(belief=0.7, disbelief=0.3, uncertainty=0.0, base_rate=0.5)
    assert opinion.is_dogmatic

    with pytest.raises(ValueError):
        opinion.to_evidence()


def test_additivity_constraint_is_enforced():
    """Opinions that violate b + d + u = 1 must be rejected (Eq. 3.1)."""
    with pytest.raises(ValueError):
        BinomialOpinion(belief=0.5, disbelief=0.5, uncertainty=0.5, base_rate=0.5)


def test_out_of_range_parameter_is_rejected():
    """All opinion parameters must lie in [0, 1]."""
    with pytest.raises(ValueError):
        BinomialOpinion(belief=1.5, disbelief=-0.5, uncertainty=0.0, base_rate=0.5)