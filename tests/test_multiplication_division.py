"""
Tests for binomial multiplication, comultiplication, division and
codivision (Chapter 7), validated against the worked examples in
Josang (2016), pp. 102-113.
"""

import math

import pytest

from subjective_logic import BinomialOpinion, codivide, comultiply, divide, multiply


def test_book_multiplication_example():
    """omega_x . omega_y = (0.15, 0.15, 0.70, 0.10), p. 102-103."""
    opinion_x = BinomialOpinion(belief=0.75, disbelief=0.15, uncertainty=0.10, base_rate=0.50)
    opinion_y = BinomialOpinion(belief=0.10, disbelief=0.00, uncertainty=0.90, base_rate=0.20)

    result = opinion_x * opinion_y

    assert math.isclose(result.belief, 0.15, abs_tol=0.01)
    assert math.isclose(result.disbelief, 0.15, abs_tol=1e-6)
    assert math.isclose(result.uncertainty, 0.70, abs_tol=0.01)
    assert math.isclose(result.base_rate, 0.10, abs_tol=1e-6)


def test_multiplication_preserves_projected_probability_product():
    """Eq. 7.2: P(x AND y) = P(x) * P(y), exact (unlike belief/disbelief/uncertainty, which are approximate)."""
    opinion_x = BinomialOpinion(belief=0.75, disbelief=0.15, uncertainty=0.10, base_rate=0.50)
    opinion_y = BinomialOpinion(belief=0.10, disbelief=0.00, uncertainty=0.90, base_rate=0.20)

    result = multiply(opinion_x, opinion_y)
    expected = opinion_x.projected_probability * opinion_y.projected_probability
    assert math.isclose(result.projected_probability, expected, abs_tol=1e-9)


def test_book_comultiplication_example():
    """omega_x (x) omega_y = (0.84, 0.06, 0.10, 0.60), p. 103-104."""
    opinion_x = BinomialOpinion(belief=0.75, disbelief=0.15, uncertainty=0.10, base_rate=0.50)
    opinion_y = BinomialOpinion(belief=0.35, disbelief=0.00, uncertainty=0.65, base_rate=0.20)

    result = opinion_x | opinion_y

    assert math.isclose(result.belief, 0.84, abs_tol=0.01)
    assert math.isclose(result.disbelief, 0.06, abs_tol=0.01)  # book rounds 0.065 -> 0.06
    assert math.isclose(result.uncertainty, 0.10, abs_tol=0.01)
    assert math.isclose(result.base_rate, 0.60, abs_tol=1e-6)


def test_comultiplication_preserves_projected_probability_or():
    """Eq. 7.4: P(x OR y) = P(x) + P(y) - P(x)*P(y), exact."""
    opinion_x = BinomialOpinion(belief=0.75, disbelief=0.15, uncertainty=0.10, base_rate=0.50)
    opinion_y = BinomialOpinion(belief=0.35, disbelief=0.00, uncertainty=0.65, base_rate=0.20)

    result = comultiply(opinion_x, opinion_y)
    px, py = opinion_x.projected_probability, opinion_y.projected_probability
    expected = px + py - px * py
    assert math.isclose(result.projected_probability, expected, abs_tol=1e-9)


def test_book_division_example():
    """omega(x AND y) / omega_y = (0.15, 0.80, 0.05, 0.40), p. 110-111."""
    opinion_xy = BinomialOpinion(belief=0.10, disbelief=0.80, uncertainty=0.10, base_rate=0.20)
    opinion_y = BinomialOpinion(belief=0.40, disbelief=0.00, uncertainty=0.60, base_rate=0.50)

    result = opinion_xy / opinion_y

    assert math.isclose(result.belief, 0.15, abs_tol=0.01)
    assert math.isclose(result.disbelief, 0.80, abs_tol=1e-6)
    assert math.isclose(result.uncertainty, 0.05, abs_tol=0.01)
    assert math.isclose(result.base_rate, 0.40, abs_tol=1e-6)


def test_multiplication_then_division_is_identity():
    """Division is the inverse of multiplication (p. 110)."""
    opinion_x = BinomialOpinion(belief=0.6, disbelief=0.2, uncertainty=0.2, base_rate=0.3)
    opinion_y = BinomialOpinion(belief=0.4, disbelief=0.1, uncertainty=0.5, base_rate=0.5)

    product = multiply(opinion_x, opinion_y)
    recovered_x = divide(product, opinion_y)

    assert math.isclose(recovered_x.belief, opinion_x.belief, abs_tol=1e-6)
    assert math.isclose(recovered_x.disbelief, opinion_x.disbelief, abs_tol=1e-6)
    assert math.isclose(recovered_x.uncertainty, opinion_x.uncertainty, abs_tol=1e-6)


def test_book_codivision_example():
    """omega(x OR y) [\\] omega_y = (0.05, 0.49, 0.46, 0.50), p. 112-113."""
    opinion_xy = BinomialOpinion(belief=0.05, disbelief=0.55, uncertainty=0.40, base_rate=0.75)
    opinion_y = BinomialOpinion(belief=0.00, disbelief=0.80, uncertainty=0.20, base_rate=0.50)

    result = codivide(opinion_xy, opinion_y)

    assert math.isclose(result.belief, 0.05, abs_tol=1e-6)
    assert math.isclose(result.disbelief, 0.49, abs_tol=0.01)
    assert math.isclose(result.uncertainty, 0.46, abs_tol=0.01)
    assert math.isclose(result.base_rate, 0.50, abs_tol=1e-6)


def test_dogmatic_multiplication_matches_plain_probability():
    """
    Section 7.4 (p. 114): for dogmatic opinions (u=0), multiplication is
    homomorphic to plain probabilistic AND, regardless of base rate.
    """
    opinion_x = BinomialOpinion(belief=0.7, disbelief=0.3, uncertainty=0.0, base_rate=0.9)
    opinion_y = BinomialOpinion(belief=0.4, disbelief=0.6, uncertainty=0.0, base_rate=0.1)

    result = opinion_x * opinion_y

    assert math.isclose(result.belief, 0.7 * 0.4, abs_tol=1e-9)
    assert math.isclose(result.uncertainty, 0.0, abs_tol=1e-9)


def test_division_raises_when_constraints_violated():
    """Eq. 7.13 requires ax < ay."""
    opinion_xy = BinomialOpinion(belief=0.1, disbelief=0.8, uncertainty=0.1, base_rate=0.6)
    opinion_y = BinomialOpinion(belief=0.4, disbelief=0.0, uncertainty=0.6, base_rate=0.5)
    with pytest.raises(ValueError):
        divide(opinion_xy, opinion_y)


def test_codivision_raises_when_constraints_violated():
    """Eq. 7.16 requires ax > ay."""
    opinion_xy = BinomialOpinion(belief=0.05, disbelief=0.55, uncertainty=0.40, base_rate=0.30)
    opinion_y = BinomialOpinion(belief=0.00, disbelief=0.80, uncertainty=0.20, base_rate=0.50)
    with pytest.raises(ValueError):
        codivide(opinion_xy, opinion_y)


def test_reliability_analysis_exact_projected_probability():
    """
    Section 7.2.1 reliability example (Table 7.1, p. 108): system
    S = w AND (x OR y) AND z. Regardless of belief/uncertainty
    approximation, the projected probability composition must be exact
    (Eq. 7.11), since P is always exact for multiply/comultiply.
    """
    opinion_w = BinomialOpinion(belief=0.90, disbelief=0.10, uncertainty=0.00, base_rate=0.90)
    opinion_x = BinomialOpinion(belief=0.50, disbelief=0.00, uncertainty=0.50, base_rate=0.80)
    opinion_y = BinomialOpinion(belief=0.00, disbelief=0.00, uncertainty=1.00, base_rate=0.80)
    opinion_z = BinomialOpinion(belief=0.00, disbelief=0.00, uncertainty=1.00, base_rate=0.90)

    x_or_y = comultiply(opinion_x, opinion_y)
    system = multiply(multiply(opinion_w, x_or_y), opinion_z)

    px, py, pw, pz = (
        opinion_x.projected_probability,
        opinion_y.projected_probability,
        opinion_w.projected_probability,
        opinion_z.projected_probability,
    )
    expected = pw * (px + py - px * py) * pz

    assert math.isclose(system.projected_probability, expected, abs_tol=1e-9)
    # Sanity check against the book's own (rounded) system reliability of ~0.80.
    assert math.isclose(system.projected_probability, 0.80, abs_tol=0.02)