"""
Tests for addition, subtraction and complement (Chapter 6), validated
against the worked examples in Josang (2016), pp. 96, 98, 99-100.
"""

import math

import pytest

from subjective_logic import BinomialOpinion, add, complement, subtract


def test_book_addition_example():
    """omega_x1 + omega_x2 = (0.30, 0.30, 0.40, 0.75), p. 96."""
    opinion_x1 = BinomialOpinion(belief=0.20, disbelief=0.40, uncertainty=0.40, base_rate=0.25)
    opinion_x2 = BinomialOpinion(belief=0.10, disbelief=0.50, uncertainty=0.40, base_rate=0.50)

    result = opinion_x1 + opinion_x2

    assert math.isclose(result.belief, 0.30, abs_tol=1e-6)
    assert math.isclose(result.disbelief, 0.30, abs_tol=1e-6)
    assert math.isclose(result.uncertainty, 0.40, abs_tol=1e-6)
    assert math.isclose(result.base_rate, 0.75, abs_tol=1e-6)


def test_addition_preserves_projected_probability_sum():
    """Eq. 6.2: P(x1Ux2) = P(x1) + P(x2)."""
    opinion_x1 = BinomialOpinion(belief=0.20, disbelief=0.40, uncertainty=0.40, base_rate=0.25)
    opinion_x2 = BinomialOpinion(belief=0.10, disbelief=0.50, uncertainty=0.40, base_rate=0.50)

    result = add(opinion_x1, opinion_x2)
    expected = opinion_x1.projected_probability + opinion_x2.projected_probability
    assert math.isclose(result.projected_probability, expected, abs_tol=1e-9)


def test_book_subtraction_example():
    """omega_union - omega_x2 = (0.20, 0.60, 0.20, 0.50), p. 98."""
    opinion_union = BinomialOpinion(belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.75)
    opinion_x2 = BinomialOpinion(belief=0.50, disbelief=0.30, uncertainty=0.20, base_rate=0.25)

    result = opinion_union - opinion_x2

    assert math.isclose(result.belief, 0.20, abs_tol=1e-6)
    assert math.isclose(result.disbelief, 0.60, abs_tol=1e-6)
    assert math.isclose(result.uncertainty, 0.20, abs_tol=1e-6)
    assert math.isclose(result.base_rate, 0.50, abs_tol=1e-6)


def test_subtraction_preserves_projected_probability_difference():
    """Eq. 6.5: P(x1) = P(x1Ux2) - P(x2)."""
    opinion_union = BinomialOpinion(belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.75)
    opinion_x2 = BinomialOpinion(belief=0.50, disbelief=0.30, uncertainty=0.20, base_rate=0.25)

    result = subtract(opinion_union, opinion_x2)
    expected = opinion_union.projected_probability - opinion_x2.projected_probability
    assert math.isclose(result.projected_probability, expected, abs_tol=1e-9)


def test_addition_then_subtraction_is_identity():
    """Subtraction is the inverse of addition (p. 97)."""
    opinion_x1 = BinomialOpinion(belief=0.20, disbelief=0.40, uncertainty=0.40, base_rate=0.25)
    opinion_x2 = BinomialOpinion(belief=0.10, disbelief=0.50, uncertainty=0.40, base_rate=0.50)

    union = opinion_x1 + opinion_x2
    recovered_x1 = union - opinion_x2

    assert math.isclose(recovered_x1.belief, opinion_x1.belief, abs_tol=1e-9)
    assert math.isclose(recovered_x1.disbelief, opinion_x1.disbelief, abs_tol=1e-9)
    assert math.isclose(recovered_x1.uncertainty, opinion_x1.uncertainty, abs_tol=1e-9)
    assert math.isclose(recovered_x1.base_rate, opinion_x1.base_rate, abs_tol=1e-9)


def test_book_complement_example():
    """complement of (0.50, 0.10, 0.40, 0.25) is (0.10, 0.50, 0.40, 0.75), p. 99-100."""
    opinion_x = BinomialOpinion(belief=0.50, disbelief=0.10, uncertainty=0.40, base_rate=0.25)

    result = ~opinion_x

    assert math.isclose(result.belief, 0.10, abs_tol=1e-6)
    assert math.isclose(result.disbelief, 0.50, abs_tol=1e-6)
    assert math.isclose(result.uncertainty, 0.40, abs_tol=1e-6)
    assert math.isclose(result.base_rate, 0.75, abs_tol=1e-6)


def test_complement_preserves_probability_complement():
    """Eq. 6.8: P(not x) = 1 - P(x)."""
    opinion_x = BinomialOpinion(belief=0.50, disbelief=0.10, uncertainty=0.40, base_rate=0.25)
    result = complement(opinion_x)
    assert math.isclose(result.projected_probability, 1.0 - opinion_x.projected_probability, abs_tol=1e-9)


def test_double_complement_is_identity():
    opinion_x = BinomialOpinion(belief=0.50, disbelief=0.10, uncertainty=0.40, base_rate=0.25)
    twice = ~(~opinion_x)
    assert math.isclose(twice.belief, opinion_x.belief, abs_tol=1e-9)
    assert math.isclose(twice.disbelief, opinion_x.disbelief, abs_tol=1e-9)
    assert math.isclose(twice.base_rate, opinion_x.base_rate, abs_tol=1e-9)


def test_addition_raises_on_zero_base_rate_sum():
    """Division by zero guard, Eq. 6.1."""
    opinion_x1 = BinomialOpinion(belief=0.3, disbelief=0.3, uncertainty=0.4, base_rate=0.0)
    opinion_x2 = BinomialOpinion(belief=0.2, disbelief=0.4, uncertainty=0.4, base_rate=0.0)
    with pytest.raises(ValueError):
        add(opinion_x1, opinion_x2)


def test_subtraction_raises_on_inconsistent_opinions():
    """Non-negativity constraint violation, Eq. 6.4."""
    opinion_union = BinomialOpinion(belief=0.5, disbelief=0.4, uncertainty=0.1, base_rate=0.6)
    opinion_x2 = BinomialOpinion(belief=0.05, disbelief=0.05, uncertainty=0.9, base_rate=0.4)
    with pytest.raises(ValueError):
        subtract(opinion_union, opinion_x2)