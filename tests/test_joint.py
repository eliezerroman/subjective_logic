"""
Tests for joint and marginal opinions (Chapter 11), validated against
the Match-Fixing Revisited example (Section 11.4, reusing the same
conditionals already validated in test_deduction.py's Match-Fixing
test). Row/column sum checks are used instead of cell-by-cell matching
against Eq. 11.24's printed matrices: cross-checking by hand found two
cells in that table that are internally inconsistent with the book's
own marginalisation property (Section 11.2.1, p. 202: "the
marginalisation a_X(x) = sum_y a_YX(y x) holds"), most likely isolated
OCR/print errors rather than a formula issue -- the row/column sums
below are self-validating and don't depend on trusting those cells.
"""

import math

from subjective_logic import MultinomialOpinion, joint_opinion, marginalize


def _match_fixing_setup():
    opinion_x = MultinomialOpinion(
        belief_masses={"x1": 0.90, "x2": 0.00}, uncertainty=0.10, base_rates={"x1": 0.1, "x2": 0.9}
    )
    conditionals = {
        "x1": MultinomialOpinion(
            belief_masses={"y1": 0.00, "y2": 0.80, "y3": 0.10},
            uncertainty=0.10,
            base_rates={"y1": 1 / 3, "y2": 1 / 3, "y3": 1 / 3},
        ),
        "x2": MultinomialOpinion(
            belief_masses={"y1": 0.70, "y2": 0.00, "y3": 0.10},
            uncertainty=0.20,
            base_rates={"y1": 1 / 3, "y2": 1 / 3, "y3": 1 / 3},
        ),
    }
    return opinion_x, conditionals


def test_joint_projected_probability_row_and_column_sums():
    """
    P_YX row sums must recover P_Y||X from the Match-Fixing deduction
    (Ch. 9: 0.148, 0.739, 0.113); column sums must recover P_X (0.91, 0.09).
    """
    opinion_x, conditionals = _match_fixing_setup()
    joint = joint_opinion(opinion_x, conditionals)
    projected = joint.projected_probabilities

    row_y1 = sum(p for (y, x), p in projected.items() if y == "y1")
    row_y2 = sum(p for (y, x), p in projected.items() if y == "y2")
    row_y3 = sum(p for (y, x), p in projected.items() if y == "y3")
    assert math.isclose(row_y1, 0.148, abs_tol=0.001)
    assert math.isclose(row_y2, 0.739, abs_tol=0.001)
    assert math.isclose(row_y3, 0.113, abs_tol=0.001)

    col_x1 = sum(p for (y, x), p in projected.items() if x == "x1")
    col_x2 = sum(p for (y, x), p in projected.items() if x == "x2")
    assert math.isclose(col_x1, 0.91, abs_tol=0.001)
    assert math.isclose(col_x2, 0.09, abs_tol=0.001)


def test_joint_base_rate_column_sums_recover_a_x():
    """
    Section 11.2.1 (p. 202): 'the marginalisation a_X(x) = sum_y
    a_YX(y x) holds'. This is the property that exposed the two
    apparent print errors in Eq. 11.24's a_YX matrix.
    """
    opinion_x, conditionals = _match_fixing_setup()
    joint = joint_opinion(opinion_x, conditionals)

    col_x1 = sum(a for (y, x), a in joint.base_rates.items() if x == "x1")
    col_x2 = sum(a for (y, x), a in joint.base_rates.items() if x == "x2")
    assert math.isclose(col_x1, 0.1, abs_tol=1e-6)
    assert math.isclose(col_x2, 0.9, abs_tol=1e-6)


def test_joint_uncertainty_approximately_matches_book():
    """Eq. 11.25: u_YX = 0.072 (loose tolerance given the table's other transcription issues)."""
    opinion_x, conditionals = _match_fixing_setup()
    joint = joint_opinion(opinion_x, conditionals)
    assert math.isclose(joint.uncertainty, 0.072, abs_tol=0.02)


def test_marginalize_recovers_exact_projected_probabilities_and_base_rates():
    """
    p. 205: marginal base rates and projected probabilities exactly
    match the original omega_X and deduced omega_Y||X -- but NOT
    necessarily their uncertainty (explicitly acknowledged as
    approximate in the book, p. 206), so uncertainty is not asserted here.
    """
    opinion_x, conditionals = _match_fixing_setup()
    joint = joint_opinion(opinion_x, conditionals)
    marginal_y, marginal_x = marginalize(joint)

    original_projected_x = opinion_x.projected_probabilities
    for x in opinion_x.domain:
        assert math.isclose(marginal_x.projected_probabilities[x], original_projected_x[x], abs_tol=1e-9)
        assert math.isclose(marginal_x.base_rates[x], opinion_x.base_rates[x], abs_tol=1e-9)

    # Deduced Y (Chapter 9) matches the marginal Y's projected probability exactly.
    from subjective_logic import multinomial_deduce

    deduced_y = multinomial_deduce(opinion_x, conditionals)
    for y in deduced_y.domain:
        assert math.isclose(marginal_y.projected_probabilities[y], deduced_y.projected_probabilities[y], abs_tol=1e-9)
        assert math.isclose(marginal_y.base_rates[y], deduced_y.base_rates[y], abs_tol=1e-9)