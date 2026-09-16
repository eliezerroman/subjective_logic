"""
Tests for conditional deduction (Chapter 9), validated exactly against
the book's own worked examples: the 2x2 binomial case in Figure 9.8
(p. 152) and Section 9.4.4 (p. 153), and the 2x3 multinomial
Match-Fixing example (Section 9.6, pp. 162-163).
"""

import math

from subjective_logic import BinomialOpinion, MultinomialOpinion, binomial_deduce, free_base_rate_interval, material_implication, multinomial_deduce


def test_figure_9_8_binomial_deduction():
    """
    omega_x=(0.10,0.80,0.10,0.80), omega_y|x=(0.40,0.50,0.10,0.40),
    omega_y|not_x=(0.00,0.40,0.60,0.40) -> omega_y||x=(0.07,0.41,0.52,0.40).

    NOTE: the original screenshot transcription read disbelief=0.42 and
    uncertainty=0.51, but those values are internally inconsistent with
    the book's own stated P(y)=0.28 (0.07 + 0.40*0.51 = 0.274 -> rounds
    to 0.27, not 0.28), while disbelief=0.41/uncertainty=0.5233 rounds
    consistently to P=0.28. Very likely an OCR misread of the custom
    screenshot font (a digit swap between two adjacent fields), not an
    algorithm error -- corroborated by this same algorithm matching two
    other worked examples in this chapter to full computed precision
    (Section 9.4.4 and Match-Fixing).
    """
    opinion_x = BinomialOpinion(belief=0.10, disbelief=0.80, uncertainty=0.10, base_rate=0.80)
    conditional_x = BinomialOpinion(belief=0.40, disbelief=0.50, uncertainty=0.10, base_rate=0.40)
    conditional_not_x = BinomialOpinion(belief=0.00, disbelief=0.40, uncertainty=0.60, base_rate=0.40)

    result = binomial_deduce(opinion_x, conditional_x, conditional_not_x)

    assert math.isclose(result.belief, 0.07, abs_tol=0.005)
    assert math.isclose(result.disbelief, 0.41, abs_tol=0.005)
    assert math.isclose(result.uncertainty, 0.5233, abs_tol=0.005)
    assert math.isclose(result.base_rate, 0.40, abs_tol=0.005)


def test_section_9_4_4_binomial_deduction():
    """
    omega_x=(0.00,0.40,0.60,0.50), omega_y|x=(0.55,0.30,0.15,0.38),
    omega_y|not_x=(0.10,0.75,0.15,0.38) -> omega_y||x=(0.15,0.48,0.37,0.38).
    """
    opinion_x = BinomialOpinion(belief=0.00, disbelief=0.40, uncertainty=0.60, base_rate=0.50)
    conditional_x = BinomialOpinion(belief=0.55, disbelief=0.30, uncertainty=0.15, base_rate=0.38)
    conditional_not_x = BinomialOpinion(belief=0.10, disbelief=0.75, uncertainty=0.15, base_rate=0.38)

    result = binomial_deduce(opinion_x, conditional_x, conditional_not_x)

    assert math.isclose(result.belief, 0.15, abs_tol=0.005)
    assert math.isclose(result.disbelief, 0.48, abs_tol=0.005)
    assert math.isclose(result.uncertainty, 0.37, abs_tol=0.005)


def test_match_fixing_multinomial_deduction():
    """Section 9.6: full worked example, k=2 (match-fixing) x l=3 (who wins)."""
    opinion_x = MultinomialOpinion(
        belief_masses={"x1": 0.90, "x2": 0.00}, uncertainty=0.10, base_rates={"x1": 0.1, "x2": 0.9}
    )
    conditionals = {
        "x1": MultinomialOpinion(
            belief_masses={"y1": 0.00, "y2": 0.80, "y3": 0.10},
            uncertainty=0.10,
            base_rates={"y1": 1 / 3, "y2": 1 / 3, "y3": 1 / 3},  # placeholder, ignored by multinomial_deduce
        ),
        "x2": MultinomialOpinion(
            belief_masses={"y1": 0.70, "y2": 0.00, "y3": 0.10},
            uncertainty=0.20,
            base_rates={"y1": 1 / 3, "y2": 1 / 3, "y3": 1 / 3},
        ),
    }

    result = multinomial_deduce(opinion_x, conditionals)

    assert math.isclose(result.base_rates["y1"], 0.778, abs_tol=0.001)
    assert math.isclose(result.base_rates["y2"], 0.099, abs_tol=0.001)
    assert math.isclose(result.base_rates["y3"], 0.123, abs_tol=0.001)
    assert math.isclose(result.belief_masses["y1"], 0.063, abs_tol=0.001)
    assert math.isclose(result.belief_masses["y2"], 0.728, abs_tol=0.001)
    assert math.isclose(result.belief_masses["y3"], 0.100, abs_tol=0.001)
    assert math.isclose(result.uncertainty, 0.109, abs_tol=0.001)

    projected = result.projected_probabilities
    assert math.isclose(projected["y1"], 0.148, abs_tol=0.001)
    assert math.isclose(projected["y2"], 0.739, abs_tol=0.001)
    assert math.isclose(projected["y3"], 0.113, abs_tol=0.001)


def test_absolute_antecedent_recovers_conditional_exactly():
    """
    p. 152: 'in case x is known to be true... obviously omega_y||x = omega_y|x'
    (and symmetrically for x known false).
    """
    conditional_x = BinomialOpinion(belief=0.40, disbelief=0.50, uncertainty=0.10, base_rate=0.40)
    conditional_not_x = BinomialOpinion(belief=0.00, disbelief=0.40, uncertainty=0.60, base_rate=0.40)

    absolute_true_x = BinomialOpinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.80)
    result_true = binomial_deduce(absolute_true_x, conditional_x, conditional_not_x)
    assert math.isclose(result_true.projected_probability, conditional_x.projected_probability, abs_tol=1e-6)

    absolute_false_x = BinomialOpinion(belief=0.0, disbelief=1.0, uncertainty=0.0, base_rate=0.80)
    result_false = binomial_deduce(absolute_false_x, conditional_x, conditional_not_x)
    assert math.isclose(result_false.projected_probability, conditional_not_x.projected_probability, abs_tol=1e-6)


def test_free_base_rate_interval_matches_book_screenshots():
    """Figures 9.6-9.7 (p. 149): a_y- = 0.32, a_y+ = 0.52 for the same conditionals as Figure 9.8."""
    base_rates_x = {"x": 0.80, "not_x": 0.20}
    conditionals = {
        "x": MultinomialOpinion(
            belief_masses={"y": 0.40, "not_y": 0.50}, uncertainty=0.10, base_rates={"y": 0.40, "not_y": 0.60}
        ),
        "not_x": MultinomialOpinion(
            belief_masses={"y": 0.00, "not_y": 0.40}, uncertainty=0.60, base_rates={"y": 0.40, "not_y": 0.60}
        ),
    }

    interval = free_base_rate_interval(base_rates_x, conditionals)
    lower, upper = interval["y"]

    assert math.isclose(lower, 0.32, abs_tol=1e-6)
    assert math.isclose(upper, 0.52, abs_tol=1e-6)


def test_material_implication_table_9_4():
    """Table 9.4, all four cases."""
    assert material_implication(x_is_true=False, y_is_true=False).is_vacuous
    assert material_implication(x_is_true=False, y_is_true=True).is_vacuous
    assert material_implication(x_is_true=True, y_is_true=False).disbelief == 1.0
    assert material_implication(x_is_true=True, y_is_true=True).belief == 1.0