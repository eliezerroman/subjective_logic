"""
Tests for subjective abduction (Chapter 10). The most reliable checks
use values printed as equations in the book's body text (Table 10.1,
p. 182; Eq. 10.67, p. 197); the Figure 10.4/10.5/10.6 checks use looser
tolerances since those come from tool screenshots in the same
problematic custom font already flagged in Chapter 9.
"""

import math

from subjective_logic import BinomialOpinion, MultinomialOpinion, binomial_abduce, binomial_invert, dependence, multinomial_abduce, multinomial_invert, relevance


def test_binomial_invert_matches_table_10_1_single_inversion():
    """
    Table 10.1 (p. 182, printed as an equation, not a screenshot):
    inverting omega_y|x=(0.80,0.20,0,0.50), omega_y|not_x=(0.20,0.80,0,
    0.50) with a_x=0.50 gives omega_x'|y=(0.72,0.12,0.16,0.50).
    """
    conditional_x = BinomialOpinion(belief=0.80, disbelief=0.20, uncertainty=0.00, base_rate=0.50)
    conditional_not_x = BinomialOpinion(belief=0.20, disbelief=0.80, uncertainty=0.00, base_rate=0.50)

    inverted_y, inverted_not_y = binomial_invert(conditional_x, conditional_not_x, base_rate_x=0.50)

    assert math.isclose(inverted_y.belief, 0.72, abs_tol=1e-6)
    assert math.isclose(inverted_y.disbelief, 0.12, abs_tol=1e-6)
    assert math.isclose(inverted_y.uncertainty, 0.16, abs_tol=1e-6)
    # By the symmetry of this specific example (p. 182's own note).
    assert math.isclose(inverted_not_y.uncertainty, 0.16, abs_tol=1e-6)


def test_repeated_inversion_uncertainty_is_non_decreasing():
    """
    Section 10.3.4 (p. 181-182): repeated inversion generally increases
    uncertainty, converging toward a theoretical maximum. This checks
    the qualitative property across several repeated inversions,
    without depending on exact convergence values.
    """
    conditional_x = BinomialOpinion(belief=0.80, disbelief=0.20, uncertainty=0.00, base_rate=0.50)
    conditional_not_x = BinomialOpinion(belief=0.20, disbelief=0.80, uncertainty=0.00, base_rate=0.50)

    uncertainties = [conditional_x.uncertainty]
    current_x, current_not_x = conditional_x, conditional_not_x
    for _ in range(6):
        current_x, current_not_x = binomial_invert(current_x, current_not_x, base_rate_x=0.50)
        uncertainties.append(current_x.uncertainty)

    for earlier, later in zip(uncertainties, uncertainties[1:]):
        assert later >= earlier - 1e-9


def test_military_intelligence_multinomial_abduction():
    """Section 10.8.2 (Eq. 10.67, p. 197): full worked multinomial abduction example."""
    base_rates_x = {"x1": 0.70, "x2": 0.20, "x3": 0.10}

    # Table 10.5: conditionals already uncertainty-maximised using aY.
    base_rates_y = {"y1": 0.35, "y2": 0.30, "y3": 0.35}
    dogmatic_conditionals = {
        "x1": MultinomialOpinion(belief_masses={"y1": 0.50, "y2": 0.25, "y3": 0.25}, uncertainty=0.0, base_rates=base_rates_y),
        "x2": MultinomialOpinion(belief_masses={"y1": 0.00, "y2": 0.50, "y3": 0.50}, uncertainty=0.0, base_rates=base_rates_y),
        "x3": MultinomialOpinion(belief_masses={"y1": 0.00, "y2": 0.25, "y3": 0.75}, uncertainty=0.0, base_rates=base_rates_y),
    }
    conditionals = {x: cond.uncertainty_maximized() for x, cond in dogmatic_conditionals.items()}

    # Verify our uncertainty-maximisation reproduces Table 10.5 exactly.
    assert math.isclose(conditionals["x1"].uncertainty, 0.7143, abs_tol=0.001)
    assert math.isclose(conditionals["x2"].uncertainty, 0.0, abs_tol=1e-6)
    assert math.isclose(conditionals["x3"].uncertainty, 0.0, abs_tol=1e-6)

    # Verify the y2 inversion column exactly (hand-checked against Table 10.6).
    inverted = multinomial_invert(conditionals, base_rates_x)
    assert math.isclose(inverted["y2"].uncertainty, 0.833, abs_tol=0.001)
    assert math.isclose(inverted["y2"].belief_masses["x1"], 0.00, abs_tol=0.001)
    assert math.isclose(inverted["y2"].belief_masses["x2"], 0.1667, abs_tol=0.001)
    assert math.isclose(inverted["y2"].belief_masses["x3"], 0.00, abs_tol=0.001)

    # Eq. 10.66: the uncertainty-maximised evidence opinion on Y.
    dogmatic_y = MultinomialOpinion(belief_masses={"y1": 0.20, "y2": 0.60, "y3": 0.20}, uncertainty=0.0, base_rates=base_rates_y)
    opinion_y = dogmatic_y.uncertainty_maximized()
    assert math.isclose(opinion_y.uncertainty, 0.5714, abs_tol=0.001)

    result = multinomial_abduce(opinion_y, conditionals, base_rates_x)

    # Eq. 10.67.
    assert math.isclose(result.uncertainty, 0.93, abs_tol=0.01)
    projected = result.projected_probabilities
    assert math.isclose(projected["x1"], 0.65, abs_tol=0.01)
    assert math.isclose(projected["x2"], 0.26, abs_tol=0.01)
    assert math.isclose(projected["x3"], 0.09, abs_tol=0.01)


def test_base_rate_fallacy_disease_a_vs_disease_b():
    """
    Section 10.5 (pp. 184-186): two tests of EQUAL quality (same
    sensitivity/specificity) give radically different diagnostic
    conclusions purely due to different base rates. Directly relevant
    to this project's degradation-analysis goals: identical model
    "quality" metrics can mask very different real-world reliability.
    """
    sensitivity = BinomialOpinion(belief=0.90, disbelief=0.05, uncertainty=0.05, base_rate=0.50)
    specificity = BinomialOpinion(belief=0.90, disbelief=0.05, uncertainty=0.05, base_rate=0.50)
    unspecificity = ~specificity  # complement, Chapter 6

    positive_test = BinomialOpinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.5)

    # Disease A: base rate 0.50 -> P(disease A | positive test) = 0.93.
    result_a = binomial_abduce(positive_test, sensitivity, unspecificity, base_rate_x=0.50)
    assert math.isclose(result_a.projected_probability, 0.93, abs_tol=0.02)

    # Disease B: same test quality, base rate 0.01 -> P(disease B | positive test) = only 0.15.
    result_b = binomial_abduce(positive_test, sensitivity, unspecificity, base_rate_x=0.01)
    assert math.isclose(result_b.projected_probability, 0.15, abs_tol=0.03)

    # The core point of the section: same test quality, very different conclusions.
    assert result_a.projected_probability - result_b.projected_probability > 0.5


def test_relevance_and_dependence():
    """Sanity checks for Definitions 10.1-10.3 using the Military Intelligence conditionals."""
    base_rates_y = {"y1": 1 / 3, "y2": 1 / 3, "y3": 1 / 3}
    conditionals = {
        "x1": MultinomialOpinion(belief_masses={"y1": 0.50, "y2": 0.25, "y3": 0.25}, uncertainty=0.0, base_rates=base_rates_y),
        "x2": MultinomialOpinion(belief_masses={"y1": 0.00, "y2": 0.50, "y3": 0.50}, uncertainty=0.0, base_rates=base_rates_y),
        "x3": MultinomialOpinion(belief_masses={"y1": 0.00, "y2": 0.25, "y3": 0.75}, uncertainty=0.0, base_rates=base_rates_y),
    }

    # y1: max(0.50,0.00,0.00) - min(...) = 0.50.
    assert math.isclose(relevance(conditionals, "y1"), 0.50, abs_tol=1e-6)
    assert 0.0 <= dependence(conditionals) <= 1.0