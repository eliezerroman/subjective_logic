"""
Tests for degree of conflict (Section 4.8), validated against the two
worked examples in Josang (2016), pp. 80-82.
"""

import math

from subjective_logic import BinomialOpinion


def test_no_conflict_despite_different_belief_masses():
    """
    omega_B1 = (0.05, 0.15, 0.80, 0.90), omega_C1 = (0.68, 0.22, 0.10, 0.90):
    same projected probability (0.77), so DC = 0.0 despite very different
    belief masses (Figure 4.14).
    """
    opinion_b1 = BinomialOpinion(belief=0.05, disbelief=0.15, uncertainty=0.80, base_rate=0.90)
    opinion_c1 = BinomialOpinion(belief=0.68, disbelief=0.22, uncertainty=0.10, base_rate=0.90)

    assert math.isclose(opinion_b1.projected_probability, 0.77, abs_tol=1e-9)
    assert math.isclose(opinion_c1.projected_probability, 0.77, abs_tol=1e-9)
    assert math.isclose(opinion_b1.degree_of_conflict_with(opinion_c1), 0.0, abs_tol=1e-9)


def test_conflict_with_different_base_rates():
    """
    omega_B2 = (0.05, 0.15, 0.80, 0.10), omega_C2 = (0.68, 0.22, 0.10, 0.10):
    same belief masses as above, different base rate -> DC = 0.10 (Eq. 4.64).
    """
    opinion_b2 = BinomialOpinion(belief=0.05, disbelief=0.15, uncertainty=0.80, base_rate=0.10)
    opinion_c2 = BinomialOpinion(belief=0.68, disbelief=0.22, uncertainty=0.10, base_rate=0.10)

    assert math.isclose(opinion_b2.degree_of_conflict_with(opinion_c2), 0.1008, abs_tol=0.001)