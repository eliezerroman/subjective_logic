"""
Tests for computational trust (Chapter 14), validated against Eq. 14.8
(two-edge discounting), Table 14.2 (multi-edge discounting), Table 14.3
(trust fusion), and Tables 14.4-14.5 (trust revision).
"""

import math

from subjective_logic import BinomialOpinion, degree_of_conflict, referral_trust_probability


def test_two_edge_discount_matches_eq_14_8():
    """Eq. 14.7-14.8: A trusts B (omega_B^A), B has opinion on X (omega_X^B) -> discounted omega_X^[A;B]."""
    trust = BinomialOpinion(belief=0.20, disbelief=0.40, uncertainty=0.40, base_rate=0.75)
    source = BinomialOpinion(belief=0.45, disbelief=0.35, uncertainty=0.20, base_rate=0.25)

    discounted = source.discount_by(trust)

    assert math.isclose(discounted.belief, 0.225, abs_tol=1e-6)
    assert math.isclose(discounted.disbelief, 0.175, abs_tol=1e-6)
    assert math.isclose(discounted.uncertainty, 0.600, abs_tol=1e-6)
    assert math.isclose(discounted.projected_probability, 0.375, abs_tol=1e-6)


def test_multi_edge_discount_matches_table_14_2():
    """Table 14.2: a 3-edge referral trust chain discounting a dogmatic target opinion."""
    edge = BinomialOpinion(belief=0.20, disbelief=0.10, uncertainty=0.70, base_rate=0.80)
    chain = [edge, edge, edge]
    source = BinomialOpinion(belief=0.80, disbelief=0.20, uncertainty=0.00, base_rate=0.10)

    trust_probability = referral_trust_probability(chain)
    assert math.isclose(trust_probability, 0.44, abs_tol=0.005)

    discounted = source.discount_by_probability(trust_probability)

    assert math.isclose(discounted.belief, 0.35, abs_tol=0.005)
    assert math.isclose(discounted.disbelief, 0.09, abs_tol=0.005)
    assert math.isclose(discounted.uncertainty, 0.56, abs_tol=0.005)
    assert math.isclose(discounted.projected_probability, 0.41, abs_tol=0.005)


def test_trust_fusion_matches_table_14_3():
    """
    Table 14.3: trust fusion is discount() composed with Chapter 12's
    cumulative fusion -- no new operator needed.
    """
    trust_b = BinomialOpinion(belief=0.40, disbelief=0.10, uncertainty=0.50, base_rate=0.60)
    opinion_b = BinomialOpinion(belief=0.90, disbelief=0.00, uncertainty=0.10, base_rate=0.40)
    trust_c = BinomialOpinion(belief=0.50, disbelief=0.00, uncertainty=0.50, base_rate=0.50)
    opinion_c = BinomialOpinion(belief=0.80, disbelief=0.10, uncertainty=0.10, base_rate=0.40)

    discounted_via_b = opinion_b.discount_by(trust_b)
    discounted_via_c = opinion_c.discount_by(trust_c)
    fused = discounted_via_b.fuse_cumulative(discounted_via_c)

    assert math.isclose(fused.belief, 0.743, abs_tol=0.001)
    assert math.isclose(fused.disbelief, 0.048, abs_tol=0.001)
    assert math.isclose(fused.uncertainty, 0.209, abs_tol=0.001)


def test_trust_revision_matches_tables_14_4_and_14_5():
    """
    Restaurant recommendation example: Bob is a vacuous, unknown source;
    Claire is confidently trusted. Simple fusion under-weights Claire's
    advice; trust revision corrects this by distrusting Bob.
    """
    trust_bob = BinomialOpinion(belief=0.00, disbelief=0.00, uncertainty=1.00, base_rate=0.90)
    opinion_bob = BinomialOpinion(belief=0.95, disbelief=0.00, uncertainty=0.05, base_rate=0.20)
    trust_claire = BinomialOpinion(belief=0.90, disbelief=0.00, uncertainty=0.10, base_rate=0.90)
    opinion_claire = BinomialOpinion(belief=0.10, disbelief=0.80, uncertainty=0.10, base_rate=0.20)

    discounted_bob = opinion_bob.discount_by(trust_bob)
    discounted_claire = opinion_claire.discount_by(trust_claire)

    # Simple fusion (Table 14.4): counter-intuitively close to 50/50.
    simple_fused = discounted_bob.fuse_cumulative(discounted_claire)
    assert math.isclose(simple_fused.projected_probability, 0.465, abs_tol=0.01)

    # Trust revision (Table 14.5).
    conflict = degree_of_conflict(discounted_bob, discounted_claire)
    assert math.isclose(conflict, 0.581, abs_tol=0.01)

    from subjective_logic import revision_factor

    rf_bob = revision_factor(trust_bob.uncertainty, trust_claire.uncertainty, conflict)
    rf_claire = revision_factor(trust_claire.uncertainty, trust_bob.uncertainty, conflict)
    assert math.isclose(rf_bob, 0.529, abs_tol=0.01)
    assert math.isclose(rf_claire, 0.053, abs_tol=0.01)

    revised_trust_bob = trust_bob.revise(rf_bob)
    revised_trust_claire = trust_claire.revise(rf_claire)

    assert math.isclose(revised_trust_bob.disbelief, 0.53, abs_tol=0.02)
    assert math.isclose(revised_trust_claire.belief, 0.85, abs_tol=0.02)

    revised_fused = opinion_bob.discount_by(revised_trust_bob).fuse_cumulative(
        opinion_claire.discount_by(revised_trust_claire)
    )
    # After revision, Claire's more-trusted, less-favourable advice dominates.
    assert math.isclose(revised_fused.projected_probability, 0.208, abs_tol=0.02)
    assert revised_fused.projected_probability < simple_fused.projected_probability