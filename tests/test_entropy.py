"""
Tests for entropy and surprisal (Section 4.7), validated against the
book's worked examples: fair coin (1 bit), fair die (2.585 bits), and
unfair die with P(six)=1/16 (4 bits surprisal), pp. 76-78.
"""

import math

from subjective_logic import BinomialOpinion, MultinomialOpinion, surprisal


def test_unfair_die_surprisal():
    """P('six') = 1/16 -> surprisal = 4 bits (p. 76)."""
    assert math.isclose(surprisal(1 / 16), 4.0, abs_tol=1e-9)


def test_fair_coin_entropy_is_one_bit():
    coin = BinomialOpinion(belief=0.5, disbelief=0.5, uncertainty=0.0, base_rate=0.5)
    assert math.isclose(coin.opinion_entropy(), 1.0, abs_tol=1e-9)


def test_fair_die_entropy_is_log2_6():
    base_rates = MultinomialOpinion.default_base_rates(range(1, 7))
    die = MultinomialOpinion(belief_masses=base_rates, uncertainty=0.0, base_rates=base_rates)
    assert math.isclose(die.opinion_entropy(), math.log2(6), abs_tol=1e-9)


def test_entropy_additivity_sharp_vague_uncertainty():
    """Eq. 4.59: H_S + H_V + H_U == H_P for any opinion."""
    coin = BinomialOpinion.from_evidence(r=3.0, s=1.0, base_rate=0.4)
    total = coin.sharpness_entropy() + coin.vagueness_entropy() + coin.uncertainty_entropy()
    assert math.isclose(total, coin.opinion_entropy(), abs_tol=1e-9)


def test_opinion_entropy_insensitive_to_uncertainty_given_same_projection():
    """Proposition 4.1 (p. 78): two opinions with the same P but different u have the same opinion entropy."""
    opinion_a = BinomialOpinion.from_probabilistic_notation(projected_probability=0.6, uncertainty=0.1, base_rate=0.5)
    opinion_b = BinomialOpinion.from_probabilistic_notation(projected_probability=0.6, uncertainty=0.5, base_rate=0.5)

    assert math.isclose(opinion_a.opinion_entropy(), opinion_b.opinion_entropy(), abs_tol=1e-9)