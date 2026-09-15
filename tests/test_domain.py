"""
Tests for domain and hyperdomain utilities, validated against the
cardinality-class example in Josang (2016), Section 2.3, Table 2.1/2.2
(quaternary domain, Figures 2.2-2.4).
"""

from subjective_logic import base_rate_of_value, composite_set, hyperdomain, relative_base_rate


def test_hyperdomain_cardinality_matches_book_example():
    """
    Table 2.1: for a domain of cardinality k=4, cardinality class 1 has 4
    values, class 2 has 6, class 3 has 4. Total |R(X)| = 2^4 - 2 = 14.
    """
    domain = ["x1", "x2", "x3", "x4"]
    values = hyperdomain(domain)

    assert len(values) == 14
    singletons = [v for v in values if len(v) == 1]
    pairs = [v for v in values if len(v) == 2]
    triples = [v for v in values if len(v) == 3]
    assert len(singletons) == 4
    assert len(pairs) == 6
    assert len(triples) == 4


def test_composite_set_excludes_singletons():
    domain = ["x1", "x2", "x3", "x4"]
    composites = composite_set(domain)
    assert len(composites) == 10  # 6 + 4, per Table 2.1
    assert all(len(c) >= 2 for c in composites)


def test_base_rate_of_composite_value():
    """Eq. 2.9: base rate of a composite is the sum of its singletons' rates."""
    base_rates = {"x1": 0.5, "x2": 0.3, "x3": 0.2}
    assert base_rate_of_value(base_rates, frozenset({"x1", "x2"})) == 0.8


def test_relative_base_rate():
    """Eq. 2.10: relative base rate via intersection."""
    base_rates = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3}
    # a(x1 | {x1, x2}) = a({x1}) / a({x1, x2}) = (1/3) / (2/3) = 0.5
    result = relative_base_rate(base_rates, frozenset({"x1"}), frozenset({"x1", "x2"}))
    assert abs(result - 0.5) < 1e-9