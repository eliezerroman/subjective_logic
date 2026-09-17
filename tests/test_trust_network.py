"""
Tests for automated DSPG trust network resolution (Chapter 15).
Validated by expressing Chapter 14's already-verified numeric examples
(Table 14.2's chain, Table 14.3's diamond) as graphs, since Chapter 15
itself contains no numeric worked example (only structural diagrams).
"""

import math

import pytest

from subjective_logic import BinomialOpinion, TrustNetwork


def test_resolve_series_chain_matches_table_14_2():
    """A 4-edge chain, reusing Table 14.2's numbers."""
    edge = BinomialOpinion(belief=0.20, disbelief=0.10, uncertainty=0.70, base_rate=0.80)
    target = BinomialOpinion(belief=0.80, disbelief=0.20, uncertainty=0.00, base_rate=0.10)

    network = TrustNetwork()
    network.add_edge("A", "B", edge)
    network.add_edge("B", "C", edge)
    network.add_edge("C", "D", edge)
    network.add_edge("D", "X", target)

    result = network.resolve("A", "X")

    assert math.isclose(result.belief, 0.35, abs_tol=0.005)
    assert math.isclose(result.disbelief, 0.09, abs_tol=0.005)
    assert math.isclose(result.uncertainty, 0.56, abs_tol=0.005)


def test_resolve_parallel_diamond_matches_table_14_3():
    """Two paths converging (a diamond), reusing Table 14.3's numbers."""
    trust_b = BinomialOpinion(belief=0.40, disbelief=0.10, uncertainty=0.50, base_rate=0.60)
    opinion_b = BinomialOpinion(belief=0.90, disbelief=0.00, uncertainty=0.10, base_rate=0.40)
    trust_c = BinomialOpinion(belief=0.50, disbelief=0.00, uncertainty=0.50, base_rate=0.50)
    opinion_c = BinomialOpinion(belief=0.80, disbelief=0.10, uncertainty=0.10, base_rate=0.40)

    network = TrustNetwork()
    network.add_edge("A", "B", trust_b)
    network.add_edge("B", "E", opinion_b)
    network.add_edge("A", "C", trust_c)
    network.add_edge("C", "E", opinion_c)

    result = network.resolve("A", "E")

    assert math.isclose(result.belief, 0.743, abs_tol=0.001)
    assert math.isclose(result.disbelief, 0.048, abs_tol=0.001)
    assert math.isclose(result.uncertainty, 0.209, abs_tol=0.001)


def test_resolve_combined_series_and_parallel():
    """
    A generalisation check beyond the book's two simple cases: one
    2-hop chain and one 1-hop path converging, cross-checked against
    the same computation done manually with discount_by/fuse_cumulative.
    """
    trust_ab = BinomialOpinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5)
    trust_bc = BinomialOpinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
    opinion_ce = BinomialOpinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.4)
    trust_ad = BinomialOpinion(belief=0.4, disbelief=0.3, uncertainty=0.3, base_rate=0.5)
    opinion_de = BinomialOpinion(belief=0.6, disbelief=0.2, uncertainty=0.2, base_rate=0.4)

    network = TrustNetwork()
    network.add_edge("A", "B", trust_ab)
    network.add_edge("B", "C", trust_bc)
    network.add_edge("C", "E", opinion_ce)
    network.add_edge("A", "D", trust_ad)
    network.add_edge("D", "E", opinion_de)

    result = network.resolve("A", "E")

    path1_probability = trust_bc.projected_probability * trust_ab.projected_probability
    path1 = opinion_ce.discount_by_probability(path1_probability)
    path2 = opinion_de.discount_by(trust_ad)
    expected = path1.fuse_cumulative(path2)

    assert math.isclose(result.belief, expected.belief, abs_tol=1e-9)
    assert math.isclose(result.disbelief, expected.disbelief, abs_tol=1e-9)
    assert math.isclose(result.uncertainty, expected.uncertainty, abs_tol=1e-9)


def test_resolve_raises_when_no_path_exists():
    """
    resolve() enumerates paths from source to sink directly (Definition
    14.7's path-based approach), so an unrelated extra edge elsewhere in
    the graph (e.g. A->Y when resolving A->X) is simply ignored -- it is
    not on any A-to-X path, so it causes no error. The genuine failure
    case is when NO path exists at all between source and sink.
    """
    op = BinomialOpinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
    network = TrustNetwork()
    network.add_edge("A", "Y", op)  # Y is unrelated to X; no path from A to X exists
    with pytest.raises(ValueError):
        network.resolve("A", "X")


def test_resolve_ignores_unrelated_edges():
    """A extra edge not on any source-to-sink path should not affect the result or cause an error."""
    op = BinomialOpinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
    network = TrustNetwork()
    network.add_edge("A", "X", op)
    network.add_edge("A", "Y", op)  # unrelated to the A-X resolution

    result = network.resolve("A", "X")

    import math
    assert math.isclose(result.belief, op.belief, abs_tol=1e-9)