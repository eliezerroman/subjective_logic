"""
Subjective trust networks: automated analysis of DSPG trust graphs.

Implements Section 15.3 (Josang, 2016, pp. 271-279): given a trust
network represented as a directed series-parallel graph (DSPG,
Definitions 15.1-15.2), automatically derives the opinion at a target
node from an analyst's perspective, by repeatedly applying:

  - series reduction: a node with exactly one inbound and one outbound
    edge collapses via two-edge trust discounting (Chapter 14).
  - parallel reduction: multiple edges between the same pair of nodes
    collapse via belief fusion (Chapter 12).

This implements Definition 15.1's graph-transformation procedure
(Figure 15.1) directly, rather than the nesting-level/PPS bookkeeping
of Sections 15.2-15.3.1: for any genuine SP-graph, repeated local
series/parallel reduction (in any valid order) converges to the same
single-edge result, so explicit nesting-level computation is an
optimisation for choosing an efficient order, not a requirement for
correctness.

NOT implemented: Section 15.4 (synthesising a DSPG from a genuinely
non-series-parallel network via exhaustive or heuristic path
selection). That method has no worked numeric example in the book to
validate against, involves substantial combinatorial complexity, and
is explicitly heuristic rather than a single correct procedure.

IMPORTANT usage caveat (Section 15.3.2, p. 277-279): only feed this
class opinions received DIRECTLY and unmodified from their source. If
a node B has already discounted its own upstream sources before
passing you its opinion, and you then discount B's opinion again by
your trust in B, you double-count the separation and get a result
inconsistent with the real topology ("hidden topology" problem). This
matters directly for multi-agent pipelines: never chain a discount
over an opinion that already silently incorporates someone else's
discounting.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

from .binomial import BinomialOpinion

_FUSION_METHODS = {
    "cumulative": "fuse_cumulative",
    "averaging": "fuse_averaging",
    "weighted": "fuse_weighted",
}


class TrustNetwork:
    """
    A directed graph of trust/belief edges, each carrying a
    BinomialOpinion, supporting automated resolution of a DSPG trust
    network (Section 15.3) into a single derived opinion.
    """

    def __init__(self) -> None:
        self._edges: Dict[Tuple[Hashable, Hashable], List[BinomialOpinion]] = {}

    def add_edge(self, source: Hashable, target: Hashable, opinion: BinomialOpinion) -> None:
        """
        Add a trust or belief edge from source to target. Referral vs.
        functional trust (Table 14.1) is not distinguished here -- both
        are handled identically by the discounting/fusion math. Multiple
        edges between the same pair are allowed (parallel paths, to be
        fused during resolve()).
        """
        self._edges.setdefault((source, target), []).append(opinion)

    def _enumerate_paths(adjacency, current, sink, path, visited):
        if current == sink:
            yield list(path)
            return
        for target, opinion in adjacency.get(current, []):
            if target in visited:
                continue  # avoid cycles; a DSPG should be acyclic (Definition 15.2)
            visited.add(target)
            path.append(opinion)
            yield from _enumerate_paths(adjacency, target, sink, path, visited)
            path.pop()
            visited.discard(target)

    def resolve(self, source: Hashable, sink: Hashable, fusion: str = "cumulative") -> BinomialOpinion:
        """
        Derive A's opinion at the sink from a DSPG, per Definition 14.7 /
        Eq. 14.13-14.14: for EACH complete path from source to sink,
        multiply the projected probabilities of every edge except the
        last (Eq. 14.13), then apply trust discounting ONCE to the last
        edge using that product (Eq. 14.14). The results from all
        complete paths are then fused (Chapter 12).

        IMPORTANT design note: this does NOT repeatedly apply the full
        two-edge discount operator hop by hop along a referral chain --
        that would be mathematically wrong (discounting is an affine
        pull toward each hop's own base rate, not a simple probability
        multiplication, so chaining it hop by hop does not reproduce
        Eq. 14.13's plain product). Instead, referral probabilities are
        multiplied as plain scalars first (via referral_trust_probability),
        and the full discount operator is applied only once, to the
        final edge of each path.

        KNOWN LIMITATION: for graphs where multiple paths converge at an
        INTERMEDIATE node (not directly at sink) before continuing
        onward -- e.g. Figure 15.4's nested structure -- this function
        enumerates full source-to-sink paths and fuses everything only
        at the sink, rather than fusing at the intermediate convergence
        point and continuing the chain from there (as Section 15.3.1's
        algorithm describes). Whether these two orders give identical
        results for such nested cases has not been established here
        (no worked numeric example exists in the book to check against).
        For a nested graph, the safe approach is to resolve() the inner
        sub-network first and feed its result as a single edge into an
        outer TrustNetwork, rather than relying on one resolve() call
        for the whole nested structure.

        Args:
            fusion: which Chapter 12 operator to use when merging
                complete paths -- "cumulative" (default), "averaging",
                or "weighted".

        Raises:
            ValueError: if no path exists from source to sink.
        """
        from .trust import referral_trust_probability

        if fusion not in _FUSION_METHODS:
            raise ValueError(f"Unknown fusion method {fusion!r}; expected one of {sorted(_FUSION_METHODS)}.")
        fuse_method_name = _FUSION_METHODS[fusion]

        adjacency: Dict[Hashable, List[Tuple[Hashable, BinomialOpinion]]] = {}
        for (u, v), opinions in self._edges.items():
            for opinion in opinions:
                adjacency.setdefault(u, []).append((v, opinion))

        paths = list(_enumerate_paths(adjacency, source, sink, [], {source}))
        if not paths:
            raise ValueError(f"No path found from {source!r} to {sink!r}.")

        discounted_per_path = []
        for path in paths:
            if len(path) == 1:
                discounted_per_path.append(path[0])
            else:
                referral_edges, functional_edge = path[:-1], path[-1]
                probability = referral_trust_probability(referral_edges)
                discounted_per_path.append(functional_edge.discount_by_probability(probability))

        result = discounted_per_path[0]
        for other in discounted_per_path[1:]:
            result = getattr(result, fuse_method_name)(other)

        return result