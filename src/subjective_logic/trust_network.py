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

    def resolve(self, source: Hashable, sink: Hashable, fusion: str = "cumulative") -> BinomialOpinion:
        """
        Reduce the DSPG between source and sink to a single derived
        opinion, per Definition 15.1's series/parallel collapse
        procedure.

        Args:
            fusion: which Chapter 12 operator to use when merging
                parallel paths -- "cumulative" (independent sources,
                the default), "averaging" (dependent sources), or
                "weighted" (confidence-weighted).

        Raises:
            ValueError: if the graph cannot be reduced to a single
            source-sink edge -- it is not a DSPG (Definition 15.2), or
            there are edges disconnected from the source/sink.
        """
        if fusion not in _FUSION_METHODS:
            raise ValueError(f"Unknown fusion method {fusion!r}; expected one of {sorted(_FUSION_METHODS)}.")
        fuse_method_name = _FUSION_METHODS[fusion]

        edges = {key: list(value) for key, value in self._edges.items()}

        progress = True
        while progress:
            progress = False

            # Parallel reduction (Definition 15.1, operation (ii)).
            for key, opinions in list(edges.items()):
                if len(opinions) > 1:
                    fused = opinions[0]
                    for other in opinions[1:]:
                        fused = getattr(fused, fuse_method_name)(other)
                    edges[key] = [fused]
                    progress = True

            # Series reduction (Definition 15.1, operation (i)): a node
            # (not source/sink) with exactly one inbound and one
            # outbound edge collapses via trust discounting.
            nodes = {node for pair in edges for node in pair}
            for node in nodes:
                if node == source or node == sink:
                    continue
                incoming = [key for key in edges if key[1] == node]
                outgoing = [key for key in edges if key[0] == node]
                if (
                    len(incoming) == 1 and len(outgoing) == 1
                    and len(edges[incoming[0]]) == 1 and len(edges[outgoing[0]]) == 1
                ):
                    trust_opinion = edges.pop(incoming[0])[0]
                    target_opinion = edges.pop(outgoing[0])[0]
                    discounted = target_opinion.discount_by(trust_opinion)
                    new_key = (incoming[0][0], outgoing[0][1])
                    edges.setdefault(new_key, []).append(discounted)
                    progress = True
                    break  # edges dict changed; restart the node scan

        if list(edges.keys()) != [(source, sink)]:
            raise ValueError(
                f"Could not reduce the network between {source!r} and {sink!r} to a single edge "
                "by series/parallel collapse -- it may not be a DSPG (Definition 15.2), or there "
                "may be edges disconnected from the source/sink. Section 15.4's synthesis method "
                "for non-series-parallel networks is not implemented here."
            )

        return edges[(source, sink)][0]