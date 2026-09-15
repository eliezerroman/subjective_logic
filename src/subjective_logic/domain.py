"""
Domain, hyperdomain and base rate utilities for Subjective Logic.

Implements the set-theoretic scaffolding needed for hyper-opinions,
following:

    Josang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
    Uncertainty. Springer. Chapter 2 (domains and hyperdomains) and
    Section 3.6 (hyper-opinions).

Values of a domain X are represented as plain hashable labels (e.g.
strings). Values of the corresponding hyperdomain R(X) -- singletons and
composites -- are represented as frozensets of those labels, so that a
singleton x becomes frozenset({"x"}) and a composite {x1, x2} becomes
frozenset({"x1", "x2"}). This makes the set operations (intersection)
needed for relative base rates (Eq. 2.10) direct and unambiguous.
"""

from __future__ import annotations

from itertools import combinations
from typing import FrozenSet, Hashable, Iterable, List, Mapping


def hyperdomain(domain: Iterable[Hashable]) -> List[FrozenSet[Hashable]]:
    """
    Build the hyperdomain R(X): the reduced powerset of X, excluding the
    empty set and the domain X itself (Definition 2.1, Eq. 2.1, p. 9).

    Cardinality: kappa = 2^k - 2, where k = |X| (verified against Table
    2.1/2.2 in the tests). This grows exponentially, so this function is
    only practical for small domains (k <= ~15 in practice).

    Returns:
        A list of frozensets: the k singletons first (in the order given
        by `domain`), followed by composites in increasing cardinality-
        class order (Section 2.3).
    """
    values = list(domain)
    k = len(values)
    if k < 2:
        raise ValueError(f"A domain must have at least 2 values to build a hyperdomain (got {k}).")

    result: List[FrozenSet[Hashable]] = [frozenset((v,)) for v in values]
    # Composite values: all proper subsets of cardinality 2 .. k-1
    # (cardinality k itself would be the domain X, which is excluded).
    for size in range(2, k):
        for combo in combinations(values, size):
            result.append(frozenset(combo))
    return result


def composite_set(domain: Iterable[Hashable]) -> List[FrozenSet[Hashable]]:
    """
    The composite set C(X): values of R(X) with cardinality >= 2
    (Definition 2.2, Eq. 2.3-2.4, p. 12).
    """
    return [x for x in hyperdomain(domain) if len(x) >= 2]


def base_rate_of_value(base_rates: Mapping[Hashable, float], value: FrozenSet[Hashable]) -> float:
    """
    Base rate a_X(x) of a (possibly composite) value x in R(X), computed
    as the sum of the base rates of the singletons it contains
    (Eq. 2.9, p. 16).
    """
    return sum(base_rates[singleton] for singleton in value)


def relative_base_rate(
    base_rates: Mapping[Hashable, float],
    x: FrozenSet[Hashable],
    xi: FrozenSet[Hashable],
) -> float:
    """
    Relative base rate a_X(x|xi) of value x relative to value xi
    (Definition 2.7, Eq. 2.10, p. 17):

        a_X(x|xi) = a_X(x intersect xi) / a_X(xi)

    Returns 0 when a_X(xi) == 0, per the definition's special case.
    """
    a_xi = base_rate_of_value(base_rates, xi)
    if a_xi == 0.0:
        return 0.0
    a_intersection = base_rate_of_value(base_rates, x & xi)
    return a_intersection / a_xi