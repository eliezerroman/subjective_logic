"""
Belief fusion: combining opinions from multiple sources about the same
variable.

Implements Sections 12.2-12.5 (Josang, 2016, pp. 207-233): belief
constraint fusion (Dempster's rule extension, for HyperOpinion), and the
three "flat" fusion operators -- cumulative, averaging, weighted -- which
work identically regardless of domain structure (no interaction between
different values is needed), implemented once here and wired into
BinomialOpinion, MultinomialOpinion and HyperOpinion.

Choosing the right operator matters (Section 12.1.2, p. 212):
- Cumulative (CBF): sources are INDEPENDENT evidence -- more sources
  means more evidence, reducing uncertainty (e.g. repeated independent
  samples of the same LLM, or several independent classifiers).
- Averaging (ABF): sources are DEPENDENT -- more sources does not mean
  more evidence (e.g. several runs of the same model on correlated
  inputs). Idempotent: fusing an opinion with itself changes nothing.
- Weighted (WBF): like averaging, but weighted by each source's
  confidence (1 - uncertainty), so a confident source dominates an
  uncertain one.

NOT implemented here: Consensus & Compromise Fusion (Section 12.6),
deliberately deferred -- see the conversation this module was built in
for the reasoning. Belief Constraint Fusion is implemented separately
in hyperopinion.py-adjacent code below since it needs the hyperdomain's
intersection structure, unlike the three operators above.
"""

from __future__ import annotations

from typing import Mapping

_TOLERANCE = 1e-9


def cumulative_fusion(
    belief_a: Mapping, uncertainty_a: float, base_rates_a: Mapping,
    belief_b: Mapping, uncertainty_b: float, base_rates_b: Mapping,
    domain,
) -> tuple:
    """
    Aleatory cumulative belief fusion, Definition 12.5 / Eq. 12.14-12.15
    (pp. 226-227): combines INDEPENDENT sources as if adding statistical
    evidence -- reduces uncertainty as sources accumulate (equivalent to
    adding the sources' Dirichlet evidence counts, Theorem 12.2).

    Returns (belief_fused, uncertainty_fused, base_rates_fused) as dicts
    over `domain`.
    """
    if uncertainty_a == 1.0 and uncertainty_b == 1.0:
        belief = {x: 0.0 for x in domain}
        uncertainty = 1.0
        base_rates = {x: (base_rates_a[x] + base_rates_b[x]) / 2 for x in domain}
    elif uncertainty_a == 0.0 and uncertainty_b == 0.0:
        # Both dogmatic: book's convention is a 0.5/0.5 limit split (p. 227).
        belief = {x: 0.5 * belief_a[x] + 0.5 * belief_b[x] for x in domain}
        uncertainty = 0.0
        base_rates = {x: 0.5 * base_rates_a[x] + 0.5 * base_rates_b[x] for x in domain}
    else:
        denom = uncertainty_a + uncertainty_b - uncertainty_a * uncertainty_b
        belief = {
            x: (belief_a[x] * uncertainty_b + belief_b[x] * uncertainty_a) / denom for x in domain
        }
        uncertainty = (uncertainty_a * uncertainty_b) / denom
        denom_a = uncertainty_a + uncertainty_b - 2 * uncertainty_a * uncertainty_b
        if denom_a == 0:
            base_rates = {x: (base_rates_a[x] + base_rates_b[x]) / 2 for x in domain}
        else:
            base_rates = {
                x: (
                    base_rates_a[x] * uncertainty_b
                    + base_rates_b[x] * uncertainty_a
                    - (base_rates_a[x] + base_rates_b[x]) * uncertainty_a * uncertainty_b
                )
                / denom_a
                for x in domain
            }
    return belief, uncertainty, base_rates


def averaging_fusion(
    belief_a: Mapping, uncertainty_a: float, base_rates_a: Mapping,
    belief_b: Mapping, uncertainty_b: float, base_rates_b: Mapping,
    domain,
) -> tuple:
    """
    Averaging belief fusion, Definition 12.7 / Eq. 12.18-12.19 (pp.
    229-230): combines DEPENDENT sources -- idempotent, does not reduce
    uncertainty just by adding more (correlated) sources.
    """
    if uncertainty_a == 0.0 and uncertainty_b == 0.0:
        belief = {x: 0.5 * belief_a[x] + 0.5 * belief_b[x] for x in domain}
        uncertainty = 0.0
    else:
        denom = uncertainty_a + uncertainty_b
        belief = {
            x: (belief_a[x] * uncertainty_b + belief_b[x] * uncertainty_a) / denom for x in domain
        }
        uncertainty = (2 * uncertainty_a * uncertainty_b) / denom
    base_rates = {x: (base_rates_a[x] + base_rates_b[x]) / 2 for x in domain}
    return belief, uncertainty, base_rates


def weighted_fusion(
    belief_a: Mapping, uncertainty_a: float, base_rates_a: Mapping,
    belief_b: Mapping, uncertainty_b: float, base_rates_b: Mapping,
    domain,
) -> tuple:
    """
    Weighted belief fusion, Definition 12.8 / Eq. 12.23-12.25 (pp.
    231-232): like averaging, but weighted by confidence (1 -
    uncertainty). Idempotent, has the vacuous opinion as neutral element
    (a source with u=1 contributes nothing).
    """
    if uncertainty_a == 1.0 and uncertainty_b == 1.0:
        belief = {x: 0.0 for x in domain}
        uncertainty = 1.0
        base_rates = {x: (base_rates_a[x] + base_rates_b[x]) / 2 for x in domain}
    elif uncertainty_a == 0.0 and uncertainty_b == 0.0:
        belief = {x: 0.5 * belief_a[x] + 0.5 * belief_b[x] for x in domain}
        uncertainty = 0.0
        base_rates = {x: 0.5 * base_rates_a[x] + 0.5 * base_rates_b[x] for x in domain}
    else:
        denom = uncertainty_a + uncertainty_b - 2 * uncertainty_a * uncertainty_b
        belief = {
            x: (
                belief_a[x] * (1 - uncertainty_a) * uncertainty_b
                + belief_b[x] * (1 - uncertainty_b) * uncertainty_a
            )
            / denom
            for x in domain
        }
        uncertainty = (2 - uncertainty_a - uncertainty_b) * uncertainty_a * uncertainty_b / denom
        base_rates = {
            x: (base_rates_a[x] * (1 - uncertainty_a) + base_rates_b[x] * (1 - uncertainty_b))
            / (2 - uncertainty_a - uncertainty_b)
            for x in domain
        }
    return belief, uncertainty, base_rates


def _harmony(x, belief_a: Mapping, uncertainty_a: float, belief_b: Mapping, uncertainty_b: float) -> float:
    """Har(x), Eq. 12.3 (p. 216): relative harmony (overlapping support) between two hyper-opinions at value x."""
    total = belief_a[x] * uncertainty_b + belief_b[x] * uncertainty_a
    for value_a, mass_a in belief_a.items():
        for value_b, mass_b in belief_b.items():
            if (value_a & value_b) == x:
                total += mass_a * mass_b
    return total


def _conflict(belief_a: Mapping, belief_b: Mapping) -> float:
    """Con, Eq. 12.4 (p. 216): total belief mass on totally disjoint value pairs."""
    total = 0.0
    for value_a, mass_a in belief_a.items():
        for value_b, mass_b in belief_b.items():
            if not (value_a & value_b):
                total += mass_a * mass_b
    return total