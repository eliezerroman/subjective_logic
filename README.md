# subjective-logic

A Python implementation of **Subjective Logic**, following Jøsang, A. (2016).
*Subjective Logic: A Formalism for Reasoning Under Uncertainty*. Springer.

Subjective Logic (SL) extends probability calculus with an explicit **uncertainty**
dimension and **belief ownership**, representing degrees of belief as opinions
`ω = (belief, disbelief, uncertainty, base_rate)` instead of single probability
values. This project implements SL's operators from the ground up, chapter by
chapter, with every formula validated against the book's own worked numerical
examples wherever one exists.

## Motivation

This module was built as part of a Master's research project studying **decision
degradation** in agentic AI systems operating under resource constraints (edge
computing). The goal is to move beyond operational metrics (accuracy, latency) and
use Subjective Logic's belief/disbelief/uncertainty/base-rate decomposition to
characterize *how* an AI agent's decisions degrade — distinguishing, for instance,
miscalibrated overconfidence from genuine lack of evidence — across classifiers,
LLMs, and RL agents, and across networks of dependent agents.

## Installation

```bash
git clone https://github.com/<your-username>/subjective-logic.git
cd subjective-logic
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick example

```python
from subjective_logic import BinomialOpinion

# An opinion built from evidence: 8 confirming observations, 2 disconfirming.
opinion = BinomialOpinion.from_evidence(r=8.0, s=2.0, base_rate=0.5)
print(opinion.projected_probability)   # 0.8
print(opinion.uncertainty)             # 0.2

# Discount it by how much you trust the source that reported it.
trust = BinomialOpinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
discounted = opinion.discount_by(trust)

# Fuse it with a second, independent opinion.
other = BinomialOpinion.from_evidence(r=5.0, s=1.0, base_rate=0.5)
fused = discounted.fuse_cumulative(other)
```

## Status

**121 automated tests, all passing.** The library covers the large majority of the
book's 17 chapters. Every implemented formula is cited to its exact equation/page
number in the source code's docstrings, and cross-checked against the book's own
worked numerical examples wherever one exists — see [Testing](#testing) and
[Known limitations](#known-limitations--things-worth-a-second-look) below for the
handful of places where that validation was partial or indirect.

## What's implemented

| Chapter | Topic | Module(s) | Notes |
|---|---|---|---|
| 2–3 | Opinion representations (binomial, multinomial, hyper-opinions), Beta/Dirichlet PDF mapping, uncertainty-maximisation, probabilistic notation | `binomial.py`, `multinomial.py`, `hyperopinion.py`, `domain.py` | `MultinomialOpinion` allows `k ≥ 2` (relaxed from the book's strict `k > 2`) so it can interoperate with Chapter 8's operators |
| 4 | Mass-sum (sharp/vague/focal belief), utility-normalised decision criteria, entropy, degree of conflict | `decision.py`, `entropy.py`, `conflict.py` | |
| 6 | Addition, subtraction, complement | `operators.py` | |
| 7 | Binomial multiplication, comultiplication, division, codivision | `operators.py` | Assumes **argument independence** — see caveats below |
| 8 | Multinomial normal/proportional multiplication, averaging-proportional and selective division, product-of-Dirichlet convenience | `multinomial_operators.py` | Multinomial division has **no general analytical solution** (the book's own words) — treat results as heuristic |
| 9 | Conditional deduction (binomial and multinomial) | `deduction.py` | Binomial deduction implemented via the multinomial algorithm, not the book's closed-form case-based formula — see caveats below |
| 10 | Relevance/dependence, subjective Bayes' theorem, abduction | `abduction.py` | |
| 11 | Joint and marginal opinions | `joint.py` | |
| 12 | Belief constraint fusion (Dempster's rule extension), cumulative fusion (aleatory + epistemic), averaging fusion, weighted fusion | `fusion.py`, `hyperopinion.py` | Consensus & Compromise Fusion **not implemented** — see below |
| 13 | Cumulative/averaging unfusion, cumulative fission | `unfusion.py` | |
| 14 | Trust discounting (two-edge and multi-edge), trust revision | `trust.py` | Trust fusion needs no new code — it's discounting composed with Ch. 12's fusion |
| 15 | Automated resolution of DSPG trust networks | `trust_network.py` | Non-DSPG network synthesis (Section 15.4) **not implemented** — see below |
| 16 | Bayesian reputation systems: time-decayed aggregation, dynamic community base rate, point estimates | `reputation.py` | Reputation scores themselves reuse Ch. 3's evidence→opinion mapping directly |
| 17 | Chain rules for subjective Bayesian networks | `subjective_networks.py` | Sections 17.1, 17.3, 17.5–17.6 are compositions of earlier chapters, demonstrated in tests rather than duplicated as new code |

## What's NOT implemented

These were deliberate scope decisions, each with a stated reason at the time:

- **Chapter 3**: Dirichlet HPDF density (§3.6.3) and Hyper-Dirichlet PDF (§3.6.5) — the latter's normalising factor has no closed-form expression per the book itself (p. 45); qualitative opinion representation (§3.7.2) — thresholds are explicitly application-specific in the source, nothing to transcribe.
- **Chapter 4**: Ambiguity (§4.9) — the book states this is future work with no formalism given.
- **Chapter 5**: Comparative/positioning chapter (Dempster-Shafer, fuzzy logic, etc.) — no computational content.
- **Chapter 8**: Projected multiplication (§8.1.5) and hypernomial product (§8.1.6) — require hyperdomain-of-a-Cartesian-product machinery not built, and the book itself deprioritises this method relative to normal/proportional multiplication (p. 121). Complex non-series-parallel reliability networks (§7.2.2) — explicitly stated as future work.
- **Chapter 12**: Consensus & Compromise Fusion (§12.6) — its three-step process requires enumerating all pairs of hyperdomain values and classifying each by intersection/union relationship, substantially more complex than the other fusion operators and deliberately deferred rather than risk an under-validated implementation.
- **Chapter 13**: Averaging fission (§13.2.3) — the book states this operator is trivial (produces the argument opinion unchanged) and not worth formalising.
- **Chapter 15**: Synthesis of a DSPG from a genuinely non-series-parallel network via exhaustive/heuristic path selection (§15.4) — no worked numeric example to validate against, substantial combinatorial complexity, and explicitly heuristic rather than a single correct procedure.
- **Chapter 16**: Continuous ratings via fuzzy triangular membership functions (§16.4.3) — illustrated with a figure but no closed-form formula given in the source.
- **Chapter 17**: Eq. 17.15's per-link inversion-then-chain method for chained abduction — the book's own Eq. 17.16 (chain forward, then invert once) is simpler and only marginally different in result; only the simpler method was implemented.
- **Section 9.2 / 16.1 / 17.1** (classical, non-opinion-based probabilistic formulas that set up context for the SL operators) — out of scope; this library implements *subjective logic*, not classical Bayesian statistics.

## Known limitations / things worth a second look

- **Binomial conditional deduction (Chapter 9, Definition 9.1)**: the book's closed-form formula has nine case distinctions comparing a conditional's belief/disbelief against its complement's. The source PDF's text extraction lost the bar diacritic distinguishing a value from its complement in several of those cases, making a literal transcription an unacceptable risk. Instead, `binomial_deduce()` wraps the unambiguous multinomial algorithm (Definition 9.2), justified by the book's own stated equivalence (p. 151) and validated against **three independent worked examples** to full computed precision. Mathematically sound, but the direct closed-form path was never implemented or tested on its own terms.
- **Figure 9.8's test values**: the transcribed screenshot values (`disbelief=0.42`, `uncertainty=0.51`) were internally inconsistent with the book's own stated projected probability for that example. The corrected values used in the test (`0.41`, `0.5233`) were derived independently and cross-checked, but if you have the physical book, a visual confirmation of that one figure would be a nice-to-have.
- **Joint opinions (Chapter 11, Eq. 11.24)**: two cells in the book's own printed example matrices were found to be numerically inconsistent with the book's own marginalisation identities (verified by hand). Likely isolated print/OCR errors; this project's tests validate via row/column-sum consistency instead of matching those specific cells.
- **`TrustNetwork.resolve()` (Chapter 15)**: validated exactly for networks where parallel paths converge only at the final sink. For networks where paths converge at an **intermediate** node before continuing onward (nested topologies, e.g. Figure 15.4), whether this path-enumeration approach is equivalent to the book's nested resolve-then-continue algorithm has **not been established** — no worked numeric example exists to check against. Safe workaround: `resolve()` inner sub-networks first and feed the result as a single edge into an outer `TrustNetwork`.
- **Chained inversion (Chapter 17)**: `chain_invert`'s result satisfies Bayes' rule only approximately (~1e-4 in the test suite) when cross-checked against a route that recomputes the marginal base rate differently. This mirrors the book's own documented "approximate, not exact" relationship between different routes through a conditional chain (§17.2.3) — not a bug, but a reminder that chained SL computations accumulate small, expected discrepancies depending on the order of operations.
- **Multiplication/comultiplication/division (Chapters 7–8)**: all assume **argument independence**. Using them on correlated agent opinions (e.g. two models sharing training data) silently understates uncertainty. Conditional (dependent) versions are Chapter 11's material implication onward, but true "conditional multiplication" for dependent arguments (mentioned in the book as Section 11.2) was not separately implemented as a standalone operator.
- **`marginal_conditionals` (Chapter 11)**: the book itself says "the exact nature of this approximation needs further investigation" (p. 206) — treat this function as exploratory, not load-bearing.
- **Multinomial division (Chapter 8)**: per the book's own words, "there is not even a correct solution to approximate" (p. 129) — `averaging_proportional_divide` and `selective_divide` are synthetic heuristics with input validation guards, not approximations of a known-correct inverse.

## Testing

```bash
pytest tests/ -v
```

Every formula is cited in its docstring to the exact equation and page number of
the source. Where the book gives a worked numerical example, tests check against
it (usually to exact computed precision; a handful of cases use a wider tolerance
where the book's own printed values appear to carry rounding or transcription
noise, always noted in the test's docstring). Where no such example exists,
self-consistency properties are used instead (e.g. round-trip identities, additivity
checks, or cross-checking a formula against an independently-derived version of
the same quantity).

## Project layout
src/subjective_logic/
├── binomial.py # Ch. 2-3: BinomialOpinion
├── multinomial.py # Ch. 2-3: MultinomialOpinion
├── hyperopinion.py # Ch. 3: HyperOpinion
├── domain.py # Ch. 2: hyperdomain/composite-set utilities
├── decision.py # Ch. 4: mass-sum, utility, decision criteria
├── entropy.py # Ch. 4: surprisal, opinion entropy
├── conflict.py # Ch. 4: projected distance, degree of conflict
├── operators.py # Ch. 6-7: +, -, complement, *, /, comultiply, codivide (binomial)
├── multinomial_operators.py # Ch. 8: multinomial multiplication/division
├── deduction.py # Ch. 9: conditional deduction
├── abduction.py # Ch. 10: subjective Bayes' theorem, abduction
├── joint.py # Ch. 11: joint and marginal opinions
├── fusion.py # Ch. 12: cumulative/averaging/weighted fusion, constraint fusion
├── unfusion.py # Ch. 13: unfusion and fission
├── trust.py # Ch. 14: trust discounting and revision
├── trust_network.py # Ch. 15: automated DSPG trust network resolution
├── reputation.py # Ch. 16: Bayesian reputation systems
└── subjective_networks.py # Ch. 17: chain rules for subjective Bayesian networks


## Citation

If you use this library in academic work, please cite the book this implementation
is based on:

A. Jøsang. Subjective Logic. en. Artificial Intelligence: Foundations, Theory,and Algorithms.
Cham:SpringerInternationalPublishing,2016.isbn:978-3-319-42335-7.doi:10.1007/
978-3-319-42337-1. url: http://link.springer.com/10.1007/978-3-319-
42337-1.

## License

MIT (see `LICENSE`).