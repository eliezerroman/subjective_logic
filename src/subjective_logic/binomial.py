"""
Binomial opinions in Subjective Logic.

This module implements the core representation of a binomial opinion,
omega_x = (b, d, u, a), as defined in:

    Josang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
    Uncertainty. Springer. Chapter 3, Section 3.4.

A binomial opinion expresses a subjective belief about a binary variable
X in {x, not_x} (e.g. "the classifier's prediction is correct" vs
"the classifier's prediction is incorrect"). Unlike a plain probability,
it separates belief, disbelief and uncertainty (vacuity of evidence),
which is what makes it useful for measuring decision degradation instead
of just accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Non-informative prior weight (Josang, Section 3.4.2, right after Eq. 3.11).
# Fixed at 2 so that a vacuous opinion with default base rate a = 0.5 maps
# to a uniform Beta(px, 1, 1) PDF. This constant is used throughout the
# book and should not be changed without a good reason.
NON_INFORMATIVE_PRIOR_WEIGHT: float = 2.0

# Small tolerance used when checking the additivity constraint b + d + u = 1,
# to absorb floating point rounding errors.
_ADDITIVITY_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class BinomialOpinion:
    """
    A binomial opinion omega_x = (b, d, u, a) about a binary variable X,
    as defined in Definition 3.1 (Josang, 2016, p. 24).

    Attributes:
        belief: b_x, belief mass in support of x being TRUE.
        disbelief: d_x, belief mass in support of x being FALSE.
        uncertainty: u_x, uncertainty mass representing vacuity of evidence.
        base_rate: a_x, prior probability of x in the absence of evidence.

    Invariants (Eq. 3.1):
        belief + disbelief + uncertainty == 1
        all of belief, disbelief, uncertainty, base_rate are in [0, 1]

    The class is immutable (frozen dataclass): opinions are treated as
    values, not mutable state, which mirrors how they are used
    mathematically in the book (every operator produces a *new* opinion).
    """

    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float

    def __post_init__(self) -> None:
        for name, value in (
            ("belief", self.belief),
            ("disbelief", self.disbelief),
            ("uncertainty", self.uncertainty),
            ("base_rate", self.base_rate),
        ):
            if not (0.0 - _ADDITIVITY_TOLERANCE <= value <= 1.0 + _ADDITIVITY_TOLERANCE):
                raise ValueError(
                    f"{name}={value!r} is out of range; all opinion "
                    f"parameters must lie in [0, 1]."
                )

        total = self.belief + self.disbelief + self.uncertainty
        if abs(total - 1.0) > _ADDITIVITY_TOLERANCE:
            raise ValueError(
                "Additivity constraint violated (Eq. 3.1): "
                f"belief + disbelief + uncertainty = {total!r}, expected 1.0. "
                "Check the evidence or parameters used to build this opinion."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def vacuous(cls, base_rate: float = 0.5) -> "BinomialOpinion":
        """
        Build a vacuous opinion (u = 1, b = d = 0), representing total
        absence of evidence (Section 3.4.1, item 5; Section 3.2, Table 3.2).

        This is the natural starting opinion for an agent that has not yet
        observed anything about the variable in question.
        """
        return cls(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=base_rate)

    @classmethod
    def from_evidence(
        cls,
        r: float,
        s: float,
        base_rate: float = 0.5,
        prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT,
    ) -> "BinomialOpinion":
        """
        Build a binomial opinion from Beta evidence counts (r, s), using
        the bijective mapping of Definition 3.3 / Eq. 3.11 (Josang, 2016,
        p. 28):

            b = r / (W + r + s)
            d = s / (W + r + s)
            u = W / (W + r + s)

        Here r is the amount of evidence observed in favour of x being
        TRUE, and s is the amount of evidence observed in favour of x
        being FALSE. W is the non-informative prior weight (default 2).

        This is the entry point we will later use to turn raw outputs
        from real agents (classifiers, LLMs, RL policies) into opinions,
        once we define how each agent type produces (r, s).
        """
        if r < 0 or s < 0:
            raise ValueError(
                f"Evidence counts must be non-negative (got r={r!r}, s={s!r})."
            )

        denominator = prior_weight + r + s
        belief = r / denominator
        disbelief = s / denominator
        uncertainty = prior_weight / denominator
        return cls(belief=belief, disbelief=disbelief, uncertainty=uncertainty, base_rate=base_rate)

    # ------------------------------------------------------------------
    # Beta PDF equivalence (Definition 3.3 / Eq. 3.11)
    # ------------------------------------------------------------------

    def to_evidence(self, prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT) -> tuple[float, float]:
        """
        Recover the equivalent Beta evidence counts (r, s) from this
        opinion, using the inverse mapping of Eq. 3.11.

            r = b * W / u
            s = d * W / u

        Raises:
            ValueError: if the opinion is dogmatic (u == 0). The book
            explicitly notes (Section 3.2, p. 20-21) that a dogmatic
            opinion corresponds to an infinite amount of evidence (a
            Dirac delta Beta PDF), which cannot be represented as a
            finite (r, s) pair.
        """
        if self.uncertainty == 0.0:
            raise ValueError(
                "Cannot recover finite evidence counts from a dogmatic "
                "opinion (u = 0): the equivalent Beta PDF is a Dirac "
                "delta, requiring infinite evidence (Josang, 2016, "
                "Section 3.2)."
            )

        r = self.belief * prior_weight / self.uncertainty
        s = self.disbelief * prior_weight / self.uncertainty
        return r, s

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def projected_probability(self) -> float:
        """
        Projected probability P(x) of the opinion, Eq. 3.2:

            P(x) = b_x + a_x * u_x

        This collapses the opinion back down to a single scalar
        probability, the way a classifier's confidence would be reported
        as a single number. Most of the value of the opinion is lost in
        this step, so prefer keeping the full (b, d, u, a) around for
        analysis, and use this only when a single number is required.
        """
        return self.belief + self.base_rate * self.uncertainty

    @property
    def variance(self, prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT) -> float:
        """
        Variance of the equivalent Beta PDF, Eq. 3.3 / Eq. 3.10:

            Var(x) = P(x) * (1 - P(x)) * u / (W + u)

        Intuition: variance grows with uncertainty mass u. A dogmatic
        opinion (u = 0) has zero variance (all evidence concentrated in
        one point); a vacuous opinion (u = 1) has the maximum possible
        variance for its projected probability.
        """
        p = self.projected_probability
        return p * (1.0 - p) * self.uncertainty / (prior_weight + self.uncertainty)

    # ------------------------------------------------------------------
    # Opinion classification (Section 3.2, Table 3.2; Section 3.4.1)
    # ------------------------------------------------------------------

    @property
    def is_vacuous(self) -> bool:
        """True if u = 1 (no evidence at all)."""
        return math.isclose(self.uncertainty, 1.0, abs_tol=_ADDITIVITY_TOLERANCE)

    @property
    def is_dogmatic(self) -> bool:
        """True if u = 0 (equivalent to a traditional, fully confident probability)."""
        return math.isclose(self.uncertainty, 0.0, abs_tol=_ADDITIVITY_TOLERANCE)

    @property
    def is_absolute(self) -> bool:
        """True if b = 1 or d = 1 (equivalent to Boolean TRUE/FALSE)."""
        return math.isclose(self.belief, 1.0, abs_tol=_ADDITIVITY_TOLERANCE) or math.isclose(
            self.disbelief, 1.0, abs_tol=_ADDITIVITY_TOLERANCE
        )

    @property
    def is_uncertain(self) -> bool:
        """True if 0 < u < 1 (partially uncertain, the most common case in practice)."""
        return 0.0 < self.uncertainty < 1.0

    # ------------------------------------------------------------------
    # Probabilistic opinion notation (Section 3.7.1, Definition 3.10/3.11)
    # ------------------------------------------------------------------

    def to_probabilistic_notation(self) -> tuple[float, float, float]:
        """
        Convert to the probabilistic opinion notation pi_x = (P(x), u_x, a_x)
        (Eq. 3.42, p. 48): the same opinion, expressed via projected
        probability instead of belief/disbelief mass. Useful for reporting
        to people more familiar with plain probabilities.
        """
        return self.projected_probability, self.uncertainty, self.base_rate

    @classmethod
    def from_probabilistic_notation(
        cls, projected_probability: float, uncertainty: float, base_rate: float
    ) -> "BinomialOpinion":
        """
        Inverse of to_probabilistic_notation, via Eq. 3.41 (p. 47):

            b_x = P(x) - a_x * u_x
            d_x = 1 - u_x - b_x
        """
        belief = projected_probability - base_rate * uncertainty
        disbelief = 1.0 - uncertainty - belief
        return cls(belief=belief, disbelief=disbelief, uncertainty=uncertainty, base_rate=base_rate)

    # ------------------------------------------------------------------
    # Sharp/vague/focal mass decomposition (Section 4.1) and mass-sum (4.2)
    # ------------------------------------------------------------------

    @property
    def sharp_belief_mass(self) -> float:
        """
        Sharp belief mass b^S_x, Eq. 4.1. Binomial opinions never contain
        vagueness (Section 4.1.2, p. 53: "In the case of binary domains,
        there can be no vague belief mass"), so this simply equals belief.
        """
        return self.belief

    @property
    def vague_belief_mass(self) -> float:
        """Always 0.0 for binomial opinions (Section 4.1.2)."""
        return 0.0

    @property
    def focal_uncertainty_mass(self) -> float:
        """Focal uncertainty mass u^F_x, Eq. 4.8: a_x * u_x."""
        return self.base_rate * self.uncertainty

    @property
    def mass_sum(self) -> tuple[float, float, float]:
        """
        Mass-sum triplet (sharp, vague, focal), Definition 4.6. Sums to
        projected_probability (Eq. 4.9): a decomposition of "how
        confident is this opinion" into evidence-backed vs. base-rate-
        backed confidence.
        """
        return (self.sharp_belief_mass, self.vague_belief_mass, self.focal_uncertainty_mass)

    def to_decision_option(self, label: str = "x", utility: float = 1.0):
        """Wrap this opinion as a decision.DecisionOption for use with choose_best_option."""
        from .decision import DecisionOption

        return DecisionOption(
            label=label,
            sharp_belief_mass=self.sharp_belief_mass,
            vague_belief_mass=self.vague_belief_mass,
            focal_uncertainty_mass=self.focal_uncertainty_mass,
            utility=utility,
        )

    # ------------------------------------------------------------------
    # Entropy (Section 4.7)
    # ------------------------------------------------------------------

    def _domain_distributions(self):
        """Internal helper: build {"x", "not_x"} dicts for entropy computation over the full binary domain."""
        p_x = self.projected_probability
        projected = {"x": p_x, "not_x": 1.0 - p_x}
        sharp = {"x": self.belief, "not_x": self.disbelief}
        vague = {"x": 0.0, "not_x": 0.0}
        focal = {"x": self.base_rate * self.uncertainty, "not_x": (1.0 - self.base_rate) * self.uncertainty}
        base_rates = {"x": self.base_rate, "not_x": 1.0 - self.base_rate}
        return projected, sharp, vague, focal, base_rates

    def opinion_entropy(self) -> float:
        """Opinion entropy H_P(omega_x), Eq. 4.55, over the full binary domain {x, not_x}."""
        from . import entropy as _entropy

        projected, *_ = self._domain_distributions()
        return _entropy.opinion_entropy(projected)

    def sharpness_entropy(self) -> float:
        """Sharpness entropy H_S(omega_x), Eq. 4.56."""
        from . import entropy as _entropy

        projected, sharp, _, _, _ = self._domain_distributions()
        return _entropy.sharpness_entropy(sharp, projected)

    def vagueness_entropy(self) -> float:
        """Vagueness entropy H_V(omega_x), Eq. 4.57. Always 0.0 for binomial opinions."""
        from . import entropy as _entropy

        projected, _, vague, _, _ = self._domain_distributions()
        return _entropy.vagueness_entropy(vague, projected)

    def uncertainty_entropy(self) -> float:
        """Uncertainty entropy H_U(omega_x), Eq. 4.58."""
        from . import entropy as _entropy

        projected, _, _, focal, _ = self._domain_distributions()
        return _entropy.uncertainty_entropy(focal, projected)

    def cross_entropy(self) -> float:
        """Base-rate to projected-probability cross entropy H_BP(omega_x), Eq. 4.60."""
        from . import entropy as _entropy

        projected, _, _, _, base_rates = self._domain_distributions()
        return _entropy.cross_entropy(base_rates, projected)

    # ------------------------------------------------------------------
    # Conflict (Section 4.8)
    # ------------------------------------------------------------------

    def degree_of_conflict_with(self, other: "BinomialOpinion") -> float:
        """
        Degree of conflict (Definition 4.20) between this opinion and
        another binomial opinion about the same variable.
        """
        from . import conflict as _conflict

        p_self = {"x": self.projected_probability, "not_x": 1.0 - self.projected_probability}
        p_other = {"x": other.projected_probability, "not_x": 1.0 - other.projected_probability}
        return _conflict.degree_of_conflict(p_self, self.uncertainty, p_other, other.uncertainty)

    # ------------------------------------------------------------------
    # Addition, subtraction, complement (Chapter 6)
    # ------------------------------------------------------------------

    def __add__(self, other: "BinomialOpinion") -> "BinomialOpinion":
        """
        Addition of opinions about two disjoint values (Definition 6.1):
        self + other gives the opinion about their union. This is NOT
        for combining multiple agents' opinions about the SAME variable
        -- that is fusion (Chapter 12, not yet implemented). See
        operators.add for the full docstring.
        """
        from .operators import add

        return add(self, other)

    def __sub__(self, other: "BinomialOpinion") -> "BinomialOpinion":
        """
        Subtraction of opinions (Definition 6.2): self - other, where
        self is the opinion about a union and other is the opinion about
        one of its parts. See operators.subtract for the full docstring.
        """
        from .operators import subtract

        return subtract(self, other)

    def __invert__(self) -> "BinomialOpinion":
        """
        Complement of the opinion (Definition 6.3): ~self gives the
        opinion about the complement value. See operators.complement.
        """
        from .operators import complement

        return complement(self)

    def __mul__(self, other: "BinomialOpinion") -> "BinomialOpinion":
        """Binomial multiplication (Definition 7.1): self AND other (assumes independence)."""
        from .operators import multiply

        return multiply(self, other)

    def __or__(self, other: "BinomialOpinion") -> "BinomialOpinion":
        """Binomial comultiplication (Definition 7.2): self OR other (assumes independence)."""
        from .operators import comultiply

        return comultiply(self, other)

    def __truediv__(self, other: "BinomialOpinion") -> "BinomialOpinion":
        """Binomial division (Definition 7.3): inverse of multiplication."""
        from .operators import divide

        return divide(self, other)

    def codivide(self, other: "BinomialOpinion") -> "BinomialOpinion":
        """Binomial codivision (Definition 7.4): inverse of comultiplication."""
        from .operators import codivide

        return codivide(self, other)

    def deduce(self, conditional_given_x: "BinomialOpinion", conditional_given_not_x: "BinomialOpinion") -> "BinomialOpinion":
        """Binomial conditional deduction (Definition 9.1): self is omega_x."""
        from .deduction import binomial_deduce

        return binomial_deduce(self, conditional_given_x, conditional_given_not_x)