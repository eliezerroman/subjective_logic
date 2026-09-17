"""
Multinomial opinions in Subjective Logic.

This module generalises binomial opinions (see binomial.py) to domains
with more than two values, following:

    Josang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
    Uncertainty. Springer. Chapter 3, Section 3.5.

A multinomial opinion is useful when an agent's decision is not simply
"correct / incorrect" but a choice among several classes (e.g. a
multi-class classifier's prediction, or one action among several taken
by an RL policy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Hashable, Mapping

from .binomial import NON_INFORMATIVE_PRIOR_WEIGHT

_ADDITIVITY_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class MultinomialOpinion:
    """
    A multinomial opinion omega_X = (b_X, u_X, a_X) about a random
    variable X over a domain of cardinality k > 2, as defined in
    Definition 3.4 (Josang, 2016, p. 30).

    Attributes:
        belief_masses: b_X, a belief mass distribution over the domain
            (one value per outcome; does not need to sum to 1 by itself).
        uncertainty: u_X, a single scalar representing vacuity of evidence,
            shared across all outcomes.
        base_rates: a_X, a base rate (prior probability) distribution over
            the domain, summing to 1 (Eq. 2.8).

    Invariants:
        sum(belief_masses.values()) + uncertainty == 1   (Eq. 2.6)
        sum(base_rates.values()) == 1                     (Eq. 2.8)
        len(domain) > 2  (binary domains should use BinomialOpinion)

    The class is immutable: belief_masses and base_rates are stored as
    read-only mappings (MappingProxyType) so an opinion cannot be mutated
    after construction, mirroring how BinomialOpinion behaves.
    """

    belief_masses: Mapping[Hashable, float]
    uncertainty: float
    base_rates: Mapping[Hashable, float]

    def __post_init__(self) -> None:
        # Freeze the mappings passed in, so mutating the dict the caller
        # used to build this opinion cannot silently change it afterwards.
        object.__setattr__(self, "belief_masses", MappingProxyType(dict(self.belief_masses)))
        object.__setattr__(self, "base_rates", MappingProxyType(dict(self.base_rates)))

        belief_domain = set(self.belief_masses.keys())
        base_rate_domain = set(self.base_rates.keys())
        if belief_domain != base_rate_domain:
            raise ValueError(
                "belief_masses and base_rates must be defined over the "
                f"same domain; got {sorted(belief_domain, key=str)} vs "
                f"{sorted(base_rate_domain, key=str)}."
            )

        if len(belief_domain) < 2:
            raise ValueError(
                "MultinomialOpinion requires a domain of cardinality k >= 2; "
                f"got k = {len(belief_domain)}."
            )

        for label, mapping in (("belief_masses", self.belief_masses), ("base_rates", self.base_rates)):
            for value_name, value in mapping.items():
                if not (0.0 - _ADDITIVITY_TOLERANCE <= value <= 1.0 + _ADDITIVITY_TOLERANCE):
                    raise ValueError(f"{label}[{value_name!r}] = {value!r} is out of range [0, 1].")

        if not (0.0 - _ADDITIVITY_TOLERANCE <= self.uncertainty <= 1.0 + _ADDITIVITY_TOLERANCE):
            raise ValueError(f"uncertainty={self.uncertainty!r} is out of range [0, 1].")

        belief_sum = sum(self.belief_masses.values())
        total = belief_sum + self.uncertainty
        if abs(total - 1.0) > _ADDITIVITY_TOLERANCE:
            raise ValueError(
                "Multinomial additivity constraint violated (Eq. 2.6): "
                f"sum(belief_masses) + uncertainty = {total!r}, expected 1.0."
            )

        base_rate_sum = sum(self.base_rates.values())
        if abs(base_rate_sum - 1.0) > _ADDITIVITY_TOLERANCE:
            raise ValueError(
                "Base rate additivity constraint violated (Eq. 2.8): "
                f"sum(base_rates) = {base_rate_sum!r}, expected 1.0."
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @property
    def domain(self) -> tuple:
        """The ordered set of outcome labels this opinion is defined over."""
        return tuple(self.belief_masses.keys())

    @staticmethod
    def default_base_rates(domain) -> dict:
        """
        Default (non-informative) base rate distribution: 1/k for each of
        the k values in the domain (Section 2.6, p. 14: "the default base
        rate of each singleton value in the domain is 1/k").
        """
        domain = list(domain)
        k = len(domain)
        return {x: 1.0 / k for x in domain}

    @classmethod
    def vacuous(cls, base_rates: Mapping[Hashable, float]) -> "MultinomialOpinion":
        """
        Build a vacuous multinomial opinion (u = 1, all belief masses 0),
        representing total absence of evidence over the given domain.
        """
        return cls(
            belief_masses={x: 0.0 for x in base_rates},
            uncertainty=1.0,
            base_rates=dict(base_rates),
        )

    @classmethod
    def from_evidence(
        cls,
        evidence: Mapping[Hashable, float],
        base_rates: Mapping[Hashable, float],
        prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT,
    ) -> "MultinomialOpinion":
        """
        Build a multinomial opinion from Dirichlet evidence counts r_X(x),
        using the forward mapping of Definition 3.6 / Eq. 3.23 (p. 37):

            b_X(x) = r_X(x) / (W + sum(r_X))
            u_X    = W / (W + sum(r_X))

        Args:
            evidence: r_X(x) for each outcome x in the domain (non-negative).
            base_rates: a_X(x) for each outcome x (must sum to 1).
            prior_weight: W, the non-informative prior weight (default 2).
        """
        if set(evidence.keys()) != set(base_rates.keys()):
            raise ValueError("evidence and base_rates must be defined over the same domain.")

        for x, r in evidence.items():
            if r < 0:
                raise ValueError(f"Evidence must be non-negative (got evidence[{x!r}]={r!r}).")

        total_evidence = sum(evidence.values())
        denominator = prior_weight + total_evidence
        belief_masses = {x: r / denominator for x, r in evidence.items()}
        uncertainty = prior_weight / denominator
        return cls(belief_masses=belief_masses, uncertainty=uncertainty, base_rates=dict(base_rates))

    # ------------------------------------------------------------------
    # Dirichlet PDF equivalence (Definition 3.6 / Eq. 3.23)
    # ------------------------------------------------------------------

    def to_evidence(self, prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT) -> dict:
        """
        Recover the equivalent Dirichlet evidence r_X(x) for each outcome,
        using the inverse mapping of Eq. 3.23:

            r_X(x) = W * b_X(x) / u_X

        Raises:
            ValueError: if the opinion is dogmatic (u = 0), since that
            requires infinite evidence (same reasoning as in
            BinomialOpinion.to_evidence).
        """
        if self.uncertainty == 0.0:
            raise ValueError(
                "Cannot recover finite evidence from a dogmatic opinion "
                "(u = 0): the equivalent Dirichlet PDF requires infinite "
                "evidence (Josang, 2016, Section 3.2)."
            )
        return {x: b * prior_weight / self.uncertainty for x, b in self.belief_masses.items()}

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def projected_probabilities(self) -> dict:
        """
        Projected probability distribution P_X(x), Eq. 3.12:

            P_X(x) = b_X(x) + a_X(x) * u_X    for every x in the domain
        """
        return {x: self.belief_masses[x] + self.base_rates[x] * self.uncertainty for x in self.domain}

    def variance(self, x: Hashable, prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT) -> float:
        """
        Variance of the equivalent Dirichlet PDF for outcome x, Eq. 3.13/3.18:

            Var_X(x) = P_X(x) * (1 - P_X(x)) * u_X / (W + u_X)
        """
        p = self.projected_probabilities[x]
        return p * (1.0 - p) * self.uncertainty / (prior_weight + self.uncertainty)

    # ------------------------------------------------------------------
    # Uncertainty-maximisation (Section 3.5.6, Eq. 3.25-3.27)
    # ------------------------------------------------------------------

    def uncertainty_maximized(self) -> "MultinomialOpinion":
        """
        Compute the uncertainty-maximised opinion omega_double_dot_X that
        preserves the same projected probability distribution P_X, but
        pushes as much belief mass as possible into uncertainty mass
        (Section 3.5.6, p. 37-38).

        This is the opinion class used for epistemic reasoning (Section
        3.3): it represents the most "honest" (least dogmatic) opinion
        consistent with a given projected probability.

            u_double_dot_X = min_i( P_X(x_i) / a_X(x_i) )          (Eq. 3.27)
            b_double_dot_X(x) = P_X(x) - a_X(x) * u_double_dot_X    (from Eq. 3.24)

        Raises:
            ValueError: if every base rate is zero, which makes Eq. 3.27
            undefined.
        """
        projected = self.projected_probabilities

        ratios = {x: projected[x] / self.base_rates[x] for x in self.domain if self.base_rates[x] > 0}
        if not ratios:
            raise ValueError("Cannot uncertainty-maximise an opinion where all base rates are zero.")

        u_max = min(ratios.values())

        new_belief = {}
        for x in self.domain:
            value = projected[x] - self.base_rates[x] * u_max
            # Clamp tiny negative values caused by floating point rounding;
            # anything beyond the additivity tolerance indicates a real bug
            # rather than rounding noise.
            if -_ADDITIVITY_TOLERANCE <= value < 0.0:
                value = 0.0
            new_belief[x] = value

        return MultinomialOpinion(belief_masses=new_belief, uncertainty=u_max, base_rates=dict(self.base_rates))

    # ------------------------------------------------------------------
    # Opinion classification (mirrors BinomialOpinion; Section 3.2)
    # ------------------------------------------------------------------

    @property
    def is_vacuous(self) -> bool:
        """True if u_X = 1 (no evidence at all)."""
        return math.isclose(self.uncertainty, 1.0, abs_tol=_ADDITIVITY_TOLERANCE)

    @property
    def is_dogmatic(self) -> bool:
        """True if u_X = 0 (equivalent to a traditional probability distribution)."""
        return math.isclose(self.uncertainty, 0.0, abs_tol=_ADDITIVITY_TOLERANCE)

    @property
    def is_absolute(self) -> bool:
        """True if some b_X(x) = 1 (absolute certainty that x is TRUE)."""
        return any(math.isclose(b, 1.0, abs_tol=_ADDITIVITY_TOLERANCE) for b in self.belief_masses.values())

    # ------------------------------------------------------------------
    # Probabilistic opinion notation (Section 3.7.1, Definition 3.10/3.11)
    # ------------------------------------------------------------------

    def to_probabilistic_notation(self) -> tuple[dict, float, dict]:
        """
        Convert to the probabilistic opinion notation pi_X = (P_X, u_X, a_X)
        (Eq. 3.40, p. 47).
        """
        return self.projected_probabilities, self.uncertainty, dict(self.base_rates)

    @classmethod
    def from_probabilistic_notation(
        cls, projected_probabilities: Mapping, uncertainty: float, base_rates: Mapping
    ) -> "MultinomialOpinion":
        """
        Inverse mapping via Eq. 3.41 (p. 47): b_X(x) = P_X(x) - a_X(x) * u_X.
        """
        belief_masses = {x: p - base_rates[x] * uncertainty for x, p in projected_probabilities.items()}
        return cls(belief_masses=belief_masses, uncertainty=uncertainty, base_rates=dict(base_rates))

    # ------------------------------------------------------------------
    # Sharp/vague/focal mass decomposition (Section 4.1) and mass-sum (4.2)
    # ------------------------------------------------------------------

    def sharp_belief_mass(self, x) -> float:
        """Sharp belief mass b^S_X(x), Eq. 4.1. Multinomial opinions never contain vagueness (Sec. 4.1.2)."""
        return self.belief_masses[x]

    def vague_belief_mass(self, x) -> float:
        """Always 0.0 for multinomial opinions (Section 4.1.2)."""
        return 0.0

    def focal_uncertainty_mass(self, x) -> float:
        """Focal uncertainty mass u^F_X(x), Eq. 4.8: a_X(x) * u_X."""
        return self.base_rates[x] * self.uncertainty

    def mass_sum(self, x) -> tuple[float, float, float]:
        """Mass-sum triplet (sharp, vague, focal) for outcome x, Definition 4.6."""
        return (self.sharp_belief_mass(x), self.vague_belief_mass(x), self.focal_uncertainty_mass(x))

    def to_decision_option(self, x, label: str = None, utility: float = 1.0):
        """Wrap outcome x of this opinion as a decision.DecisionOption for use with choose_best_option."""
        from .decision import DecisionOption

        return DecisionOption(
            label=label if label is not None else str(x),
            sharp_belief_mass=self.sharp_belief_mass(x),
            vague_belief_mass=self.vague_belief_mass(x),
            focal_uncertainty_mass=self.focal_uncertainty_mass(x),
            utility=utility,
        )

    # ------------------------------------------------------------------
    # Entropy (Section 4.7)
    # ------------------------------------------------------------------

    def opinion_entropy(self) -> float:
        """Opinion entropy H_P(omega_X), Eq. 4.55."""
        from . import entropy as _entropy

        return _entropy.opinion_entropy(self.projected_probabilities)

    def sharpness_entropy(self) -> float:
        """Sharpness entropy H_S(omega_X), Eq. 4.56."""
        from . import entropy as _entropy

        sharp = {x: self.sharp_belief_mass(x) for x in self.domain}
        return _entropy.sharpness_entropy(sharp, self.projected_probabilities)

    def vagueness_entropy(self) -> float:
        """Vagueness entropy H_V(omega_X), Eq. 4.57. Always 0.0 for multinomial opinions."""
        from . import entropy as _entropy

        vague = {x: 0.0 for x in self.domain}
        return _entropy.vagueness_entropy(vague, self.projected_probabilities)

    def uncertainty_entropy(self) -> float:
        """Uncertainty entropy H_U(omega_X), Eq. 4.58."""
        from . import entropy as _entropy

        focal = {x: self.focal_uncertainty_mass(x) for x in self.domain}
        return _entropy.uncertainty_entropy(focal, self.projected_probabilities)

    def cross_entropy(self) -> float:
        """Base-rate to projected-probability cross entropy H_BP(omega_X), Eq. 4.60."""
        from . import entropy as _entropy

        return _entropy.cross_entropy(self.base_rates, self.projected_probabilities)

    # ------------------------------------------------------------------
    # Conflict (Section 4.8)
    # ------------------------------------------------------------------

    def degree_of_conflict_with(self, other: "MultinomialOpinion") -> float:
        """Degree of conflict (Definition 4.20) with another multinomial opinion over the same domain."""
        from . import conflict as _conflict

        return _conflict.degree_of_conflict(
            self.projected_probabilities, self.uncertainty, other.projected_probabilities, other.uncertainty
        )

    def __mul__(self, other: "MultinomialOpinion") -> "MultinomialOpinion":
        """Normal multinomial multiplication (Section 8.1.2): the recommended default method."""
        from .multinomial_operators import normal_multiply

        return normal_multiply(self, other)

    def proportional_multiply(self, other: "MultinomialOpinion") -> "MultinomialOpinion":
        """Proportional multinomial multiplication (Section 8.1.4): faster, slightly more aggressive alternative."""
        from .multinomial_operators import proportional_multiply

        return proportional_multiply(self, other)

    def divide(self, opinion_y: "MultinomialOpinion") -> "MultinomialOpinion":
        """Averaging proportional division (Section 8.3.2). Synthetic -- see multinomial_operators docstring."""
        from .multinomial_operators import averaging_proportional_divide

        return averaging_proportional_divide(self, opinion_y)

    def selective_divide(self, opinion_y: "MultinomialOpinion", observed_value) -> "MultinomialOpinion":
        """Selective division (Section 8.3.3), assuming opinion_y asserts observed_value as certain."""
        from .multinomial_operators import selective_divide

        return selective_divide(self, opinion_y, observed_value)

    def deduce(self, conditionals: Mapping, base_rates_y: Mapping = None) -> "MultinomialOpinion":
        """Multinomial conditional deduction (Definition 9.2): self is the parent evidence opinion omega_X."""
        from .deduction import multinomial_deduce

        return multinomial_deduce(self, conditionals, base_rates_y)

    def abduce(self, conditionals, base_rates_x, base_rates_y=None) -> "MultinomialOpinion":
        """Multinomial abduction (Definition 10.7): self is the evidence opinion omega_Y."""
        from .abduction import multinomial_abduce

        return multinomial_abduce(self, conditionals, base_rates_x, base_rates_y)

    def joint_with(self, conditionals, base_rates_y=None) -> "MultinomialOpinion":
        """Joint opinion omega_YX (Section 11.2): self is the parent evidence opinion omega_X."""
        from .joint import joint_opinion

        return joint_opinion(self, conditionals, base_rates_y)

    def marginalize(self) -> tuple:
        """Marginalise this joint opinion (domain of (y, x) tuples) onto its two factor variables (Section 11.3.1)."""
        from .joint import marginalize as _marginalize

        return _marginalize(self)

    def fuse_cumulative(self, other: "MultinomialOpinion") -> "MultinomialOpinion":
        """Aleatory cumulative belief fusion (Definition 12.5): self and other are INDEPENDENT sources."""
        from .fusion import cumulative_fusion

        belief, u, base_rates = cumulative_fusion(
            self.belief_masses, self.uncertainty, self.base_rates,
            other.belief_masses, other.uncertainty, other.base_rates, self.domain,
        )
        return MultinomialOpinion(belief_masses=belief, uncertainty=u, base_rates=base_rates)

    def fuse_epistemic_cumulative(self, other: "MultinomialOpinion") -> "MultinomialOpinion":
        """Epistemic cumulative belief fusion (Definition 12.6): cumulative fusion, then uncertainty-maximised."""
        return self.fuse_cumulative(other).uncertainty_maximized()

    def fuse_averaging(self, other: "MultinomialOpinion") -> "MultinomialOpinion":
        """Averaging belief fusion (Definition 12.7): self and other are DEPENDENT sources."""
        from .fusion import averaging_fusion

        belief, u, base_rates = averaging_fusion(
            self.belief_masses, self.uncertainty, self.base_rates,
            other.belief_masses, other.uncertainty, other.base_rates, self.domain,
        )
        return MultinomialOpinion(belief_masses=belief, uncertainty=u, base_rates=base_rates)

    def fuse_weighted(self, other: "MultinomialOpinion") -> "MultinomialOpinion":
        """Weighted belief fusion (Definition 12.8): averaging weighted by each source's confidence."""
        from .fusion import weighted_fusion

        belief, u, base_rates = weighted_fusion(
            self.belief_masses, self.uncertainty, self.base_rates,
            other.belief_masses, other.uncertainty, other.base_rates, self.domain,
        )
        return MultinomialOpinion(belief_masses=belief, uncertainty=u, base_rates=base_rates)

    def unfuse_cumulative(self, known: "MultinomialOpinion") -> "MultinomialOpinion":
        """Cumulative unfusion (Definition 13.1)."""
        from .unfusion import cumulative_unfusion

        belief, u = cumulative_unfusion(self.belief_masses, self.uncertainty, known.belief_masses, known.uncertainty, self.domain)
        return MultinomialOpinion(belief_masses=belief, uncertainty=u, base_rates=dict(self.base_rates))

    def unfuse_averaging(self, known: "MultinomialOpinion") -> "MultinomialOpinion":
        """Averaging unfusion (Definition 13.2)."""
        from .unfusion import averaging_unfusion

        belief, u = averaging_unfusion(self.belief_masses, self.uncertainty, known.belief_masses, known.uncertainty, self.domain)
        return MultinomialOpinion(belief_masses=belief, uncertainty=u, base_rates=dict(self.base_rates))

    def split_cumulative(self, phi: float) -> tuple:
        """Cumulative fission (Definition 13.3)."""
        from .unfusion import cumulative_fission

        b1, u1, b2, u2 = cumulative_fission(self.belief_masses, self.uncertainty, phi, self.domain)
        return (
            MultinomialOpinion(belief_masses=b1, uncertainty=u1, base_rates=dict(self.base_rates)),
            MultinomialOpinion(belief_masses=b2, uncertainty=u2, base_rates=dict(self.base_rates)),
        )