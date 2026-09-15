"""
Hyper-opinions in Subjective Logic.

Hyper-opinions generalise multinomial opinions by allowing belief mass to
be assigned to composite values (sets of singleton outcomes), following:

    Josang, A. (2016). Subjective Logic. Springer. Section 3.6.

Use case: an agent that can only say "the answer is x1 or x2, but I can't
tell which" -- e.g. a classifier confident about ruling out some classes
but not about a single winner. This is different from uncertainty mass,
which expresses "I have no evidence at all".

Scope note: this module implements the *opinion representation* and its
closed-form mappings (projected probability, projection to a multinomial
opinion, and evidence mapping). It does NOT implement the Dirichlet HPDF
density (Section 3.6.3) or the Hyper-Dirichlet PDF (Section 3.6.5),
since the latter's normalising factor B(rX, aX) has no closed-form
expression and requires numerical integration (Josang, 2016, p. 45).
This mirrors the design already used for BinomialOpinion and
MultinomialOpinion, where the Beta/Dirichlet densities themselves are
also not implemented, only the opinion <-> evidence mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import FrozenSet, Hashable, Mapping

from .binomial import NON_INFORMATIVE_PRIOR_WEIGHT
from .domain import hyperdomain, relative_base_rate
from .multinomial import MultinomialOpinion

_ADDITIVITY_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class HyperOpinion:
    """
    A hyper-opinion omega_X = (b_X, u_X, a_X) on a hypervariable X over
    hyperdomain R(X), as defined in Definition 3.7 (Josang, 2016, p. 39).

    Attributes:
        belief_masses: b_X, belief mass distribution over R(X). Keys are
            frozensets of domain labels: frozenset({"x1"}) for a
            singleton, frozenset({"x1", "x2"}) for a composite value.
        uncertainty: u_X, a single scalar (vacuity of evidence).
        base_rates: a_X, base rate distribution over the *underlying
            domain* X only (not R(X)); composite base rates are derived
            on demand via Eq. 2.9 (see domain.py).

    Invariant (hypernomial additivity, Eq. 2.7):
        sum(belief_masses.values()) + uncertainty == 1
    """

    belief_masses: Mapping[FrozenSet[Hashable], float]
    uncertainty: float
    base_rates: Mapping[Hashable, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "belief_masses", MappingProxyType(dict(self.belief_masses)))
        object.__setattr__(self, "base_rates", MappingProxyType(dict(self.base_rates)))

        singletons = set(self.base_rates.keys())
        if len(singletons) <= 2:
            raise ValueError(
                "HyperOpinion requires an underlying domain of cardinality "
                f"k > 2 (Definition 3.7, Josang 2016, p. 39); got k = "
                f"{len(singletons)}. Use MultinomialOpinion or BinomialOpinion instead."
            )

        expected_hyperdomain = set(hyperdomain(singletons))
        actual_domain = set(self.belief_masses.keys())
        if actual_domain != expected_hyperdomain:
            missing = expected_hyperdomain - actual_domain
            extra = actual_domain - expected_hyperdomain
            raise ValueError(
                "belief_masses must be defined over the full hyperdomain R(X) "
                f"implied by base_rates' domain. Missing: {missing or None}, "
                f"unexpected: {extra or None}."
            )

        for value, mass in self.belief_masses.items():
            if not (0.0 - _ADDITIVITY_TOLERANCE <= mass <= 1.0 + _ADDITIVITY_TOLERANCE):
                raise ValueError(f"belief_masses[{set(value)!r}] = {mass!r} is out of range [0, 1].")

        if not (0.0 - _ADDITIVITY_TOLERANCE <= self.uncertainty <= 1.0 + _ADDITIVITY_TOLERANCE):
            raise ValueError(f"uncertainty={self.uncertainty!r} is out of range [0, 1].")

        total = sum(self.belief_masses.values()) + self.uncertainty
        if abs(total - 1.0) > _ADDITIVITY_TOLERANCE:
            raise ValueError(
                "Hypernomial additivity constraint violated (Eq. 2.7): "
                f"sum(belief_masses) + uncertainty = {total!r}, expected 1.0."
            )

        base_rate_sum = sum(self.base_rates.values())
        if abs(base_rate_sum - 1.0) > _ADDITIVITY_TOLERANCE:
            raise ValueError(
                "Base rate additivity constraint violated (Eq. 2.8): "
                f"sum(base_rates) = {base_rate_sum!r}, expected 1.0."
            )

    @property
    def domain(self) -> tuple:
        """The underlying singleton domain X (not the hyperdomain)."""
        return tuple(self.base_rates.keys())

    @property
    def hyperdomain_values(self) -> tuple:
        """The full hyperdomain R(X) this opinion's belief mass is defined over."""
        return tuple(self.belief_masses.keys())

    @classmethod
    def vacuous(cls, base_rates: Mapping[Hashable, float]) -> "HyperOpinion":
        """A vacuous hyper-opinion (u = 1, all belief masses 0) over R(X)."""
        values = hyperdomain(base_rates.keys())
        return cls(belief_masses={v: 0.0 for v in values}, uncertainty=1.0, base_rates=dict(base_rates))

    @classmethod
    def from_evidence(
        cls,
        evidence: Mapping[FrozenSet[Hashable], float],
        base_rates: Mapping[Hashable, float],
        prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT,
    ) -> "HyperOpinion":
        """
        Build a hyper-opinion from Dirichlet HPDF evidence counts r_X(x)
        over R(X), using the forward mapping of Definition 3.9 / Eq. 3.35
        (p. 43) -- structurally identical to the multinomial mapping
        (Eq. 3.23), just applied to R(X) instead of X:

            b_X(x) = r_X(x) / (W + sum(r_X))
            u_X    = W / (W + sum(r_X))
        """
        expected = set(hyperdomain(base_rates.keys()))
        if set(evidence.keys()) != expected:
            raise ValueError("evidence must be defined over the full hyperdomain R(X).")
        for x, r in evidence.items():
            if r < 0:
                raise ValueError(f"Evidence must be non-negative (got evidence[{set(x)!r}]={r!r}).")

        total_evidence = sum(evidence.values())
        denominator = prior_weight + total_evidence
        belief_masses = {x: r / denominator for x, r in evidence.items()}
        uncertainty = prior_weight / denominator
        return cls(belief_masses=belief_masses, uncertainty=uncertainty, base_rates=dict(base_rates))

    def to_evidence(self, prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT) -> dict:
        """Inverse of from_evidence: r_X(x) = W * b_X(x) / u_X for x in R(X)."""
        if self.uncertainty == 0.0:
            raise ValueError(
                "Cannot recover finite evidence from a dogmatic opinion (u = 0): "
                "requires infinite evidence."
            )
        return {x: b * prior_weight / self.uncertainty for x, b in self.belief_masses.items()}

    @property
    def projected_probabilities(self) -> dict:
        """
        Projected probability distribution P_X(x) for each singleton x in
        the underlying domain X, Eq. 3.28 (p. 40):

            P_X(x) = sum_{xi in R(X)} a_X(x|xi) * b_X(xi) + a_X(x) * u_X

        This sums to 1 over X (Eq. 3.29.a): projecting back down to
        singletons restores ordinary probability additivity, even though
        the hyperdomain's own belief masses are super-additive.
        """
        result = {}
        for x in self.domain:
            x_singleton = frozenset((x,))
            contribution = sum(
                relative_base_rate(self.base_rates, x_singleton, xi) * b
                for xi, b in self.belief_masses.items()
            )
            result[x] = contribution + self.base_rates[x] * self.uncertainty
        return result

    def to_multinomial(self) -> MultinomialOpinion:
        """
        Project this hyper-opinion onto a multinomial opinion with the
        same projected probability distribution, Eq. 3.30 (p. 40):

            b'_X(x) = sum_{xi in R(X)} a_X(x|xi) * b_X(xi)

        The book notes (p. 40) that P(omega_X) == P(omega'_X): projecting
        preserves the projected probability exactly. This is the
        recommended way to "flatten" a hyper-opinion when a simpler
        representation is needed downstream -- e.g. uncertainty-
        maximisation (Section 3.5.6) has no direct hyper-opinion
        equivalent (p. 39), so it must go through this projection first.
        """
        multinomial_belief = {}
        for x in self.domain:
            x_singleton = frozenset((x,))
            multinomial_belief[x] = sum(
                relative_base_rate(self.base_rates, x_singleton, xi) * b
                for xi, b in self.belief_masses.items()
            )
        return MultinomialOpinion(
            belief_masses=multinomial_belief, uncertainty=self.uncertainty, base_rates=dict(self.base_rates)
        )

    @property
    def vagueness(self) -> float:
        """
        Total belief mass assigned to composite (non-singleton) values in
        R(X): a simple measure of "vagueness", as opposed to
        "uncertainty" mass. Full treatment is in Section 4.1.2 (not yet
        implemented); included here since it falls directly out of the
        representation already built.
        """
        return sum(mass for value, mass in self.belief_masses.items() if len(value) >= 2)