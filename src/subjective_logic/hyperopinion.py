"""
Hyper-opinions in Subjective Logic.

Hyper-opinions generalise multinomial opinions by allowing belief mass to
be assigned to composite values (sets of singleton outcomes), following:

    Josang, A. (2016). Subjective Logic. Springer. Section 3.6, and
    Sections 4.1-4.2, 4.7-4.8 for the sharp/vague/focal decomposition,
    entropy and conflict measures, which are where vagueness (unique to
    hyper-opinions) actually matters.

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
from .domain import base_rate_of_value, hyperdomain, relative_base_rate
from .multinomial import MultinomialOpinion

_ADDITIVITY_TOLERANCE: float = 1e-9
_TOLERANCE: float = 1e-9


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
        (Eq. 3.23), just applied to R(X) instead of X.
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

    # ------------------------------------------------------------------
    # Projected probability (Eq. 3.28), generalised to any x in R(X)
    # ------------------------------------------------------------------

    def projected_probability_of(self, x: FrozenSet[Hashable]) -> float:
        """
        Projected probability for ANY value x in R(X) (not just
        singletons), Eq. 3.28 (p. 40):

            P_X(x) = sum_{xi in R(X)} a_X(x|xi) * b_X(xi) + a_X(x) * u_X

        Generalises `projected_probabilities` (singletons only, i.e. x in
        X) to arbitrary hyperdomain values, needed for the mass-sum
        additivity check in Section 4.2 (Eq. 4.9), which applies to
        composites too.
        """
        contribution = sum(
            relative_base_rate(self.base_rates, x, xi) * b for xi, b in self.belief_masses.items()
        )
        return contribution + base_rate_of_value(self.base_rates, x) * self.uncertainty

    @property
    def projected_probabilities(self) -> dict:
        """Projected probability distribution P_X(x) for each singleton x in domain X (Eq. 3.28)."""
        return {x: self.projected_probability_of(frozenset((x,))) for x in self.domain}

    def to_multinomial(self) -> MultinomialOpinion:
        """
        Project this hyper-opinion onto a multinomial opinion with the
        same projected probability distribution, Eq. 3.30 (p. 40).
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

    # ------------------------------------------------------------------
    # Sharp/vague/focal mass decomposition (Section 4.1) and mass-sum (4.2)
    # ------------------------------------------------------------------

    def sharp_belief_mass(self, x: FrozenSet[Hashable]) -> float:
        """
        Sharp belief mass b^S_X(x), Definition 4.1 / Eq. 4.1 (p. 51-52):
        belief mass that discriminates specifically in favour of x,
        summed from every value fully contained in x.

            b^S_X(x) = sum_{xi subseteq x} b_X(xi)
        """
        return sum(b for xi, b in self.belief_masses.items() if xi <= x)

    def vague_belief_mass(self, x: FrozenSet[Hashable]) -> float:
        """
        Vague belief mass b^V_X(x), Definition 4.3 / Eq. 4.3 (p. 52-53):
        belief mass "leaking in" to x from composite values that overlap
        x without being contained in it, weighted by relative base rate.

            b^V_X(x) = sum_{xi in C(X), xi not subseteq x} a_X(x|xi) * b_X(xi)

        Validated against the worked example in Section 4.1.3 (Eq. 4.6-4.7).
        """
        total = 0.0
        for xi, b in self.belief_masses.items():
            if len(xi) < 2:
                continue  # only composite values xi in C(X)
            if xi <= x:
                continue  # already counted as sharp belief mass for x
            total += relative_base_rate(self.base_rates, x, xi) * b
        return total

    def focal_uncertainty_mass(self, x: FrozenSet[Hashable]) -> float:
        """Focal uncertainty mass u^F_X(x), Definition 4.5 / Eq. 4.8 (p. 55): a_X(x) * u_X."""
        return base_rate_of_value(self.base_rates, x) * self.uncertainty

    def mass_sum(self, x: FrozenSet[Hashable]) -> tuple:
        """
        Mass-sum triplet (sharp, vague, focal) for value x, Definition 4.6
        / Eq. 4.10. Sums to projected_probability_of(x) (Eq. 4.9).
        """
        return (self.sharp_belief_mass(x), self.vague_belief_mass(x), self.focal_uncertainty_mass(x))

    @property
    def total_sharp_belief_mass(self) -> float:
        """Total sharp belief mass b^TS_X, Definition 4.2 / Eq. 4.2: sum of belief mass on singletons."""
        return sum(self.belief_masses[frozenset((x,))] for x in self.domain)

    @property
    def total_vague_belief_mass(self) -> float:
        """
        Total vague belief mass b^TV_X, Definition 4.4 / Eq. 4.4: sum of
        belief mass assigned to composite values. Equivalent to the
        `vagueness` property already defined in this class.
        """
        return sum(mass for value, mass in self.belief_masses.items() if len(value) >= 2)

    @property
    def total_mass_sum(self) -> tuple:
        """Total mass-sum (b^TS_X, b^TV_X, u_X), Definition 4.7 / Eq. 4.12. Sums to 1 (Eq. 4.11)."""
        return (self.total_sharp_belief_mass, self.total_vague_belief_mass, self.uncertainty)

    @property
    def vagueness(self) -> float:
        """Total belief mass assigned to composite (non-singleton) values in R(X). Same as total_vague_belief_mass."""
        return self.total_vague_belief_mass

    def to_decision_option(self, x: FrozenSet[Hashable], label: str = None, utility: float = 1.0):
        """Wrap value x of this opinion as a decision.DecisionOption for use with choose_best_option."""
        from .decision import DecisionOption

        return DecisionOption(
            label=label if label is not None else str(set(x)),
            sharp_belief_mass=self.sharp_belief_mass(x),
            vague_belief_mass=self.vague_belief_mass(x),
            focal_uncertainty_mass=self.focal_uncertainty_mass(x),
            utility=utility,
        )

    # ------------------------------------------------------------------
    # Entropy (Section 4.7)
    # ------------------------------------------------------------------

    def opinion_entropy(self) -> float:
        """Opinion entropy H_P(omega_X), Eq. 4.55, over domain X (singletons only)."""
        from . import entropy as _entropy

        return _entropy.opinion_entropy(self.projected_probabilities)

    def sharpness_entropy(self) -> float:
        """Sharpness entropy H_S(omega_X), Eq. 4.56."""
        from . import entropy as _entropy

        sharp = {x: self.sharp_belief_mass(frozenset((x,))) for x in self.domain}
        return _entropy.sharpness_entropy(sharp, self.projected_probabilities)

    def vagueness_entropy(self) -> float:
        """Vagueness entropy H_V(omega_X), Eq. 4.57."""
        from . import entropy as _entropy

        vague = {x: self.vague_belief_mass(frozenset((x,))) for x in self.domain}
        return _entropy.vagueness_entropy(vague, self.projected_probabilities)

    def uncertainty_entropy(self) -> float:
        """Uncertainty entropy H_U(omega_X), Eq. 4.58."""
        from . import entropy as _entropy

        focal = {x: self.focal_uncertainty_mass(frozenset((x,))) for x in self.domain}
        return _entropy.uncertainty_entropy(focal, self.projected_probabilities)

    def cross_entropy(self) -> float:
        """Base-rate to projected-probability cross entropy H_BP(omega_X), Eq. 4.60."""
        from . import entropy as _entropy

        return _entropy.cross_entropy(self.base_rates, self.projected_probabilities)

    # ------------------------------------------------------------------
    # Conflict (Section 4.8)
    # ------------------------------------------------------------------

    def degree_of_conflict_with(self, other: "HyperOpinion") -> float:
        """Degree of conflict (Definition 4.20) with another hyper-opinion over the same domain."""
        from . import conflict as _conflict

        return _conflict.degree_of_conflict(
            self.projected_probabilities, self.uncertainty, other.projected_probabilities, other.uncertainty
        )

    def fuse_constraint(self, other: "HyperOpinion") -> "HyperOpinion":
        """
        Belief constraint fusion (Definition 12.3, Eq. 12.1-12.2): an
        extension of Dempster's rule, suited for merging PREFERENCES
        (each source narrows down acceptable values) rather than
        evidence about a shared truth (p. 215-216: the book explicitly
        warns this is NOT the right operator for combining multiple
        agents' evidence about the same fact -- use cumulative,
        averaging or weighted fusion for that instead).

        Raises:
            ValueError: if the two opinions are totally conflicting
            (Con = 1, Eq. 12.4) -- no compromise exists (Section 12.2.6).
        """
        from .fusion import _conflict, _harmony

        conflict = _conflict(self.belief_masses, other.belief_masses)
        if conflict >= 1.0 - _TOLERANCE:
            raise ValueError(
                "Cannot apply belief constraint fusion: the opinions are totally conflicting "
                "(Con = 1); no compromise exists (Section 12.2.6, p. 225)."
            )

        ua, ub = self.uncertainty, other.uncertainty
        denom = 1.0 - conflict

        if ua < 1.0 or ub < 1.0:
            belief = {
                value: _harmony(value, self.belief_masses, ua, other.belief_masses, ub) / denom
                for value in self.hyperdomain_values
            }
            uncertainty = (ua * ub) / denom
            base_rates = {
                x: (self.base_rates[x] * (1 - ua) + other.base_rates[x] * (1 - ub)) / (2 - ua - ub)
                for x in self.domain
            }
        else:  # ua == ub == 1
            belief = {value: 0.0 for value in self.hyperdomain_values}
            uncertainty = 1.0
            base_rates = {x: (self.base_rates[x] + other.base_rates[x]) / 2 for x in self.domain}

        return HyperOpinion(belief_masses=belief, uncertainty=uncertainty, base_rates=base_rates)