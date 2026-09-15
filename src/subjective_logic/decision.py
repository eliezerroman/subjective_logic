"""
Decision making from opinions.

Implements Sections 4.2-4.4 (Josang, 2016, pp. 56-64): decomposing an
opinion's projected probability into sharp belief, vague belief and
focal uncertainty mass (the "mass-sum"), normalising by utility so
options over different domains/agents can be compared, and applying the
book's 3-level decision criteria to pick the best option.

Use case for this project: weighting a decision by an agent's opinion
rather than by a bare projected probability -- e.g. picking the more
trustworthy of two agents' recommendations, where "trustworthy" accounts
for how much of their confidence is backed by sharp evidence versus
vague or absent evidence, not just the raw probability number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class DecisionOption:
    """
    One candidate option in a decision, characterised by its mass-sum
    (Definition 4.6, Eq. 4.10) and its utility.

    Attributes:
        label: a human-readable name for this option.
        sharp_belief_mass: b^S_X(x), Eq. 4.1.
        vague_belief_mass: b^V_X(x), Eq. 4.3 (0.0 for binomial/multinomial).
        focal_uncertainty_mass: u^F_X(x), Eq. 4.8.
        utility: lambda_X(x), the payoff/value of this option if it
            occurs (can be negative). Defaults to 1.0 (all options
            equally weighted, Eq. 4.23).
    """

    label: str
    sharp_belief_mass: float
    vague_belief_mass: float
    focal_uncertainty_mass: float
    utility: float = 1.0

    @property
    def projected_probability(self) -> float:
        """
        Mass-sum additivity (Eq. 4.9): the three components sum back up
        to the projected probability of this option.
        """
        return self.sharp_belief_mass + self.vague_belief_mass + self.focal_uncertainty_mass


def utility_normalized_probability(option: DecisionOption, max_abs_utility: float) -> float:
    """
    Utility-normalised probability, Eq. 4.15 (p. 60):

        P^N(x) = utility(x) * P(x) / max_abs_utility

    max_abs_utility (lambda+) should be the greatest absolute utility
    across ALL options being compared, not just this one, so options
    from different domains become comparable.
    """
    if max_abs_utility == 0:
        raise ValueError("max_abs_utility must be non-zero.")
    return option.utility * option.projected_probability / max_abs_utility


def utility_normalized_mass_sum(option: DecisionOption, max_abs_utility: float) -> tuple:
    """
    Utility-normalised mass-sum, Eq. 4.16-4.18 (p. 60): sharp, vague and
    focal uncertainty mass each scaled by utility / max_abs_utility.

    Note (Definition 4.10, p. 61): these normalised masses are purely
    synthetic (not real probabilities/masses) -- they exist only to make
    options comparable, and per Section 4.4 the book explicitly warns
    NOT to use the utility-normalised sharp mass as a tie-breaker in
    choose_best_option; the raw (non-normalised) sharp mass is used
    instead (see that function's docstring).
    """
    if max_abs_utility == 0:
        raise ValueError("max_abs_utility must be non-zero.")
    factor = option.utility / max_abs_utility
    return (
        option.sharp_belief_mass * factor,
        option.vague_belief_mass * factor,
        option.focal_uncertainty_mass * factor,
    )


def choose_best_option(
    options: List[DecisionOption], tolerance: float = 1e-9
) -> Optional[DecisionOption]:
    """
    Apply the decision criteria of Section 4.4 (p. 63), in priority order:

      1. Greatest utility-normalised probability (Eq. 4.15) wins.
      2. Among ties, greatest RAW sharp belief mass wins (the book
         explicitly says utility-normalised sharp mass is not meaningful
         for this comparison, p. 62: "it is not meaningful to consider
         utility-normalised sharp belief mass for choosing between
         options").
      3. Among remaining ties, least RAW focal uncertainty mass
         (equivalently, greatest vague belief mass) wins.

    Returns None if a genuine tie remains after all three criteria --
    the book calls this "a difficult decision" (Section 4.4, step (h))
    and leaves it unresolved; the caller must decide how to break it.

    Validated against the book's urn betting example (Section 4.3,
    pp. 61-62): two options with equal utility-normalised probability
    (0.2) are broken by comparing raw sharp belief mass.
    """
    if not options:
        raise ValueError("Cannot choose among an empty list of options.")

    max_abs_utility = max(abs(o.utility) for o in options)
    if max_abs_utility == 0:
        raise ValueError("At least one option must have non-zero utility.")

    # Step 1: utility-normalised probability.
    scored = [(utility_normalized_probability(o, max_abs_utility), o) for o in options]
    best_score = max(score for score, _ in scored)
    candidates = [o for score, o in scored if abs(score - best_score) <= tolerance]
    if len(candidates) == 1:
        return candidates[0]

    # Step 2: raw sharp belief mass (NOT utility-normalised, per book's warning above).
    best_sharp = max(o.sharp_belief_mass for o in candidates)
    candidates = [o for o in candidates if abs(o.sharp_belief_mass - best_sharp) <= tolerance]
    if len(candidates) == 1:
        return candidates[0]

    # Step 3: least raw focal uncertainty mass.
    least_focal = min(o.focal_uncertainty_mass for o in candidates)
    candidates = [o for o in candidates if abs(o.focal_uncertainty_mass - least_focal) <= tolerance]
    if len(candidates) == 1:
        return candidates[0]

    return None  # Genuine tie: Section 4.4, step (h), "a difficult decision".