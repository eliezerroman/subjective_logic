"""
Multinomial multiplication and division.

Implements Sections 8.1.2, 8.1.4, 8.1.7, 8.3.2 and 8.3.3 (Josang, 2016,
pp. 115-132): generalisations of binomial multiplication/division
(Chapter 7) to domains of arbitrary cardinality, via the Cartesian
product of two factor domains.

Scope note: 8.1.5 (Projected Multiplication) and 8.1.6 (Hypernomial
Product) are intentionally NOT implemented. They require constructing
the hyperdomain of the Cartesian product domain (pairs of hypervalues,
not just singletons), a non-trivial extension of the existing domain
machinery, and the book itself deprioritises this method (p. 121: "the
normal or the proportional multinomial multiplication operator...are
therefore the preferred methods to be used").

WARNING on division (Sections 8.3.2-8.3.3): unlike binomial division
(Chapter 7), multinomial division has NO general analytical solution
(p. 129): "there is not even a correct solution to approximate." The
methods below are synthetic heuristics, not approximations of a true
inverse. Use them only when you understand this limitation.
"""

from __future__ import annotations

from typing import Mapping

from .binomial import NON_INFORMATIVE_PRIOR_WEIGHT
from .multinomial import MultinomialOpinion

_TOLERANCE = 1e-9


def _product_domain(domain_x, domain_y):
    return [(x, y) for x in domain_x for y in domain_y]


def _theoretical_max_uncertainty(projected: Mapping, base_rates: Mapping) -> float:
    """Eq. 3.27, computed directly from (projected probability, base rate) pairs, without needing a full opinion instance."""
    ratios = {x: p / base_rates[x] for x, p in projected.items() if base_rates[x] > 0}
    if not ratios:
        raise ValueError("Cannot compute theoretical max uncertainty: all base rates are zero.")
    return min(ratios.values())


def normal_multiply(opinion_x: MultinomialOpinion, opinion_y: MultinomialOpinion) -> MultinomialOpinion:
    """
    Normal multinomial multiplication, Section 8.1.2 (p. 118-120):
    given INDEPENDENT opinions on separate domains X and Y, computes the
    opinion on the Cartesian product domain X x Y using the two-step
    procedure of Eq. 8.17-8.19 -- the most conservative (uncertainty-
    preserving) method, and the one the book recommends for general use
    (p. 127).

    Preserves exact multiplication of projected probabilities (Eq. 8.8):
    P_XY(x,y) = P_X(x) * P_Y(y).

    WARNING -- independence assumption, same caveat as binomial
    multiplication (Chapter 7): do not use on opinions from correlated
    sources.

    Validated exactly against the book's egg gender/mutation example
    (Table 8.2-8.3, pp. 125-127).
    """
    domain_x, domain_y = opinion_x.domain, opinion_y.domain
    px, py = opinion_x.projected_probabilities, opinion_y.projected_probabilities
    bx, by = opinion_x.belief_masses, opinion_y.belief_masses
    ax, ay = opinion_x.base_rates, opinion_y.base_rates

    product_domain = _product_domain(domain_x, domain_y)
    projected_xy = {(x, y): px[x] * py[y] for x, y in product_domain}
    singleton_xy = {(x, y): bx[x] * by[y] for x, y in product_domain}
    base_rates_xy = {(x, y): ax[x] * ay[y] for x, y in product_domain}

    candidates = [
        (projected_xy[key] - singleton_xy[key]) / base_rates_xy[key]
        for key in product_domain
        if base_rates_xy[key] > 0
    ]
    if not candidates:
        raise ValueError("Cannot compute normal multinomial product: all product base rates are zero.")

    uncertainty_xy = max(min(candidates), 0.0)  # clamp tiny negative float noise
    belief_xy = {key: projected_xy[key] - base_rates_xy[key] * uncertainty_xy for key in product_domain}

    return MultinomialOpinion(belief_masses=belief_xy, uncertainty=uncertainty_xy, base_rates=base_rates_xy)


def proportional_multiply(opinion_x: MultinomialOpinion, opinion_y: MultinomialOpinion) -> MultinomialOpinion:
    """
    Proportional multinomial multiplication, Section 8.1.4 (p. 120-121):
    computes the product uncertainty as the proportional average of the
    factors' uncertainty relative to their theoretical maxima (Eq. 8.22-
    8.23), producing "slightly less uncertainty" than the normal product
    (p. 121) -- a faster, slightly more aggressive alternative.

    Same exact-P guarantee and independence caveat as normal_multiply.
    Validated exactly against Table 8.3's "Proportional product" row.
    """
    px, py = opinion_x.projected_probabilities, opinion_y.projected_probabilities
    ax, ay = opinion_x.base_rates, opinion_y.base_rates

    product_domain = _product_domain(opinion_x.domain, opinion_y.domain)
    projected_xy = {(x, y): px[x] * py[y] for x, y in product_domain}
    base_rates_xy = {(x, y): ax[x] * ay[y] for x, y in product_domain}

    u_max_x = _theoretical_max_uncertainty(px, ax)
    u_max_y = _theoretical_max_uncertainty(py, ay)
    u_max_xy = _theoretical_max_uncertainty(projected_xy, base_rates_xy)

    denominator = u_max_x + u_max_y
    uncertainty_xy = 0.0 if denominator == 0 else u_max_xy * (opinion_x.uncertainty + opinion_y.uncertainty) / denominator

    belief_xy = {key: projected_xy[key] - base_rates_xy[key] * uncertainty_xy for key in product_domain}

    return MultinomialOpinion(belief_masses=belief_xy, uncertainty=uncertainty_xy, base_rates=base_rates_xy)


def multiply_dirichlet(
    evidence_x: Mapping,
    base_rates_x: Mapping,
    evidence_y: Mapping,
    base_rates_y: Mapping,
    prior_weight: float = NON_INFORMATIVE_PRIOR_WEIGHT,
):
    """
    Product of two evidence-Dirichlet PDFs, Section 8.1.7 (p. 123-124):
    a convenience wrapper around the book's 4-step procedure (Figure
    8.2), composed from pieces already implemented elsewhere: build
    opinions from evidence, multiply (normal method), convert back to
    evidence.

    Returns (evidence_xy, base_rates_xy) for the product Dirichlet PDF.
    """
    opinion_x = MultinomialOpinion.from_evidence(evidence_x, base_rates_x, prior_weight)
    opinion_y = MultinomialOpinion.from_evidence(evidence_y, base_rates_y, prior_weight)
    opinion_xy = normal_multiply(opinion_x, opinion_y)
    return opinion_xy.to_evidence(prior_weight), dict(opinion_xy.base_rates)


def averaging_proportional_divide(opinion_xy: MultinomialOpinion, opinion_y: MultinomialOpinion) -> MultinomialOpinion:
    """
    Averaging proportional division, Section 8.3.2 (p. 129-131): given
    the opinion on a product domain X x Y and the opinion on factor Y,
    computes a SYNTHETIC opinion on X, since -- per the book -- no exact
    inverse of multiplication generally exists (p. 129).

    Method: averages the per-y candidate quotient probabilities
    P_XY(x,y)/P_Y(y) (Eq. 8.43-8.48), then derives uncertainty as a
    proportional average of the argument uncertainties (Eq. 8.50-8.54).

    Special case validated exactly: when opinion_xy IS the normal or
    proportional product of some opinion_x and opinion_y, division
    recovers opinion_x's projected probabilities exactly (because then
    P_XY(x,y)/P_Y(y) = P_X(x) for every y, so averaging identical values
    changes nothing) -- see tests.

    Raises:
        ValueError: if opinion_xy's domain is not the full Cartesian
        product of an X-domain and opinion_y's domain, if Eq. 8.44's
        condition is violated (division by a zero-probability outcome
        that itself has positive joint probability), or if the base
        rates are inconsistent with the "realistic" factorisation
        assumption of Eq. 8.42.
    """
    domain_y = list(opinion_y.domain)
    domain_x, seen = [], set()
    for x, y in opinion_xy.domain:
        if y not in domain_y:
            raise ValueError(f"Product opinion contains Y-value {y!r} not in the divisor's domain.")
        if x not in seen:
            seen.add(x)
            domain_x.append(x)

    if set(opinion_xy.domain) != set(_product_domain(domain_x, domain_y)):
        raise ValueError("Product opinion's domain is not the full Cartesian product of its X and Y components.")

    p_xy, p_y = opinion_xy.projected_probabilities, opinion_y.projected_probabilities

    def infer_ax(x):
        for y in domain_y:
            if opinion_y.base_rates[y] > 0:
                return opinion_xy.base_rates[(x, y)] / opinion_y.base_rates[y]
        raise ValueError("Cannot infer base rate a_X: all base rates of opinion_y are zero.")

    a_x = {x: infer_ax(x) for x in domain_x}
    total_a = sum(a_x.values())
    if total_a == 0:
        raise ValueError("Cannot infer a valid base rate distribution a_X: all inferred values are zero.")
    a_x = {x: v / total_a for x, v in a_x.items()}

    l = len(domain_y)
    p_pre_x = {}
    for x in domain_x:
        total = 0.0
        for y in domain_y:
            pxy, py_val = p_xy[(x, y)], p_y[y]
            if pxy == 0 and py_val == 0:
                term = 0.0  # Eq. 8.43 limit rule
            elif pxy > 0 and py_val == 0:
                raise ValueError(f"No division is possible: P({(x, y)!r}) > 0 but P({y!r}) = 0 (Eq. 8.44).")
            else:
                term = pxy / py_val
            total += term
        p_pre_x[x] = total / l

    normalization = sum(p_pre_x.values())
    if normalization == 0:
        raise ValueError("Cannot normalise: preliminary quotient probabilities are all zero.")
    p_avg_x = {x: p / normalization for x, p in p_pre_x.items()}

    u_max_x = _theoretical_max_uncertainty(p_avg_x, a_x)
    u_max_y = _theoretical_max_uncertainty(p_y, opinion_y.base_rates)
    u_max_xy = _theoretical_max_uncertainty(p_xy, opinion_xy.base_rates)

    if opinion_xy.uncertainty == 0:
        ratio = 0.0  # Eq. 8.53 convention
    elif u_max_xy == 0:
        raise ValueError("Theoretical max uncertainty of the product opinion is zero, but its actual uncertainty is not.")
    else:
        ratio = opinion_xy.uncertainty / u_max_xy

    u_pre_x = ratio * (u_max_x + u_max_y) - opinion_y.uncertainty
    uncertainty_x = min(max(u_pre_x, 0.0), 1.0)  # Eq. 8.54 range constraint

    belief_x = {}
    for x in domain_x:
        value = p_avg_x[x] - a_x[x] * uncertainty_x
        if value < -_TOLERANCE:
            raise ValueError(
                f"Averaging proportional division produced a negative belief mass for {x!r} (b={value!r}). "
                "The input opinions may be inconsistent with the realistic base-rate assumption (Eq. 8.42)."
            )
        belief_x[x] = max(value, 0.0)

    return MultinomialOpinion(belief_masses=belief_x, uncertainty=uncertainty_x, base_rates=a_x)


def selective_divide(opinion_xy: MultinomialOpinion, opinion_y: MultinomialOpinion, observed_value) -> MultinomialOpinion:
    """
    Selective division, Section 8.3.3 (p. 131-132): assumes a specific
    value y_j of Y has been OBSERVED (opinion_y must be the absolute
    opinion b_Y(y_j)=1), and extracts the corresponding "slice" of the
    joint opinion as an opinion on X (Eq. 8.57-8.61).

    IMPORTANT: for the result to be a valid (additive) opinion, the
    joint opinion_xy must itself be consistent with y_j being certain
    -- i.e. all of its probability mass on the y_j "column" (this is
    the case, for example, when opinion_xy was built from evidence that
    only ever observed y_j). This function validates that condition and
    raises a clear error otherwise, rather than silently returning an
    opinion whose probabilities don't sum to 1.

    Raises:
        ValueError: if opinion_y is not an absolute opinion on
        observed_value, or if opinion_xy's implied marginal for
        observed_value isn't (numerically) 1.
    """
    if opinion_y.belief_masses.get(observed_value, 0.0) != 1.0:
        raise ValueError(
            f"opinion_y must be an absolute opinion asserting {observed_value!r} is TRUE "
            "(belief mass 1.0), per Eq. 8.57."
        )

    domain_y = list(opinion_y.domain)
    domain_x, seen = [], set()
    for x, y in opinion_xy.domain:
        if y not in domain_y:
            raise ValueError(f"Product opinion contains Y-value {y!r} not in the divisor's domain.")
        if x not in seen:
            seen.add(x)
            domain_x.append(x)

    p_xy = opinion_xy.projected_probabilities
    p_x = {x: p_xy[(x, observed_value)] for x in domain_x}  # Eq. 8.58, since P_Y(y_j) = 1

    total = sum(p_x.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Selective division is only valid when the joint opinion's marginal for {observed_value!r} "
            f"is 1 (got {total!r}). The joint opinion is not consistent with {observed_value!r} being "
            "certain -- see this function's docstring."
        )

    def infer_ax(x):
        if opinion_y.base_rates[observed_value] == 0:
            raise ValueError("Cannot infer base rate a_X: divisor's base rate for observed_value is zero.")
        return opinion_xy.base_rates[(x, observed_value)] / opinion_y.base_rates[observed_value]

    a_x = {x: infer_ax(x) for x in domain_x}
    total_a = sum(a_x.values())
    a_x = {x: v / total_a for x, v in a_x.items()}

    if opinion_xy.uncertainty == 0:
        uncertainty_x = 0.0
    else:
        u_max_x = _theoretical_max_uncertainty(p_x, a_x)
        u_max_xy = _theoretical_max_uncertainty(p_xy, opinion_xy.base_rates)
        uncertainty_x = 0.0 if u_max_xy == 0 else opinion_xy.uncertainty * u_max_x / u_max_xy  # Eq. 8.59

    belief_x = {x: p_x[x] - a_x[x] * uncertainty_x for x in domain_x}

    return MultinomialOpinion(belief_masses=belief_x, uncertainty=uncertainty_x, base_rates=a_x)