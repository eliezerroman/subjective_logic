"""Subjective Logic for Agentic AI: a Python implementation of Josang's
Subjective Logic formalism, applied to measuring decision degradation
in agentic AI systems under resource constraints."""

from .binomial import BinomialOpinion, NON_INFORMATIVE_PRIOR_WEIGHT
from .conflict import conjunctive_certainty, degree_of_conflict, projected_distance
from .decision import DecisionOption, choose_best_option, utility_normalized_mass_sum, utility_normalized_probability
from .domain import base_rate_of_value, composite_set, hyperdomain, relative_base_rate
from .entropy import cross_entropy, opinion_entropy, sharpness_entropy, surprisal, uncertainty_entropy, vagueness_entropy
from .hyperopinion import HyperOpinion
from .multinomial import MultinomialOpinion
from .operators import add, codivide, comultiply, complement, divide, multiply, subtract
from .multinomial_operators import averaging_proportional_divide, multiply_dirichlet, normal_multiply, proportional_multiply, selective_divide
from .deduction import binomial_deduce, free_base_rate_interval, material_implication, multinomial_deduce
from .abduction import binomial_abduce, binomial_invert, dependence, independence, irrelevance, multinomial_abduce, multinomial_invert, relevance
from .joint import joint_opinion, marginal_conditionals, marginalize
from .fusion import averaging_fusion, cumulative_fusion, weighted_fusion
from .unfusion import averaging_unfusion, cumulative_fission, cumulative_unfusion

__all__ = [
    "BinomialOpinion",
    "MultinomialOpinion",
    "HyperOpinion",
    "NON_INFORMATIVE_PRIOR_WEIGHT",
    "hyperdomain",
    "composite_set",
    "base_rate_of_value",
    "relative_base_rate",
    "DecisionOption",
    "choose_best_option",
    "utility_normalized_probability",
    "utility_normalized_mass_sum",
    "surprisal",
    "opinion_entropy",
    "sharpness_entropy",
    "vagueness_entropy",
    "uncertainty_entropy",
    "cross_entropy",
    "projected_distance",
    "conjunctive_certainty",
    "degree_of_conflict",
    "add",
    "complement",
    "subtract",
    "multiply", 
    "comultiply", 
    "divide", 
    "codivide",
    "averaging_proportional_divide", 
    "multiply_dirichlet", 
    "normal_multiply",
    "proportional_multiply", 
    "selective_divide",
    "binomial_deduce", 
    "free_base_rate_interval", 
    "material_implication", 
    "multinomial_deduce",
    "binomial_abduce", 
    "binomial_invert", 
    "dependence", 
    "independence", 
    "irrelevance",
    "multinomial_abduce", 
    "multinomial_invert", 
    "relevance",
    "joint_opinion", 
    "marginal_conditionals", 
    "marginalize",
    "averaging_fusion", 
    "cumulative_fusion", 
    "weighted_fusion",
    "averaging_unfusion", 
    "cumulative_fission", 
    "cumulative_unfusion",
]