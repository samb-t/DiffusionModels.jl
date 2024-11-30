# Notes:
# A continuous time version of this needs to have a change function
# which is a function of time, and decides when a transition should
# take place

# Have a read of "Score-Based Continuous-Time Discrete Diffusion Models"
# https://arxiv.org/abs/2211.16750

"""
    Absorbing Diffusion Models
"""
struct AbsorbingDiffusion <: AbstractDiscreteDiffusion
    ...
end

# For both discrete and continuous time, decide approximately how many
# changes have taken place by this time, and make that many changes
function marginal()
    ...
end

# Discrete and continuous time are similar. In this time period, decide
# the probability of a per-dimension change occuring, and make mask out
# Discrete time:
#   - What is the probability of masking at this step
# Continuous time:
#   - What is the probability that a token was masked in the presceeding
#     time period.
function q_transition()
    ...
end

function p_transition()
    ...
end


# TODO:
# - Support hybrid loss from d3pm
# - Support Unleashing Transformers loss weighting
# - Support MaskGIT sampling
# - Support microsoft paper multinomial+absorbing transitions




# "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution"
# https://arxiv.org/abs/2310.16834
# https://github.com/louaaron/Score-Entropy-Discrete-Diffusion

# Base on SciMLBase.DiscreteProblem(f::ODEFunction, ...)
# s.t. uₙ₊₁ = f(uₙ, p, tₙ₊₁)

struct AbsorbingScoreEntropyDiffusion <: AbstractCategoricalDiffusion
    schedule
    score_fn
    dim
end

function marginal(d::AbsorbingScoreEntropyDiffusion, x_start::AbstractArray, t::AbstractVector)
    sigma, dsigma = schedule(t)
    move_chance = 1 .- exp.(-sigma)
    move_indices = rand(size(x_start)...) .< move_chance
    # i_pert = where(move_indices, d.dim, x_start)
end

function sample_prior(d::AbsorbingScoreEntropyDiffusion, shape)
    ones(shape...) .* (d.dim - 1)
end
