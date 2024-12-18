# Notes:
# A continuous time version of this needs to have a change function
# which is a function of time, and decides when a transition should
# take place

# Have a read of "Score-Based Continuous-Time Discrete Diffusion Models"
# https://arxiv.org/abs/2211.16750


# NOTE: Do we really want to couple the jumps with steps?
# So every element in the batch jumps at the same time?

"""
    Absorbing Diffusion Models
"""
struct AbsorbingDiffusion{S,D,M} <: AbstractCategoricalDiffusion where {D<:NTuple{N,Int},M<:Integer} where N
    schedule::S
    dims::D
    mask_token::M
end

function marginal(
    d::AbsorbingDiffusion{S, D, M},
    x_start::AbstractArray{M},
    t::AbstractVector,
) where {S, D, M}
    a = marginal_mean_coeff(d.schedule, t)
    # TODO: The [p, 1-p] might need to be the other way around
    to_dist = (value, p) -> DiscreteNonParametric([d.mask_token, value], [p, 1-p])
    # Replicate `a` to match the dimensions of `x_start`
    a_expanded = reshape(a, ntuple(i -> 1, ndims(x_start)-1)..., size(a, 1))
    # Apply the distribution function to the entire array using broadcasting
    dist = to_dist.(x_start, a_expanded)
    return dist
end

struct Exact end

struct Approximate
    num_jumps::Int
end

function affect!(
    integrator,
    d::AbsorbingDiffusion{S,D,M},
) where {S,D,M}
    x = integrator.u
    x_shape = size(x)
    other_dims = setdiff(1:ndims(x), d.dims)
    # move the dimensions to the front then reshape so that the dimensions are flat
    x = permutedims(x, (d.dims..., other_dims...))
    x = reshape(x, prod(x_shape[d.dims]), other_dims...)

    # 1. sample the index to absorb
    indices = rand(1:size(x, 1), other_dims)
    # 2. scatter in the mask token at locations specified by indices. Hack for now
    for k in CartesianIndices(indices)
        index = indices[k]
        x[index, k] .= d.mask_token
    end
end

# Holy trait for ScheduleVariabilityTrait
function get_jump(
    d::AbsorbingDiffusion{S,D,M},
    ::Exact,
) where {S,D,M}
    return get_jump(
        d, Exact(), ScheduleVariabilityTrait(S)
    )
end

function get_jump(
    d::AbsorbingDiffusion{S,D,M},
    ::Exact,
    ::ConstantRateSchedule,
) where {S,D,M}
    affect_fn!(integrator) = affect!(integrator, d)
    # NOTE: jump_rate should probably be `beta`
    rate_fn(u, p, t) = jump_rate(d.jump_schedule, t)
    return ConstantJump(rate_fn, affect_fn!)
end

function get_jump(
    d::AbsorbingDiffusion{S,D,M},
    ::Exact,
    ::VariableRateSchedule,
) where {S,D,M}
    affect_fn!(integrator) = affect!(integrator, d)
    # NOTE: jump_rate should probably be `beta`
    rate_fn(u, p, t) = jump_rate(d.jump_schedule, t)
    return VariableJump(rate_fn, affect_fn!)
end

function get_jump(
    d::AbsorbingDiffusion{S,D,M},
    approximation::Approximate,
) where {S,D,M}
    throw(NotImplementedError())
    c(du, u, p, t, counts, mark) = begin
        # calculates the update given `counts` number of jumps for each
        # jump process in the interval
        # TODO: affect! used above is a special case of this really,
        # where counts=1

    end
    rate_fn(out, u, p, t) = (out .= jump_rate(d.jump_schedule, t))
    return RegularJump(rate_fn, c, approximation.num_jumps)
end

function get_forward_diffeq(
    d::AbstractCategoricalDiffusion,
    x::AbstractArray{Integer},
    tspan::Tuple{AbstractFloat, AbstractFloat};
    aggregator=Direct(),
    approach=Exact(),
)
    prob = DiscreteProblem(x, tspan, p=nothing)
    jump = get_jump(d, approach)
    return JumpProblem(prob, aggregator, jump)
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
