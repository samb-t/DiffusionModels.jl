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
struct AbsorbingDiffusion{S,D,M} <: AbstractDiscreteDiffusion where {D<:NTuple{N,Int},M<:Integer} where N
    schedule::S
    dims::D
    mask_token::M
end

# For both discrete and continuous time, decide approximately how many
# changes have taken place by this time, and make that many changes
function marginal()
end


struct Exact end

struct Approximate
    num_jumps::Int
end

function affect!(d::AbsorbingDiffusion, integrator)
    x = integrator.u
    x_shape = size(x)
    other_dims = setdiff(1:ndims(x), d.dims)
    # move the dimensions to the front then reshape so that the dimensions are flat
    x = permutedims(x, (d.dims..., other_dims...))
    x = reshape(x, prod(x_shape[d.dims]), other_dims...)

    # 1. sample the index to absorb
    indices = rand(1:size(x, 1), other_dims)
    # 2. scatter in the mask token in the indices locations. Hack for now
    for k in CartesianIndices(indices)
        index = indices[k]
        x[index, k] .= d.mask_token
    end
end

function get_jump(
    d::AbsorbingDiffusion{S},
    ::Exact,
) where S <: AbstractConstantRateSchedule
    affect_fn!(integrator) = affect!(d, integrator)
    rate_fn(u, p, t) = jump_rate(d.jump_schedule, t)
    return ConstantJump(rate_fn, affect_fn!)
end

function get_jump(
    d::AbsorbingDiffusion{S},
    ::Exact,
) where S <: AbstractVariableRateSchedule
    affect_fn!(integrator) = affect!(d, integrator)
    rate_fn(u, p, t) = jump_rate(d.jump_schedule, t)
    return VariableJump(rate_fn, affect_fn!)
end

function get_jump(
    d::AbsorbingDiffusion,
    approximation::Approximate,
)
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
    d::AbstractDiscreteDiffusion,
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
