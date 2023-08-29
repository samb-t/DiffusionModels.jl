
abstract type AbstractJumpDiffusion <: AbstractDiffusion end

# TODO: replace AbstractGaussianDiffusion with AbstractSDEDiffusion
# TODO: DimensionalGaussianJumpDiffusion??
struct DimensionalJumpDiffusion{D<:AbstractGaussianDiffusion,T} <: AbstractJumpDiffusion
    diffusion::D
    jump_schedule::AbstractJumpSchedule
    dim::Int
    score_fn::T
end

function DimensionalJumpDiffusion(
    diffusion::D,
    jump_schedule::AbstractJumpSchedule,
    dim::Int;
    score_fn::T=nothing,
) where {D<:AbstractGaussianDiffusion,T}
    DimensionalJumpDiffusion{D,T}(diffusion, jump_schedule, dim, score_fn)
end

# TODO: Make fill with missing like get_jump
function marginal(
    d::DimensionalJumpDiffusion,
    x_start::AbstractArray, 
    t::AbstractFloat,
)
    # 1. remove dimensions
    integral = rate_integral(d.jump_schedule, t)
    dims_deleted = rand(Poisson(integral))
    dims_keep = max(1, size(x_start, d.dim) - dims_deleted)
    indices = rand(1:size(x_start, d.dim), dims_keep)
    x = selectdim(x_start, d.dim, indices)
    # 2. add noise
    x = marginal(d.diffusion, x, [t])
    return x
end

function get_jump(d::DimensionalJumpDiffusion)
    affect!(integrator) = begin
        x = integrator.u
        t_new = integrator.t
        t_prev = integrator.tprev
        integral = rate_integral(d.jump_schedule, t_new) - rate_integral(d.jump_schedule, t_prev)
        dims_deleted = rand(Poisson(integral), size(x, ndims(x)))
        dims_left = count(!ismissing, x[1,:,:], dims=1)
        max_dims_deleted = dims_left .- 1
        dims_deleted = min.(dims_deleted, max_dims_deleted)

        # TODO: Below assumes length dim is second from end rather than d.dim
        # TODO: This could be much more efficient with a single scatter
        #       (currently have to permute dim to end)
        for (x_i, d) in zip(eachslice(x, dims=ndims(x)), dims_deleted)
            # TODO: don't fix to 2D tensors
            indices = rand(collect(eachindex(skipmissing(view(x_i, 1, :)))), d)
            # TODO: Does missing work with cuda or is NaN necessary?
            missings = fill(missing, length(indices))
            missings = reshape(missings, 1, :)
            missings = repeat(missings, outer=(size(x_i, 1), 1))
            NNlib.scatter!((i,j)->j, x_i, missings, indices)
        end
        # TODO: Already be inplace so might not be necessary?
        integrator.u = x
    end
    # TODO: Tempting to make all schedules f(schedule, u, p, t)?
    rate_fn(u, p, t) = rate(d.jump_schedule, t)
    return ConstantRateJump(rate_fn, affect!)
end

# or get_forward. Or get_forward_process
# or remove "get" from any of these
function get_forward_diffeq(
    d::AbstractJumpDiffusion,
    x::AbstractArray,
    tspan::Tuple{AbstractFloat, AbstractFloat};
    aggregator=Direct() # AbstractAggregatorAlgorithm
)
    x = convert(AbstractArray{Union{Missing, eltype(x)}}, x)
    prob = get_forward_sde(d.diffusion, x, tspan)
    jump = get_jump(d)
    return JumpProblem(prob, aggregator, jump)
end

# TODO: This should be added to RecursiveArrayTools.jl?
#       Or change DifferentialEquations.jl to support missing
# Also TODO: This should be implemented properly
function RecursiveArrayTools.recursive_bottom_eltype(a::Type{Union{Missing, T}}) where T
    return Union{Missing, T}
end

function RecursiveArrayTools.recursive_unitless_bottom_eltype(a::Type{Union{Missing, T}}) where T
    return Union{Missing, T}
end