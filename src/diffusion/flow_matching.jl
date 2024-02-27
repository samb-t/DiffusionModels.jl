abstract type AbstractCFM <: AbstractDiffusion end

# TODO: Generalise LinearCFM to any CFM.
# TODO: Make CFM abstract, and allow specific 
#       instantiations with different numbers of points.
#       To start with just any two, and anything to Gaussian.

struct LinearCFM{T<:AbstractFloat} <: AbstractCFM
    sigma::T
end

function marginal(
    d::LinearCFM, 
    x_start::AbstractArray, 
    x_end::AbstractArray, 
    t::AbstractVector,
)
    shape = tuple([1 for _ in 1:ndims(x_start)-1]..., length(t))
    t = reshape(t, shape)
    mean = t .* x_start .+ (1 .- t) .* x_end
    return MdNormal(mean, d.sigma)
end

function conditional_vector_field(
    d::LinearCFM,
    x_start::AbstractArray, 
    x_end::AbstractArray, 
    t::AbstractVector,
    x_t::AbstractArray
)
    return x_end .- x_start
end

function loss(
    d::LinearCFM,
    x_start::AbstractArray, 
    x_end::AbstractArray, 
    t::AbstractVector,
)
    eps = randn!(similar(x_start))
    x_t = marginal(d, x_start, x_end, t)
    x_t = x_t.mean .+ x_t.std .* eps
    u_t = conditional_vector_field(d, x_start, x_end, t, x_t)
    t, x_t, u_t, eps
end


@kwdef struct OTPlanSampler{F,T}
    ot_fn::F=sinkhorn # E.g. emd or sinkhorn
    reg::T=0.05
    reg_m::T=1.0
    normalize_cost::Bool=false
end

function get_map(
    o::OTPlanSampler,
    x_start::AbstractArray,
    x_end::AbstractArray,
)
    batch_size = size(x_start, ndims(x_start))
    # TODO: move a and b to same device as x_start/x_end
    a = fill(1 / batch_size, batch_size)
    b = fill(1 / batch_size, batch_size)
    x_start = reshape(x_start, (:,batch_size))
    x_end = reshape(x_end, (:,batch_size))
    C = pairwise(SqEuclidean(), x_start, x_end; dims=2)
    if o.normalize_cost
        C = C / maximum(C)
    end
    y = o.ot_fn(a, b, C, o.reg)
    return y
end

function sample_plan(
    o::OTPlanSampler,
    x_start::AbstractArray,
    x_end::AbstractArray,
)
    batch_size = size(x_start, ndims(x_start))
    map = get_map(o, x_start, x_end)
    map = reshape(map, :)
    map = map ./ sum(map)
    choices = rand(Categorical(map), batch_size)
    choices = divrem.(choices, batch_size)
    i = first.(choices) .+ 1
    j = last.(choices) .+ 1
    return gather(x_start, i), gather(x_end, j)
end


struct ExactOptimalTransportCFM{T<:AbstractFloat} <: AbstractCFM
    sigma::T
    ot_sampler::OTPlanSampler
end

function marginal(
    d::ExactOptimalTransportCFM, 
    x_start::AbstractArray, 
    x_end::AbstractArray, 
    t::AbstractVector,
)
    # shape = tuple([1 for _ in 1:ndims(x_start)-1]..., length(t))
    # t = reshape(t, shape)
    # mean = t .* x_start .+ (1 .- t) .* x_end
    # return MdNormal(mean, d.sigma)
end

function conditional_vector_field(
    d::ExactOptimalTransportCFM,
    x_start::AbstractArray, 
    x_end::AbstractArray, 
    t::AbstractVector,
)
    # return x_end .- x_start
end



function reparameterise_rand(
    d::MdNormal
)
    ...
end


# function marginal(d::CFM, x_start::AbstractArray, x_end::AbstractArray, t::AbstractVector)
#     shape = tuple([1 for _ in 1:ndims(x_start)-1]..., length(t))
#     t = reshape(t, shape)
#     coeff = sqrt.(alpha_cumulative(d.schedule, t))
#     mean = coeff .* x_start .+ (1 .- coeff) .* x_end
#     return MdNormal(mean, d.sigma)
# end
