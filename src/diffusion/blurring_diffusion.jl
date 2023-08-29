# TODO: Make this a special case of soft diffusion?
"""
    Blurring Diffusion Models from [1]:
        q(xₜ | x₀) = N(xₜ | Aₜ⋅x₀, σₜ²⋅I)
        
    [1] Blurring Diffusion Models.
        Hoogeboom et al. 2022
"""
# TODO: No reason why both VP and VE schedules can't be supported. Would be
#       great if there was a nice way to merge them into one SDE type.
#       - Possibly by defining sigma and alpha functions in schedule.
#       - VE is defined using sigma. VP is defined using alpha
#       - The other is determined using a shared function over similar schedules.
struct BlurringDiffusion{N,T} <: AbstractDiffusion
    noise_schedule::VPNoiseSchedule
    frequency_schedule::FrequencySchedule
    dims::NTuple{N,Int}
    score_fn::T
end

function BlurringDiffusion(
    noise_schedule::VPNoiseSchedule, 
    frequency_schedule::FrequencySchedule;
    dims::NTuple{N,Int}=(1,2),
    score_fn::T=nothing,
) where {N,T}
    BlurringDiffusion{N,T}(noise_schedule, frequency_schedule, dims, score_fn)
end

# function get_drift_diffusion(d::BlurringDiffusion)

# end

function marginal(d::BlurringDiffusion, x_start::AbstractArray, t::AbstractVector)
    # TODO: reshape t to same shape as x
    x_freq = dct(x_start, d.dims)
    
    t = reshape(t, (1, 1, 1, length(t)))
    freq_scale = alpha_cumulative(d.frequency_schedule, t)
    mean = sqrt.(alpha_cumulative(d.noise_schedule, t))
    std = sqrt.(1 .- alpha_cumulative(d.noise_schedule, t))
    z = randn!(similar(x_start))

    return idct(mean .* freq_scale .* x_freq, d.dims) .+ std .* z
end

function sample_prior(d::BlurringDiffusion,  dims::Tuple{Int}; kwargs...)
    randn(dims, kwargs...)
end






# @kwdef struct VPDiffusion{T} <: AbstractGaussianDiffusion
#     schedule::VPNoiseSchedule
#     score_fn::T=nothing
# end

# function get_drift_diffusion(d::VPDiffusion)
#     drift(x,p,t) = -0.5 * beta(d.schedule, t) .* x
#     diffusion(x,p,t) = sqrt(beta(d.schedule, t))
#     return drift, diffusion
# end

# function marginal(d::VPDiffusion, x_start::AbstractArray, t::AbstractVector)
#     t = reshape(t, (1, 1, 1, length(t)))
#     mean = sqrt.(alpha_cumulative(d.schedule, t)) .* x_start
#     std = sqrt.(1 .- alpha_cumulative(d.schedule, t))
#     std = repeat(std, outer=(size(mean)[1:end-1]..., 1))
#     return rand(MdNormal(mean, std))
# end

# function sample_prior(d::VPDiffusion,  dims::Tuple{Int}; kwargs...)
#     randn(dims, kwargs...)
# end