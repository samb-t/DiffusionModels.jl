abstract type AbstractSchedule end
abstract type AbstractNoiseSchedule <: AbstractSchedule end
abstract type VPNoiseSchedule <: AbstractNoiseSchedule end
abstract type VENoiseSchedule <: AbstractNoiseSchedule end

# TODO: Just do alpha, sigma, and e.g. log SNR

function alpha_cumulative(schedule::AbstractNoiseSchedule, t::AbstractArray)
    alpha_cum(t::AbstractFloat) = alpha_cumulative(schedule, t)
    return alpha_cum.(t)
end

function beta(schedule::AbstractNoiseSchedule, t::AbstractArray)
    beta_this(t::AbstractFloat) = beta(schedule, t)
    return beta_this.(t)
end


# TODO: I'm really not sure on the squaring at the end of the sigmoid and CosineSchedule
# schedules. Have a strong feeling it's not necessary. E.g. for Cosine, there's already
# a square in there. Not sure about for sigmoid though

# Schedules from "On the Importance of Noise Scheduling for Diffusion Models"
# Which are similar to Simple Diffusion, but more hyperparams.
# TODO: Add SNR scaling from this paper and Simple Diffusion
@kwdef struct CosineSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    t_start::T=0.0
    t_end::T=1.0
    tau::T=1.0
    clip_min::T=1e-9
end

function alpha_cumulative(schedule::CosineSchedule, t::AbstractFloat)
    v_start = cos(schedule.t_start * π / 2) ^ (2 * schedule.tau)
    v_end = cos(schedule.t_end * π / 2) ^ (2 * schedule.tau)
    output = cos((t * (schedule.t_end - schedule.t_start) + schedule.t_start) * π / 2) ^ (2 * schedule.tau)
    output = (v_end - output) / (v_end - v_start)
    output = clamp(output, schedule.clip_min, 1)
    return convert(typeof(t), output)
end

# TODO: Do this when t_start, t_end, tau, etc are all used
# To make sure there are no singularities
# Since e.g. for improved diffusion they set t_min=0.008 (and t_max=1.008??)
function beta(schedule::CosineSchedule, t::AbstractFloat)
    output = clamp(π * schedule.tau * tan(π * t / 2), 0, 999)
    return convert(typeof(t), output)
end

@kwdef struct SigmoidSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    t_start::T=-3.0
    t_end::T=3.0
    tau::T=1.0
    clip_min::T=1e-9
end

function alpha_cumulative(schedule::SigmoidSchedule, t::AbstractFloat)
    v_start = sigmoid(schedule.t_start / schedule.tau)
    v_end = sigmoid(schedule.t_end / schedule.tau)
    output = sigmoid((t * (schedule.t_end - schedule.t_start) + schedule.t_start) / schedule.tau)
    output = (v_end - output) / (v_end - v_start)
    output = clamp(output, schedule.clip_min, 1)
    return convert(typeof(t), output)
end

# TODO: Add tau
function beta(schedule::SigmoidSchedule, t::AbstractFloat)
    output = (schedule.t_end - schedule.t_start) - (schedule.t_end - schedule.t_start) / (1 + exp(t * (schedule.t_end - schedule.t_start) + schedule.t_start))
    return convert(typeof(t), output)
end


@kwdef struct LinearSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    beta_start::T=0.1
    beta_end::T=20
    clip_min::T=1e-9
end

function alpha_cumulative(schedule::LinearSchedule, t::AbstractFloat)
    log_mean_coeff = -0.5 * t ^ 2 * (schedule.beta_end - schedule.beta_start) - t * schedule.beta_start
    output = exp(log_mean_coeff)
    output = clamp(output, schedule.clip_min, 1)
    return convert(typeof(t), output)
end

function beta(schedule::LinearSchedule, t::AbstractFloat)
    beta_t = schedule.beta_start + t * (schedule.beta_end - schedule.beta_start)
    return convert(typeof(t), beta_t)
end


# Used in "On the Importance of Noise Scheduling for Diffusion Models"
# This is also the schedule used in Absorbing Diffusion
# Originally from "Deep unsuper-vised learning using nonequilibrium thermodynamics"
struct LinearMutualInfoSchedule <: VPNoiseSchedule
    clip_min::AbstractFloat
end

LinearMutualInfoSchedule(; clip_min=1e-9) = LinearMutualInfoSchedule(clip_min)

function alpha_cumulative(schedule::LinearMutualInfoSchedule, t::AbstractFloat)
    output = 1-t
    output = clamp(output, schedule.clip_min, 1)
    return convert(typeof(t), output)
end

function beta(schedule::LinearMutualInfoSchedule, t::AbstractFloat)
    output = 1 / (1 - t)
    return convert(typeof(t), output)
end


# TODO: Learned noise schedule from Variational Diffusion Models


## Frequency Schedules #

# TODO: Support various numbers of dims to blur. Not only images
@kwdef struct FrequencySchedule
    schedule::VPNoiseSchedule
    sigma_max_blur::AbstractFloat
    img_dim::Int
    min_scale::AbstractFloat=0.001
end

# function alpha_cumulative(s::FrequencySchedule, t::AbstractFloat)
#     # compute dissipation time
#     sigma_blur = s.sigma_max_blur * (1 - alpha_cumulative(s.schedule, t))
#     dissipation_time = sigma_blur ^ 2 / 2

#     # frequencies
#     freq = π .* collect(0:(s.img_dim - 1)) ./ s.img_dim
#     lambda = reshape(freq, (s.img_dim,1,1)).^2 + reshape(freq, (1,s.img_dim,1)).^2

# end

function alpha_cumulative(s::FrequencySchedule, t::AbstractArray)
    # compute dissipation time
    sigma_blur = s.sigma_max_blur .* (1 .- alpha_cumulative(s.schedule, t))
    dissipation_time = sigma_blur .^ 2 ./ 2

    # frequencies
    freq = π .* collect(0:(s.img_dim - 1)) ./ s.img_dim
    lambda = reshape(freq, (s.img_dim,1,1,1)).^2 .+ reshape(freq, (1,s.img_dim,1,1)).^2

    # compute scaling for frequencies
    scaling = exp.(-lambda .* dissipation_time) .* (1 .- s.min_scale)
    return scaling .+ s.min_scale
end




## Jump Schedules ##

abstract type AbstractJumpSchedule <: AbstractSchedule end

@kwdef struct ConstantJumpSchedule <: AbstractJumpSchedule
    max_dim::Int
    minimum_dims::Int=1
    std_mult::AbstractFloat=0.7
end

function rate(s::ConstantJumpSchedule, t::AbstractFloat)
    c = s.max_dim - s.minimum_dims
    (2 * c + s.std_mult^2 + sqrt((s.std_mult^2 + 2 * c)^2 - 4 * c^2)) / 2
end

function rate_integral(s::ConstantJumpSchedule, t::AbstractFloat)
    return rate(s, t) * t
end

# TODO: Rename all schedules to have same function names so these functions can be shared
function rate_integral(schedule::AbstractNoiseSchedule, t::AbstractArray)
    rate_int(t::AbstractFloat) = rate_integral(schedule, t)
    return rate_int.(t)
end

function rate(schedule::AbstractNoiseSchedule, t::AbstractArray)
    rate_this(t::AbstractFloat) = rate(schedule, t)
    return rate_this.(t)
end
