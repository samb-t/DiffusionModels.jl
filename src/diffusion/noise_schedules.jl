abstract type AbstractSchedule end
abstract type AbstractNoiseSchedule <: AbstractSchedule end
abstract type AbstractGaussianNoiseSchedule <: AbstractNoiseSchedule end
abstract type VPNoiseSchedule <: AbstractGaussianNoiseSchedule end
abstract type VENoiseSchedule <: AbstractGaussianNoiseSchedule end

# Define interface for gaussian noise schedules
function marginal_mean_coeff end
function marginal_std_coeff end
function drift_coeff end
function diffusion_coeff end
function log_snr end
function beta end

@required AbstractGaussianNoiseSchedule begin
    marginal_mean_coeff(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
    marginal_std_coeff(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
    drift_coeff(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
    diffusion_coeff(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
    # log_snr(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
    # beta(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
end

function marginal_mean_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractVector)
    return marginal_mean_coeff.(Ref(s), t)
end

function marginal_std_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractVector)
    return marginal_std_coeff.(Ref(s), t)
end


"""
    Define in terms of marginal distribution
        x_t = αₜ⋅x₀ + σₜ⋅ϵ

    log-SNR at time t is
        λₜ = log(αₜ²/σₜ²)

    A variance preserving (VP) forward process can be defined either
    in terms of α and σ, or from the log-SNR as
        αₜ² = sigmoid(λₜ)   and   σₜ² = sigmoid(-λₜ)
    By default we define the log-SNR, but you can add a new noise schedule
    by defining α and σ instead.

    For the SDE we define
        dx = f(x, t)dt + g(t) dw
    These can be defined using β(t)
        β(t) = d/dt log(1 + e^{-λₜ})

    For the VP case,
        f(x, t) = -0.5 β(t) x    and    g(t) = β(t)
"""


# TODO: If these funcs end up being used for non-gaussian schedules,
# give them some more general names

# TODO: Add necessary clipping to all coefficients

# TODO: Square root in the SDE? APplies to both VP and VE
function marginal_mean_coeff(s::VPNoiseSchedule, t::AbstractFloat)
    λₜ = log_snr(s::VPNoiseSchedule, t)
    return sqrt(sigmoid(λₜ))
end

function marginal_std_coeff(s::VPNoiseSchedule, t::AbstractFloat)
    λₜ = log_snr(s::VPNoiseSchedule, t)
    return sqrt(sigmoid(-λₜ))
end

function marginal_mean_coeff(::VENoiseSchedule, t::AbstractFloat)
    return 1
end

function marginal_std_coeff(s::VENoiseSchedule, t::AbstractFloat)
    λₜ = log_snr(s::VPNoiseSchedule, t)
    return sqrt(exp(-λₜ))
end

# TODO: Add a default beta function that applies to both VE and VP
# and calculates β(t) = d/dt log(1 + e^{-λₜ}) with automatic symbolic
# differentiation


# TODO: Think of better names than drift/diffusion so it makes more sense
# to apply to the discrete state space too.
function drift_coeff(s::VPNoiseSchedule, t::AbstractFloat)
    return -0.5 * beta(s, t)
end

function diffusion_coeff(s::VPNoiseSchedule, t::AbstractFloat)
    return sqrt(beta(s, t))
end

function drift_coeff(s::VENoiseSchedule, t::AbstractFloat)
    return 0
end

function diffusion_coeff(s::VENoiseSchedule, t::AbstractFloat)
    sqrt(beta(s, t))
end


@kwdef struct CosineSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    t_start::T=0.0
    t_end::T=1.0
    tau::T=1.0
    clip_min::T=1e-9
    # shift::T=0.0
end

# TODO: Use beta in here instead
function log_snr(s::CosineSchedule, t::AbstractFloat)
    return -2 * log(tan(π * t / 2)) # + 2 * s.shift
end

# TODO: use shift in here
function beta(s::CosineSchedule, t::AbstractFloat)
    return π * s.tau * tan(π * t / 2)
end


@kwdef struct LinearSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    beta_start::T=0.1
    beta_end::T=20.0
    clip_min::T=1e-9
end

# TODO: Can this be simplified?
function log_snr(s::LinearSchedule, t::AbstractFloat)
    alpha_bar = exp(-0.5 * t ^ 2 * (s.beta_end - s.beta_start) - t * s.beta_start)
    snr = alpha_bar / (1 - alpha_bar)
    return log(snr)
end

function beta(s::LinearSchedule, t::AbstractFloat)
    return s.beta_start + t * (s.beta_end - s.beta_start)
end


# Used in "On the Importance of Noise Scheduling for Diffusion Models"
# This is also the schedule used in Absorbing Diffusion
# Originally from "Deep unsuper-vised learning using nonequilibrium thermodynamics"
struct LinearMutualInfoSchedule <: VPNoiseSchedule end

function log_snr(::LinearMutualInfoSchedule, t::AbstractFloat)
    return log((1 - t) / t)
end

function beta(::LinearMutualInfoSchedule, t::AbstractFloat)
    return 1 / (1 - t)
end


@kwdef struct SigmoidSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    t_start::T=-3.0
    t_end::T=3.0
    tau::T=1.0
end

function log_snr(s::SigmoidSchedule, t::AbstractFloat)
end

# TODO: Add tau
function beta(schedule::SigmoidSchedule, t::AbstractFloat)
    output = (schedule.t_end - schedule.t_start) - (schedule.t_end - schedule.t_start) / (1 + exp(t * (schedule.t_end - schedule.t_start) + schedule.t_start))
    return convert(typeof(t), output)
end


# TODO: Add EDM Schedule from Karras et al. 2022
# NOTE: This schedule is different at train and sampling time
# @kwdef struct EDMTrainSchedule{T<:AbstractFloat} <: VENoiseSchedule

# end





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
    return rate_t
end
