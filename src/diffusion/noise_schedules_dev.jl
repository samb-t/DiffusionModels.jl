abstract type AbstractSchedule end
abstract type AbstractNoiseSchedule <: AbstractSchedule end
abstract type VPNoiseSchedule <: AbstractNoiseSchedule end
abstract type VENoiseSchedule <: AbstractNoiseSchedule end

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
    beta_end::T=20
    clip_min::T=1e-9
end

# TODO: Surely this is wrong and needs to use beta_start and beta_end ...
# function log_snr(s::LinearSchedule, t::AbstractFloat)
#     return -log(exp(t ^ 2 - 1))
# end

function beta(s::LinearSchedule, t::AbstractFloat)
    return schedule.beta_start + t * (schedule.beta_end - schedule.beta_start)
end


# TODO: Add EDM Schedule from Karras et al. 2022
# NOTE: This schedule is different at train and sampling time
# @kwdef struct EDMTrainSchedule{T<:AbstractFloat} <: VENoiseSchedule

# end
