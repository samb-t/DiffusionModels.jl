@doc raw"""
    AbstractSchedule

Abstract type for schedules. TODO: write docstring.
"""
abstract type AbstractSchedule end

@doc raw"""
    AbstractNoiseSchedule

Abstract type for noise schedules. TODO: write docstring.
"""
abstract type AbstractNoiseSchedule <: AbstractSchedule end

@doc raw"""
    AbstractGaussianNoiseSchedule

Abstract type for Gaussian noise schedules. TODO: write docstring.
"""
abstract type AbstractGaussianNoiseSchedule <: AbstractNoiseSchedule end

@doc raw"""
    VPNoiseSchedule

Abstract type for variance preserving (VP) noise schedules. TODO: write docstring.
"""
abstract type VPNoiseSchedule <: AbstractGaussianNoiseSchedule end

# TODO: Add VENoiseSchedule docstring
abstract type VENoiseSchedule <: AbstractGaussianNoiseSchedule end

@doc raw"""
    marginal_mean_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractFloat)
    marginal_mean_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractVector)

For a Gaussian diffusion model with marginal distribution ``x_t = \alpha_t\cdot x_0 + \sigma_t\cdot\epsilon``,
this function returns ``\alpha_t``, the mean of the marginal distribution at time `t`.

## Variance Preserving

A variance preserving (VP) forward process `VPNoiseSchedule <: AbstractGaussianNoiseSchedule` can be defined
either in terms of `marginal_mean_coeff` (``\alpha``) and `marginal_std_coeff` (``\sigma``),
or by defining `log_snr`, the log-SNR ``\lambda_t``. ``\alpha`` and ``\sigma`` then get calculated
as follows:

``\alpha^2 = \text{sigmoid}(\lambda_t)``
``\sigma^2 = \text{sigmoid}(-\lambda_t)``

where ``\lambda_t = \log(\alpha_t^2/\sigma_t^2)``.

## Variance Exploding

TODO: Similar for VE forward process.

## Example

```jldoctest
julia> s = CosineSchedule()
julia> marginal_mean_coeff(s, 0.3)
0.8910065241883679
```
"""
function marginal_mean_coeff end

@doc raw"""
    marginal_std_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractFloat)
    marginal_std_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractVector)

For a Gaussian diffusion model with marginal distribution ``x_t = \alpha_t\cdot x_0 + \sigma_t\cdot\epsilon``,
this function returns ``\sigma_t``, the standard deviation of the marginal distribution at time `t`.

## Variance Preserving

A variance preserving (VP) forward process `VPNoiseSchedule <: AbstractGaussianNoiseSchedule` can be defined
either in terms of `marginal_mean_coeff` (``\alpha``) and `marginal_std_coeff` (``\sigma``),
or by defining `log_snr`, the log-SNR ``\lambda_t``. ``\alpha`` and ``\sigma`` then get calculated

``math
    \alpha^2 = \text{sigmoid}(\lambda_t)
``

``math
    \sigma^2 = \text{sigmoid}(-\lambda_t)
``

where ``\lambda_t = \log(\alpha_t^2/\sigma_t^2)``.

## Variance Exploding

TODO: Similar for VE forward process.

## Example

```jldoctest
julia> s = CosineSchedule()
julia> marginal_std_coeff(s, 0.3)
0.4539904997395468
```
"""
function marginal_std_coeff end

@doc raw"""
    drift_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractFloat)

For a Gaussian diffusion model with SDE ``dx = f(x, t)dt + g(t)dw``,
this function returns the drift coefficient `f(x, t)` at time `t`.

For the SDE we define
    dx = f(x, t)dt + g(t) dw
These can be defined using β(t)
    β(t) = d/dt log(1 + e^{-λₜ})

For the VP case,
    f(x, t) = -0.5 β(t) x    and    g(t) = β(t)

## Example

```jldoctest
julia> s = CosineSchedule()
julia> drift_coeff(s, 0.3)
-0.8003607044743674
```
"""
function drift_coeff end

@doc raw"""
    diffusion_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractFloat)

For a Gaussian diffusion model with SDE ``dx = f(x, t)dt + g(t)dw``,
this function returns the diffusion coefficient `g(t)` at time `t`.

## Example

```jldoctest
julia> s = CosineSchedule()
julia> diffusion_coeff(s, 0.3)
1.2651961938564054
```
"""
function diffusion_coeff end

@doc raw"""
    log_snr(s::AbstractGaussianNoiseSchedule, t::AbstractFloat)

For a Gaussian diffusion model with marginal distribution ``x_t = \alpha_t\cdot x_0 + \sigma_t\cdot\epsilon``,
this function returns the log signal-to-noise ratio (SNR) at time ``t``: ``λ_t = \log(\alpha_t^2/\sigma_t^2)``.

## Example

```jldoctest
julia> s = CosineSchedule()
julia> log_snr(s, 0.3)
1.3485509552536332
```
"""
function log_snr end

@doc raw"""
    beta(s::AbstractGaussianNoiseSchedule, t::AbstractFloat)

For a Gaussian diffusion model with SDE ``dx = f(x, t)dt + g(t)dw``,
this function returns the function `β(t)` that defines the SDE.

## Example

```jldoctest
julia> s = CosineSchedule()
julia> beta(s, 0.3)
1.6007214089487347
```
"""
function beta end

# TODO: Can these be moved to next to each function + docstring?
# When I tried this it complained about the interface being defined multiple times
@required AbstractGaussianNoiseSchedule begin
    marginal_mean_coeff(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
    marginal_std_coeff(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
    drift_coeff(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
    diffusion_coeff(::AbstractGaussianNoiseSchedule, ::AbstractFloat)
end

function marginal_mean_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractVector)
    return marginal_mean_coeff.(Ref(s), t)
end

function marginal_std_coeff(s::AbstractGaussianNoiseSchedule, t::AbstractVector)
    return marginal_std_coeff.(Ref(s), t)
end

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

function marginal_mean_coeff(::VENoiseSchedule, ::T) where {T<:AbstractFloat}
    return T(1)
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
function drift_coeff(s::VPNoiseSchedule, t::T) where {T<:AbstractFloat}
    return -beta(s, t) / 2
end

function diffusion_coeff(s::VPNoiseSchedule, t::AbstractFloat)
    return sqrt(beta(s, t))
end

function drift_coeff(::VENoiseSchedule, ::T) where {T<:AbstractFloat}
    return T(0)
end

function diffusion_coeff(s::VENoiseSchedule, t::AbstractFloat)
    return sqrt(beta(s, t))
end

@doc raw"""
    CosineSchedule{AbstractFloat}()

A variance preserving (VP) noise schedule... [nichol2021improved](@cite)

## Example

```jldoctest
julia> s = CosineSchedule()
julia> marginal_mean_coeff(s, 0.3)
0.8910065241883679
```
"""
struct CosineSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    # tau::T = 1.0
    # clip_min::T = 1e-9
    # shift::T = 0.0
end

# TODO: Use beta in here instead?
function log_snr(::CosineSchedule{T}, t::T) where {T<:AbstractFloat}
    return -2 * log(tan(π * t / 2)) # + 2 * s.shift
end

# TODO: use shift in here
function beta(::CosineSchedule{T}, t::T) where {T<:AbstractFloat}
    return π * tan(π * t / 2)
end

@doc raw"""
    LinearSchedule{AbstractFloat}(beta_start::AbstractFloat=0.1, beta_end::AbstractFloat=20.0, clip_min::AbstractFloat=1e-9)

A variance preserving (VP) noise schedule... [ho2020denoising](@cite)

## Example

```jldoctest
julia> s = LinearSchedule()
julia> marginal_mean_coeff(s, 0.3)
0.6295500003364489
```
"""
@kwdef struct LinearSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    beta_start::T = 0.1
    beta_end::T = 20.0
    clip_min::T = 1e-9
end

# TODO: Can this be simplified?
function log_snr(s::LinearSchedule{T}, t::T) where {T<:AbstractFloat}
    alpha_bar = exp(-0.5 * t^2 * (s.beta_end - s.beta_start) - t * s.beta_start)
    snr = alpha_bar / (1 - alpha_bar)
    return log(snr)
end

function beta(s::LinearSchedule{T}, t::T) where {T<:AbstractFloat}
    return s.beta_start + t * (s.beta_end - s.beta_start)
end

# Used in "On the Importance of Noise Scheduling for Diffusion Models"
# This is also the schedule used in Absorbing Diffusion
# Originally from "Deep unsuper-vised learning using nonequilibrium thermodynamics"

@doc raw"""
    LinearMutualInfoSchedule{AbstractFloat}()

A variance preserving (VP) noise schedule... [chen2023importance](@cite)

## Example

```jldoctest
julia> s = LinearMutualInfoSchedule()
julia> marginal_mean_coeff(s, 0.3)
0.8366600265340756
```
"""
struct LinearMutualInfoSchedule{T<:AbstractFloat} <: VPNoiseSchedule end

function log_snr(::LinearMutualInfoSchedule{T}, t::T) where {T<:AbstractFloat}
    return log((1 - t) / t)
end

function beta(::LinearMutualInfoSchedule{T}, t::T) where {T<:AbstractFloat}
    return 1 / (1 - t)
end

@kwdef struct SigmoidSchedule{T<:AbstractFloat} <: VPNoiseSchedule
    t_start::T = -3.0
    t_end::T = 3.0
    tau::T = 1.0
end

# function log_snr(s::SigmoidSchedule, t::AbstractFloat)
# end

# TODO: Add tau
function beta(schedule::SigmoidSchedule{T}, t::T) where {T<:AbstractFloat}
    output =
        (schedule.t_end - schedule.t_start) -
        (schedule.t_end - schedule.t_start) /
        (1 + exp(t * (schedule.t_end - schedule.t_start) + schedule.t_start))
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
    minimum_dims::Int = 1
    std_mult::AbstractFloat = 0.7
end

function rate(s::ConstantJumpSchedule, t::AbstractFloat)
    c = s.max_dim - s.minimum_dims
    return (2 * c + s.std_mult^2 + sqrt((s.std_mult^2 + 2 * c)^2 - 4 * c^2)) / 2
end

function rate_integral(s::ConstantJumpSchedule, t::AbstractFloat)
    return rate(s, t) * t
end

function rate_integral(schedule::AbstractNoiseSchedule, t::AbstractArray)
    rate_int(t::AbstractFloat) = rate_integral(schedule, t)
    return rate_int.(t)
end

function rate(schedule::AbstractNoiseSchedule, t::AbstractArray)
    rate_this(t::AbstractFloat) = rate(schedule, t)
    return rate_t
end
