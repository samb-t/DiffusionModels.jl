# TODO:
# 1. Convert to AbstractSDEDiffusion
# 2. Allow converting to ODEDiffusion
# 3. Allow converting to a discrete diffusion?
#       - as SciMLBase.DiscreteProblem
#       - DiscreteProblem(f::ODEFunction, u0, tspan) is already allowed

"""
    Variance Preserving diffusion process from [1] defined in SDE form as
                    dxₜ = -0.5β(t)x dt + √(β(t)) dw               (Eqn. 11 in [2])
    The variance is "preseved" throughout since xₜ = √ᾱ(t)⋅x₀ + √(1-ᾱ(t))⋅z.

    For ᾱ(t) = ∫₀ᵗ (1-β(t̄))dt̄, if ᾱ(0)=1 and ᾱ(1)=0 then the process is
    defined in finite time and becomes an instance of the stochastic
    interpolants framework.

    Args:
        `schedule::VPNoiseSchedule`: A variance preserving noise schedule.
        `score_fn` (optional): The score function ∇log p(xₜ).

    [1] Denoising Diffusion Probabilistic Models.
        Ho et al. NeurIPS 2020.
    [2] Score-Based Generative Modeling Through Stochastic Differential Equations.
        Song et al. ICLR 2021.
"""
struct VPDiffusion{S<:VPNoiseSchedule,T} <: AbstractGaussianDiffusion
    schedule::S
    score_fn::T
end

function VPDiffusion(schedule::VPNoiseSchedule; score_fn::T=nothing) where T
    VPDiffusion{T}(schedule, score_fn)
end

function get_drift_diffusion(d::VPDiffusion)
    drift(x,p,t) = -0.5 * beta(d.schedule, t) .* x
    diffusion(x,p,t) = sqrt(beta(d.schedule, t))
    return drift, diffusion
end

function marginal(d::VPDiffusion, x_start::AbstractArray, t::AbstractVector)
    shape = tuple([1 for _ in 1:ndims(x_start)-1]..., length(t))
    t = reshape(t, shape)
    mean = sqrt.(alpha_cumulative(d.schedule, t)) .* x_start
    std = sqrt.(1 .- alpha_cumulative(d.schedule, t))
    std = repeat(std, outer=(size(mean)[1:end-1]..., 1))
    return rand(MdNormal(mean, std))
end

function sample_prior(d::VPDiffusion,  dims::Tuple{Int}; kwargs...)
    randn(dims, kwargs...)
end

# TODO: add score_fn function which wraps the function, e.g divide by std etc?


"""
    Variance Exploding diffusion process from [1] defined in SDE form as
                            dxₜ = √(∂ₜσ²(t)) dw                    (Eqn. 9 in [2])
    where the variance "explodes", σ²(t)→∞ as t→∞

    Args:
        `schedule::VENoiseSchedule`: A variance exploding noise schedule.

    [1] Generative Modeling by Estimating Gradients of the Data Distribution.
        Song and Ermon 2019.
    [2] Score-Based Generative Modeling Through Stochastic Differential Equations.
        Song et al. ICLR 2021.
"""
struct VEDiffusion{S<:VENoiseSchedule,T} <: AbstractGaussianDiffusion
    schedule::S
    score_fn::T
end

function VEDiffusion(schedule::VENoiseSchedule; score_fn::T=nothing) where T
    VEDiffusion{T}(schedule, score_fn)
end

function get_drift_diffusion(d::VEDiffusion)
    drift(x,p,t) = 0f0
    diffusion(x,p,t) = sqrt(beta(d.schedule, t))
    return drift, diffusion
end

function marginal(d::VEDiffusion, x_start::AbstractArray, t::AbstractVector)
    shape = tuple([1 for _ in 1:ndims(x_start)-1]..., length(t))
    t = reshape(t, shape)
    std = alpha_cumulative(d.schedule, t)
    std = repeat(std, outer=(size(mean)[1:end-1]..., 1))
    return rand(MdNormal(x_start, std))
end

function sample_prior(d::VEDiffusion,  dims::Tuple{Int}; kwargs...)
    randn(dims, kwargs...) .* alpha_cumulative(d.schedule, 1.0)
end



### Shared functions ###

function get_diffeq_function(d::AbstractGaussianDiffusion)
    drift, diffusion = get_drift_diffusion(d)
    return SDEFunction(drift, diffusion)
end

function get_forward_diffeq(
    d::AbstractGaussianDiffusion,
    x::AbstractArray,
    tspan::Tuple{AbstractFloat, AbstractFloat},
)
    @assert tspan[1] < tspan[2]
    diffeq_fn = get_diffeq_function(d)
    prob = SDEProblem(diffeq_fn, x, tspan)
    return prob
end

# TODO: But what about ODE samplers? Convert SDE to ODE first?
# Reverse SDE is also an SDE (Anderson 1982). See Eqn. 6 from Song et al. 2021
function get_backward_diffeq(
    d::AbstractGaussianDiffusion,
    x::AbstractArray,
    tspan::Tuple{AbstractFloat, AbstractFloat},
)
    @assert tspan[1] > tspan[2]
    @assert !isnothing(d.score_fn)
    drift, diffusion = get_drift_diffusion(d) # TODO: Change to get_diffeq_function?
    reverse_drift(x,p,t) = drift(x,p,t) .- diffusion(x,p,t).^2 .* d.score_fn(x,p,t)
    prob = SDEProblem(drift, diffusion, x, tspan)
    return prob
end


# Older

function target(
    d::AbstractGaussianDiffusion,
    x_start::AbstractArray,
    x_t::AbstractArray,
    t::AbstractVector,
)
    # target depends on d.score_fn.parameterisation
    ...
end

function set_score_fn(d::D, score_fn) where D <: AbstractGaussianDiffusion
    return D(d.schedule, score_fn)
end

function get_forward_sde(
    d::AbstractGaussianDiffusion,
    x::AbstractArray,
    tspan::Tuple{AbstractFloat, AbstractFloat},
)
    @assert tspan[1] < tspan[2]
    drift, diffusion = get_drift_diffusion(d)
    prob = SDEProblem(drift, diffusion, x, tspan)
    return prob
end


# Reverse SDE is also an SDE (Anderson 1982). See Eqn. 6 from Song et al. 2021
function get_backward_sde(
    d::AbstractGaussianDiffusion,
    x::AbstractArray,
    tspan::Tuple{AbstractFloat, AbstractFloat},
)
    @assert tspan[1] > tspan[2]
    @assert !isnothing(d.score_fn)
    drift, diffusion = get_drift_diffusion(d)
    reverse_drift(x,p,t) = drift(x,p,t) .- diffusion(x,p,t).^2 .* d.score_fn(x,p,t)
    prob = SDEProblem(drift, diffusion, x, tspan)
    return prob
end

function get_ode(
    d::AbstractGaussianDiffusion,
    x::AbstractArray,
    tspan::Tuple{AbstractFloat, AbstractFloat},
)
    @assert !isnothing(d.score_fn)
    drift, diffusion = get_drift_diffusion(d)
    ode(x,p,t) = drift(x,p,t) - diffusion(x,p,t).^2 .* d.score_fn(x,p,t) .* 0.5
    prob = ODEProblem(ode, x, tspan)
    return prob
end

# TODO: dt is only for things like EM. For other solvers, want to use
# reltol, abstol etc. So instead have kwargs for the solver?
function transition(
    d::AbstractGaussianDiffusion,
    x_t::AbstractArray,
    tspan::Tuple{AbstractFloat, AbstractFloat},
    alg::StochasticDiffEqAlgorithm;
    kwargs...
)
    if tspan[1] < tspan[2]
        prob = get_forward_sde(d, x_t, tspan)
    elseif tspan[1] > tspan[2]
        prob = get_backward_sde(d, x_t, tspan)
    else
        throw(Error("TODO: Make proper error"))
    end
    sol = solve(prob, alg; kwargs...)
    return sol.u[end]
end

function transition(
    d::AbstractGaussianDiffusion,
    x_t::AbstractArray,
    tspan::Tuple{AbstractFloat, AbstractFloat},
    alg::OrdinaryDiffEqAlgorithm;
    kwargs...
)
    prob = get_ode(d, x_t, tspan)
    sol = solve(prob, alg; kwargs...)
    return sol.u[end]
end

# TODO: dtype and device
function sample(
    d::AbstractGaussianDiffusion,
    dims::NTuple{N, Int},
    alg::Union{StochasticDiffEqAlgorithm, OrdinaryDiffEqAlgorithm}=EM(),
    kwargs...
) where N
    x = sample_prior(d, dims)
    return transition(d, x, (1,0), alg; kwargs...)
end

function encode(
    d::AbstractGaussianDiffusion,
    x::AbstractArray,
    alg::OrdinaryDiffEqAlgorithm=Euler(),
    kwargs...
)
    return transition(d, x, (1,0), alg; kwargs...)
end

function denoising_loss_fn(
    d::AbstractGaussianDiffusion,
    model,
    x::AbstractArray;
    p=nothing,
    eps=1f-5,
)
    t = rand(size(x, ndims(x))) * (1.0 - eps) + eps
    marginal_dist = marginal(d, x, t)
    z = randn!(similar(x))
    perturbed_data = marginal_dist.mean .+ margin.std .* z
    score = d.score_fn(s, model, perturbed_data, p, t)

    losses = (score .* marginal_dist.std .+ z) .^ 2
    return mean(losses, dims=1:(ndims(losses)-1))
end


# TODO: Surely you want to make sure the diffusion model and the parameterisation have the same schedule?
# Not completely necessarily tbf, as you sometimes want a different schedule from train time to test time...
# So maybe actually this is the best way...
# As it also makes sense for the parameterisations to depend on the schedule, as look at the
# definitions below.

"""
    Score ∇ₓlog p(x) ≈ s(xₜ,t,θ) = f(xₜ,t,θ)
"""
abstract type AbstractScoreParameterisation <: AbstractModelParameterisation end

"""
    Predict the noise ϵ
    s(x,p,t) = -ϵ(x,t,θ)/σₜ
"""
struct NoiseScoreParameterisation{S<:AbstractNoiseSchedule} <: AbstractScoreParameterisation
    schedule::S
end

function get_target(
    p::NoiseScoreParameterisation,
    x_start::AbstractArray,
    x_t::AbstractArray,
    t::AbstractVector,
)
    ...
end

"""
    Predict x₀
    s(x,p,t) = -σₜ⁻²(x-αₜ⋅̂x(x,p,t))
"""
struct StartScoreParameterisation{S<:AbstractNoiseSchedule} <: AbstractScoreParameterisation
    schedule::S
end

"""
    Predict v, defined as
    v = αₜϵ + σₜx₀
    s(x,p,t) = -(v(x,p,t)+σₜx)/(αₜσₜ)
"""
struct VPredictScoreParameterisation{S<:AbstractNoiseSchedule} <: AbstractScoreParameterisation
    schedule::S
end

struct ScoreFunction{F, P} where {P<:AbstractScoreParameterisation}
    model::F
    parameterisation::P
end

function (f::ScoreFunction{F,P})(x,p,t) where {F,P::NoiseScoreParameterisation}
    return -f.model(x,p,t) ./ sigma(f.parameterisation.schedule, t)
end

function (f::ScoreFunction{F,P})(x,p,t) where {F,P::StartScoreParameterisation}
    sigma_t = sigma(f.parameterisation.schedule, t)
    alpha_t = alpha(f.parameterisation.schedule, t)
    return -sigma_t.^(-2) .* (x .- alpha_t .* f.model(x,p,t))
end

function (f::ScoreFunction{F,P})(x,p,t) where {F,P::VPredictScoreParameterisation}
    sigma_t = sigma(f.schedule, t)
    alpha_t = alpha(f.schedule, t)
    return -(f.model(x,p,t) .+ sigma_t) ./ (alpha_t .* sigma_t)
end
