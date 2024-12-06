# TODO:
# 1. Convert to AbstractSDEDiffusion
# 2. Allow converting to ODEDiffusion
# 3. Allow converting to a discrete diffusion?
#       - as SciMLBase.DiscreteProblem
#       - DiscreteProblem(f::ODEFunction, u0, tspan) is already allowed

"""
    Score ∇ₓlog p(x) ≈ s(xₜ,t,θ) = f(xₜ,t,θ)
"""
abstract type AbstractScoreParameterisation <: AbstractModelParameterisation end

@required AbstractScoreParameterisation begin
    get_target(
        ::AbstractScoreParameterisation, ::AbstractArray, ::AbstractArray, ::AbstractVector
    )
end

"""
    Predict the noise ϵ
    s(x,p,t) = -ϵ(x,t,θ)/σₜ
"""
struct NoiseScoreParameterisation{S<:AbstractNoiseSchedule} <: AbstractScoreParameterisation
    schedule::S
end

# For simplicity for now. But some of these args can definitely go away
function get_target(
    ::NoiseScoreParameterisation,
    x_start::AbstractArray,
    noise::AbstractArray,
    t::AbstractVector,
)
    return noise
end

"""
    Predict x₀
    s(x,p,t) = -σₜ⁻²(x-αₜ⋅̂x(x,p,t))
"""
struct StartScoreParameterisation{S<:AbstractNoiseSchedule} <: AbstractScoreParameterisation
    schedule::S
end

function get_target(
    ::StartScoreParameterisation,
    x_start::AbstractArray,
    noise::AbstractArray,
    t::AbstractVector,
)
    return x_start
end

"""
    Predict v, defined as
    v = αₜϵ + σₜx₀
    s(x,p,t) = -(v(x,p,t)+σₜx)/(αₜσₜ)
"""
struct VPredictScoreParameterisation{S<:AbstractNoiseSchedule} <:
       AbstractScoreParameterisation
    schedule::S
end

function get_target(
    parameterisation::VPredictScoreParameterisation,
    x_start::AbstractArray,
    noise::AbstractArray,
    t::AbstractVector,
)
    shape = ((1 for _ in 1:(ndims(x_start) - 1))..., length(t))
    sigma_t = marginal_std_coeff(parameterisation.schedule, t)
    alpha_t = marginal_mean_coeff(parameterisation.schedule, t)
    sigma_t = reshape(sigma_t, shape)
    alpha_t = reshape(alpha_t, shape)
    v = alpha_t .* noise .+ sigma_t .* x_start
    return v
end

struct ScoreFunction{F,P<:AbstractScoreParameterisation}
    model::F
    parameterisation::P
end

function (f::ScoreFunction{F,P})(x, p, t) where {F,P<:NoiseScoreParameterisation}
    return -f.model(x, p, t) ./ marginal_std_coeff(f.parameterisation.schedule, t)
end

function (f::ScoreFunction{F,P})(x, p, t) where {F,P<:StartScoreParameterisation}
    sigma_t = marginal_std_coeff(f.parameterisation.schedule, t)
    alpha_t = marginal_mean_coeff(f.parameterisation.schedule, t)
    return -sigma_t .^ (-2) .* (x .- alpha_t .* f.model(x, p, t))
end

function (f::ScoreFunction{F,P})(x, p, t) where {F,P<:VPredictScoreParameterisation}
    sigma_t = marginal_std_coeff(f.parameterisation.schedule, t)
    alpha_t = marginal_mean_coeff(f.parameterisation.schedule, t)
    return -(f.model(x, p, t) .+ sigma_t) ./ (alpha_t .* sigma_t)
end

# NOTE: RequiredInterfaces doesn't support all of these function definitions yet

# TODO: Pop dims and tspan in here and use the @concrete macro?
struct VPDiffusion{S<:VPNoiseSchedule} <: AbstractGaussianDiffusion
    schedule::S
end

struct VEDiffusion{S<:VENoiseSchedule} <: AbstractGaussianDiffusion
    schedule::S
end

function marginal(d::VPDiffusion, x_start::AbstractArray, t::AbstractVector)
    shape = ((1 for _ in 1:(ndims(x_start) - 1))..., length(t))

    mean_coeff = marginal_mean_coeff(d.schedule, t)
    std = marginal_std_coeff(d.schedule, t)

    mean_coeff = reshape(mean_coeff, shape)
    std = reshape(std, shape)
    std = repeat(std; outer=(size(x_start)[1:(end - 1)]..., 1))

    mean = mean_coeff .* x_start

    return MdNormal(mean, std)
end

function get_drift_diffusion(d::VPDiffusion)
    drift(x, p, t) = -0.5 .* drift_coeff(d.schedule, t) .* x
    diffusion(x, p, t) = diffusion_coeff(d.schedule, t)
    return drift, diffusion
end

# TODO: Not true! For VE Diffusion this is much larger!
# TODO: Need device etc.
function sample_prior(d::AbstractGaussianDiffusion, dims::Tuple{N, Int}; kwargs...) where N
    return randn(dims, kwargs...)
end

function get_diffeq_function(d::AbstractGaussianDiffusion)
    drift, diffusion = get_drift_diffusion(d)
    return SDEFunction(drift, diffusion)
end

function get_forward_diffeq(
    d::AbstractGaussianDiffusion,
    x::AbstractArray,
    tspan::Tuple{AbstractFloat,AbstractFloat},
)
    @assert tspan[1] < tspan[2]
    diffeq_fn = get_diffeq_function(d)
    prob = SDEProblem(diffeq_fn, x, tspan)
    return prob
end

function get_backward_diffeq(
    d::AbstractGaussianDiffusion,
    score_fn::ScoreFunction{F,P},
    x::AbstractArray,
    tspan::Tuple{AbstractFloat,AbstractFloat},
) where {F,P}
    @assert tspan[1] > tspan[2]
    @assert !isnothing(score_fn)
    drift, diffusion = get_drift_diffusion(d) # TODO: Change to get_diffeq_function?
    reverse_drift(x, p, t) = drift(x, p, t) .- diffusion(x, p, t) .^ 2 .* score_fn(x, p, t)
    diffeq_fn = SDEFunction(reverse_drift, diffusion)
    prob = SDEProblem(diffeq_fn, x, tspan)
    return prob
end

# TODO: dtype and device
# TODO: Store dims within GaussianDiffusion? I think yes
function sample(
    d::AbstractGaussianDiffusion,
    score_fn::ScoreFunction{F,P},
    dims::NTuple{N,Int},
    alg::AbstractSDEAlgorithm;
    dt::AbstractFloat,
    kwargs...,
) where {F,P,N}
    x = sample_prior(d, dims)
    prob = get_backward_diffeq(d, score_fn, x, (1.0, 0.0))
    sol = solve(prob, alg; dt=dt, kwargs...)
    return sol
end

# TODO: Don't think you should need to pass in d here. Use
# score_fn.parameterisation.schedule instead
function denoising_loss_fn(
    d::AbstractGaussianDiffusion,
    x::AbstractArray,
    score_fn::ScoreFunction{F,P};
    p=nothing,
    eps=1.0f-5,
) where {F,P}
    t = rand(size(x, ndims(x))) .* (1.0 - eps) .+ eps
    marginal_dist = marginal(d, x, t)
    z = randn!(similar(x))
    perturbed_data = marginal_dist.mean .+ marginal_dist.std .* z

    model = score_fn.model
    parameterisation = score_fn.parameterisation

    target = get_target(parameterisation, x, z, t)

    pred = model(perturbed_data, p, t)

    losses = (pred .- target) .^ 2
    return mean(losses) #mean(losses; dims=1:(ndims(losses) - 1))
end
