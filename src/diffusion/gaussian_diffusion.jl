# TODO:
# 1. Convert to AbstractSDEDiffusion
# 2. Allow converting to ODEDiffusion
# 3. Allow converting to a discrete diffusion?
#       - as SciMLBase.DiscreteProblem
#       - DiscreteProblem(f::ODEFunction, u0, tspan) is already allowed

@doc raw"""
    AbstractScoreParameterisation

Abstract type for parameterisations of the score function ``\nabla_x \log p(x)``.

Diffusion models are typically not trained to directly predict the score, but are
trained with different objectives that result in higher sample quality or are
more stable to optimise.

`AbstractScoreParameterisation`s are used for the following reasons:
- to define a `get_target` method that returns the target used in the training
  objective.
- to define a `ScoreFunction` that takes the model and the parameterisation
  and returns the score.
"""
abstract type AbstractScoreParameterisation <: AbstractModelParameterisation end

@doc raw"""
    get_target(
        parameterisation::AbstractScoreParameterisation,
        x_start::AbstractArray,
        noise::AbstractArray,
        t::AbstractVector,
    )

Return the target used in the training objective.

## Arguments
- `parameterisation::AbstractScoreParameterisation`: The parameterisation of the score function.
- `x_start::AbstractArray`: The initial data.
- `noise::AbstractArray`: The noise.
- `t::AbstractVector`: The time points.

## Returns
- `target::AbstractArray`: The target used in the training objective.

## Example

```jldoctest
julia> schedule = CosineSchedule{Float32}()
julia> parameterisation = NoiseScoreParameterisation(schedule)
julia> x_start = randn(Float32, 10, 3)
julia> noise = randn(Float32, 10, 3)
julia> t = rand(Float32, 3)
julia> target = get_target(parameterisation, x_start, noise, t)
julia> size(target)
(10, 3)
julia> target == noise
true
```
"""
function get_target end

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

@doc raw"""
    ScoreFunction(model, parameterisation<:AbstractScoreParameterisation)
    (::ScoreFunction)(x, p, t)

Takes a model trained to predict the specified `parameterisation`. When
called, returns the score.

## Arguments

- `model`: The model trained to predict the specified `parameterisation`.
- `parameterisation`: The parameterisation of the score function.

## Returns

- `score::AbstractArray`: The score.

## Example

```jldoctest
julia> model = (x,p,t) -> x
julia> schedule = CosineSchedule{Float32}()
julia> parameterisation = StartScoreParameterisation(schedule)
julia> score_fn = ScoreFunction(model, parameterisation)
julia> x = randn(Float32, 10, 3)
julia> p = nothing
julia> t = rand(Float32, 3)
julia> score = score_fn(x, p, t)
julia> size(score)
(10, 3)
```
"""
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

@doc raw"""
    AbstractGaussianDiffusion

Abstract type for Gaussian diffusions.

```math
dx = f(x, t)dt + g(t)dw
```

where `f` is the drift and `g` is the diffusion coefficient.
"""
abstract type AbstractGaussianDiffusion <: AbstractDiffusion end

# NOTE: RequiredInterfaces doesn't support all of these function definitions yet

# TODO: Pop dims and tspan in here and use the @concrete macro?
@doc raw"""
    GaussianDiffusion(schedule::AbstractGaussianNoiseSchedule)

A Gaussian diffusion model with a specified noise schedule. The noise schedule
is used to define the drift and diffusion coefficients.

## Example

```jldoctest
julia> schedule = CosineSchedule{Float32}()
julia> diffusion = GaussianDiffusion(schedule)
julia> x_start = randn(Float32, 10, 3)
julia> t = rand(Float32, 3)
julia> dist = marginal(diffusion, x_start, t)
julia> size(mean(dist))
(10, 3)
julia> size(std(dist))
(10, 3)
```
"""
struct GaussianDiffusion{S<:AbstractGaussianNoiseSchedule} <: AbstractGaussianDiffusion
    schedule::S
end

# TODO: Should all funcs below be defined based on AbstractGaussianDiffusion or GaussianDiffusion{S}???

@doc raw"""
    marginal(d::AbstractGaussianDiffusion, x_start::AbstractArray, t::AbstractVector)

Return the marginal distribution at time `t` given the initial data `x_start`.
"""
function marginal(d::AbstractGaussianDiffusion, x_start::AbstractArray, t::AbstractVector)
    shape = ((1 for _ in 1:(ndims(x_start) - 1))..., length(t))

    mean_coeff = marginal_mean_coeff(d.schedule, t)
    std = marginal_std_coeff(d.schedule, t)

    mean_coeff = reshape(mean_coeff, shape)
    std = reshape(std, shape)
    std = repeat(std; outer=(size(x_start)[1:(end - 1)]..., 1))

    mean = mean_coeff .* x_start

    return MdNormal(mean, std)
end

@doc raw"""
    get_drift_diffusion(d::AbstractGaussianDiffusion)

Return the drift and diffusion functions for the Gaussian diffusion model.
"""
function get_drift_diffusion(d::AbstractGaussianDiffusion)
    drift(x, p, t) = -0.5 .* drift_coeff(d.schedule, t) .* x
    diffusion(x, p, t) = diffusion_coeff(d.schedule, t)
    return drift, diffusion
end
# TODO: Should the -0.5 here be pulled into the schedule?

@doc raw"""
    sample_prior(d::GaussianDiffusion, dims::Tuple{N, Int})

Sample from the prior distribution of the Gaussian diffusion model.
"""
function sample_prior(d::GaussianDiffusion{S}, dims::Tuple{N,Int}) where {S,N}
    # Works for now but should have a better solution
    T = S.parameters[1]
    # NOTE: This way the mean is actually very close to 0 but not quite for some schedules.
    # More of a schedule problem than this function though
    prior = marginal(d, ones(T, dims), ones(T, dims[end]))
    sample = rand(prior)
    return sample
end
# TODO: If all other functions are on AbstractGaussianDiffusion, this should be too.
# TODO: Need device, dtype etc.

@doc raw"""
    get_diffeq_function(d::AbstractGaussianDiffusion)

Return the `SDEfunction` for the Gaussian diffusion model.
"""
function get_diffeq_function(d::AbstractGaussianDiffusion)
    drift, diffusion = get_drift_diffusion(d)
    return SDEFunction(drift, diffusion)
end

@doc raw"""
    get_forward_diffeq(
        d::AbstractGaussianDiffusion,
        x::AbstractArray,
        tspan::Tuple{AbstractFloat,AbstractFloat},
    )

Return the forward `SDEProblem` for the Gaussian diffusion model.
"""
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

@doc raw"""
    get_backward_diffeq(
        d::AbstractGaussianDiffusion,
        score_fn::ScoreFunction{F,P},
        x::AbstractArray,
        tspan::Tuple{AbstractFloat,AbstractFloat},
    )

Return the backward `SDEProblem` for the Gaussian diffusion model.
"""
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

@doc raw"""
    sample(
        d::AbstractGaussianDiffusion,
        score_fn::ScoreFunction{F,P},
        dims::NTuple{N,Int},
        alg::AbstractSDEAlgorithm;
        dt::AbstractFloat,
        kwargs...
    )

Sample from the Gaussian diffusion model using the specified `score_fn` and `alg`.
"""
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
# TODO: dtype and device
# TODO: Store dims within GaussianDiffusion? I think yes

@doc raw"""
    denoising_loss_fn(
        d::AbstractGaussianDiffusion,
        x::AbstractArray,
        score_fn::ScoreFunction{F,P};
        p=nothing,
        eps=1.0f-5,
    )

Return the denoising loss for the Gaussian diffusion model.
"""
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
# TODO: Don't think you should need to pass in d here. Use
# score_fn.parameterisation.schedule instead
