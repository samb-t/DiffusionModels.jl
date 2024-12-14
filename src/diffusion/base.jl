abstract type AbstractDiffusion end

abstract type AbstractContinuousTimeDiffusion <: AbstractDiffusion end
# abstract type AbstractDiscreteTimeDiffusion <: AbstractContinuousTimeDiffusion end

abstract type AbstractCategoricalDiffusion <: AbstractDiffusion end

abstract type AbstractModelParameterisation end


# TODO: Make below functions have a required interface?
# If it turns out the interface is indeed the same


@doc raw"""
    marginal(d::AbstractGaussianDiffusion, x_start::AbstractArray, t::AbstractVector)
    marginal(d::DimensionalJumpDiffusion, x_start::AbstractArray, t::AbstractFloat)

Return the marginal distribution at time `t` given the initial data `x_start`.
"""
function marginal end


@doc raw"""
    sample_prior(d::GaussianDiffusion, dims::Tuple{N, Int})

Sample from the prior distribution of the Gaussian diffusion model.
"""
function sample_prior end


@doc raw"""
    get_diffeq_function(d::AbstractGaussianDiffusion)

Return the Differential Equation Function.
"""
function get_diffeq_function end


# TODO: Tspan should probably be stored in the Diffusion Model struct
# For now, assume always in [0,1]?
@doc raw"""
    get_forward_diffeq(d::AbstractGaussianDiffusion, x::AbstractArray,
        tspan::Tuple{AbstractFloat,AbstractFloat})

Return the forward Differential Equation Problem.
"""
function get_forward_diffeq end


@doc raw"""
    get_backward_diffeq(d::AbstractGaussianDiffusion, score_fn::ScoreFunction{F,P},
        x::AbstractArray, tspan::Tuple{AbstractFloat,AbstractFloat})

Return the backward Differential Equation Problem.
"""
function get_backward_diffeq end


@doc raw"""
    sample(d::AbstractGaussianDiffusion, score_fn::ScoreFunction{F,P}, dims::NTuple{N,Int},
        alg::AbstractSDEAlgorithm; dt::AbstractFloat, kwargs...)

Sample from the Diffusion model.
"""
function sample end


# TODO: Make this just loss_fn?
@doc raw"""
    denoising_loss_fn(d::AbstractGaussianDiffusion, x::AbstractArray,
        score_fn::ScoreFunction{F,P}; p=nothing, eps=1.0f-5)

Return the denoising loss function.
"""
function denoising_loss_fn end
