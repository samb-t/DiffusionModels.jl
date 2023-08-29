"""
    A diffusion process described with the notation from 'Simple Diffusion':
        q(xₜ | x₀) = N(xₜ | αₜ⋅x₀, σₜ²⋅I)
    Stores the hyperparameters used to define the diffusion process.
"""
struct GaussianDiffusion <: AbstractGaussianDiffusion
    schedule::NoiseSchedule
    alpha::Function
    sigma::Function
end

"""
    Variance Preserving diffusion process originally from [1] defined so 
    that αₜ² = 1 - σₜ²:
        q(xₜ | x₀) = N(xₜ | √γₜ ⋅ x₀, (1-γₜ)⋅I)
    Using notation from [2].

    Args:
        `gamma::VPNoiseSchedule`: A variance preserving noise schedule.

    [1] Denoising Diffusion Probabilistic Models. 
        Ho et al. 2020.
    [2] On the Importance of Noise Scheduling for Diffusion Models. 
        Ting Chen 2023.
"""
function VPDiffusion(sched::VPNoiseSchedule)
    # alpha = t -> sqrt(sched(t))
    # sigma = t -> sqrt(1 - sched(t))
    alpha = t -> sqrt(alpha_cumulative(sched, t))
    sigma = t -> sqrt(1 - alpha_cumulative(sched, t))
    GaussianDiffusion(sched, alpha, sigma)
end

"""
    Variance Exploding diffusion process originally from [1]
    q(xₜ | x₀) = N(xₜ | x₀, σₜ²⋅I)

    Args:
        `sigma::VENoiseSchedule`: A variance exploding noise schedule.

    [1] Generative Modeling by Estimating Gradients of the Data Distribution. 
        Song and Ermon 2019.
"""
function VEDiffusion(sched::VENoiseSchedule)
    alpha = t -> 1
    sigma = t -> sched(t)
    GaussianDiffusion(sched, alpha, sigma)
end

"""
    Calculates the marginal distribution of a Simple Diffusion process
        q(xₜ | x₀) = N(xₜ | αₜ⋅x₀, σₜ²⋅I)
    
    Args:
        `d::GaussianDiffusion`: A Simple Diffusion object
        `x_start::AbstractArray`: x₀
        `t::AbstractMatrix`: The at which to get the marginal distribution
"""
function marginal(
    d::GaussianDiffusion, 
    x_start::AbstractArray, 
    t::AbstractVector,
)
    t = reshape(t, (1, 1, 1, length(t)))
    mean = @. d.alpha(t) * x_start
    std = @. d.sigma(t)
    std = repeat(std, outer=(size(mean)[1:end-1]..., 1))
    return MdNormal(mean, std)
end

"""
    Calculates the transition distribution of a Simple Diffusion process
    from x(t) to x(t + dt)
        q(x(t + dt) | x(t)) where dt > 0
        q(xₜ | xₛ) = N(xₜ | αₜₛ⋅zₛ, σₜₛ²⋅I)
            where αₜₛ = αₜ / αₛ
            and σₜₛ² = σₜ² - αₜₛ⋅σₛ²
            and t > s
    
    Args:
        `d::GaussianDiffusion`: A Simple Diffusion object
        `x_t::AbstractArray`: x₍
        `t::AbstractArray`:` The current time step
        `dt::AbstractMatrix`: The time step to transition by
"""
function q_transition(
    d::GaussianDiffusion, 
    x_t::AbstractArray,
    t::AbstractVector,
    dt::AbstractVector,
)
    t = reshape(t, (1, 1, 1, length(t)))
    next_t = t .+ dt
    alpha_ratio = @. d.alpha(next_t) / d.alpha(t)
    mean = alpha_ratio .* x_t
    std = @. sqrt(d.sigma(next_t)^2 - alpha_ratio * d.sigma(t)^2)
    std = repeat(std, outer=(size(mean)[1:end-1]..., 1))
    return MdNormal(mean, std)
end

# TODO: Currently this is only for VPDiffusion
#       Instead have drift_diffusion function separately for VP and VEDiffusion
#       And this can just use that Instead
#       Not sure about how to do marginal though unless that is separate for the different types
#       Or the struct stores the alpha/sigma functions but I'm not a fan of that
#       Or the schedule itself decides the different components for mean/variance??
# TODO: Pass in current_t and next_t, as well as dt=(next_t-current_t), which is for the solver???
# TODO: Ideally, this would be properly batched to allow different ts for different elements of the batch
function q_transition_solver(
    d::GaussianDiffusion, 
    x_t::AbstractArray,
    t::AbstractFloat,
    dt::AbstractFloat;
    alg::StochasticDiffEqAlgorithm=EM(),
)
    drift(x,p,t) = -0.5 * beta(d.schedule, t) .* x
    diffusion(x,p,t) = fill!(similar(x), sqrt(beta(d.schedule, t)))
    prob = SDEProblem(drift, diffusion, x_t, (t, t+dt))
    integrator = init(prob, alg; dt=dt)
    step!(integrator)
    return integrator.u
end

"""
    Calculates the posterior distribution of a Simple Diffusion process:
        q(xₛ | xₜ, x₀) = N(xₜ | μₜ₋ₛ, σₜ₋ₛ²⋅I)
            where s < t
            and  μₜ₋ₛ = αₜₛ⋅σₛ²/σₜ²⋅xₜ + αₛ⋅σₜₛ²/σₜ²⋅x₀
            and σₜ₋ₛ² = σₜₛ²⋅σₛ²/σₜ²
    
    Args:
        `d::GaussianDiffusion`: A Simple Diffusion object
        `x_t::DiffusionState`: x₍ and t
        `x_start::AbstractArray`: x₀
        `dt::AbstractMatrix`: The time step to transition by
"""
function q_posterior(
    d::GaussianDiffusion, 
    x_t::AbstractArray,
    t::AbstractVector,
    x_start::AbstractArray, 
    dt::AbstractVector,
)
    t = reshape(t, (1, 1, 1, length(t)))
    next_t = t .- dt

    alpha_t = @. d.alpha(t)
    alpha_next_t = @. d.alpha(next_t)
    var_t = @. d.sigma(t) ^ 2
    var_next_t = @. d.sigma(next_t) ^ 2

    alpha_ratio = @. alpha_t / alpha_next_t
    var_ratio = @. var_next_t / var_t
    var_change = @. d.sigma(t)^2 - alpha_ratio * d.sigma(next_t)^2

    mean = @. alpha_ratio * var_ratio * x_t +
        alpha_next_t * var_change / var_t * x_start
    std = @. sqrt(var_change *  var_ratio)
    std = repeat(std, outer=(size(mean)[1:end-1]..., 1))

    return MdNormal(mean, std)
end

# TODO: Might want to use different gamma fn when sampling
# i.e. small betas vs large betas in VP
# p(xₛ | xₜ) = q(xₛ | xₜ, x₀)
"""
    A single transition through the reverse diffusion process
"""
function p_transition(
    d::GaussianDiffusion, 
    x_t::AbstractArray,
    t::AbstractVector,
    dt::AbstractVector,
    denoising_fn::Function,
)
    pred_x_start = denoising_fn(x_t, x_t).pred_x_start
    x_next = q_posterior(d, x_t, t, pred_x_start, dt)
    return x_next
end


"""
    Sample x(T).
"""
function prior(
    d::GaussianDiffusion, 
    dims::Tuple{Int};
    kwargs...,
)
    randn(dims, kwargs...)
end