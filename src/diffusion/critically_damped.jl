# """
#     Critically Damped Diffusion Process from [1]
#         [dxₜ  dvₜ] = [M⁻¹vₜ  -xₜ]β dt + [0  -ΓM⁻¹vₜ]β dt + [0  √(2Γβ)] dw
#                      |______  ______|   |______________  _______________|
#                             ||                         ||
#                   Hamiltonian component     Ornstein-Uhlbeck process

#     Runs diffusion in an augmented space with "velocities" vₜ. Noise is applied
#     in velocity space rather than in data space xₜ resulting in a smoother
#     diffusion that is easier to learn and sample from.

#     Adapted from https://github.com/nv-tlabs/CLD-SGM

#     Args:
#         `schedule::VPNoiseSchedule`: A variance preserving noise schedule.
#         `score_fn` (optional): The score function ∇log p(xₜ).
#         `m_inv` (optional): The inverse mass M⁻¹.
#         `dim` (optional): Dimension to split u into x and v e.g. for images dim=3.
#         `gamma` (optional): How noisy the initial velocity is, p(v₀)=N(0,γMI).

#     [1] Score-Based Generative Modeling with Critically-Damped Langevin Diffusion
#         Dockhorn et al. ICLR 2022.
# """

struct CriticallyDampedDiffusion{T} <: AbstractGaussianDiffusion
    schedule::VPNoiseSchedule
    score_fn::T
    m_inv
    dim
    gamma
end

function CriticallyDampedDiffusion(
    schedule::VPNoiseSchedule;
    score_fn::T=nothing,
    m_inv=4.0,
    dim=1,
    gamma=0.04,
) where T
    CriticallyDampedDiffusion{T}(schedule, score_fn, m_inv, dim, gamma)
end

function get_drift_diffusion(d::CriticallyDampedDiffusion)
    drift(x,p,t) = begin
        x, v = chunk(x, 2, dims=d.dim) # from MLUtils
        f = 2 / sqrt(d.m_inv)
        beta_t = beta(d.schedule, t)
        drift_x = d.m_inv * beta_t .* v
        drift_v = -beta_t .* x .- f .* d.m_inv .* beta_t .* v
        return cat(drift_x, drift_v, dims=d.dim)
    end
    diffusion(x,p,t) = begin
        x, v = chunk(x, 2, dims=d.dim)
        f = 2 / sqrt(d.m_inv)
        beta_t = beta(d.schedule, t)
        diffusion_x = fill!(similar(x), 0)
        diffusion_v = fill!(similar(v), sqrt(2 * f * beta_t))
        return cat(diffusion_x, diffusion_v, dims=d.dim)
    end
    return drift, diffusion
end

function marginal(
    d::CriticallyDampedDiffusion,
    x_start::AbstractArray,
    t::AbstractVector
)
    x, v = chunk(x_start, 2, dims=d.dim)
    f = 2 / sqrt(d.m_inv)
    g = 1 / f
    t = reshape(t, (1, 1, 1, length(t)))
    alpha_int = alpha_cumulative(d.schedule, t)
    beta_int = -log.(alpha_int)
    coeff_mean = alpha_int .^ (2.0 * g)

    ### Compute Mean ###
    mean_x = @. coeff_mean * (2 * beta_int * g * x + 4 * beta_int * g^2 * v + x)
    mean_v = @. coeff_mean * (-beta_int * x - 2 * beta_int * g * v + v)
    mean = cat(mean_x, mean_v, dims=d.dim)

    ### Compute Covariance ###
    # Variances at time t=0
    var0x = fill!(similar(x), 0)
    var0v = fill!(similar(x), d.gamma / d.m_inv)

    # Calculate "per-dimension" covariance matrix
    multiplier = coeff_mean .^ 2
    var_xx = @. var0x + (1 / multiplier) - 1 + 4 * beta_int * g * (var0x - 1) +
             4 * beta_int ^ 2 * g ^ 2 * (var0x - 2) + 16 * g ^ 4 * beta_int ^ 2 * var0v
    var_xv = @. -var0x * beta_int + 4 * g ^ 2 * beta_int * var0v -
             2 * g * beta_int ^ 2 * (var0x - 2) - 8 * g ^ 3 * beta_int ^ 2 * var0v
    var_vv = @. f ^ 2 * ((1 / multiplier) - 1) / 4 + f * beta_int - 4 * g * beta_int * var0v +
             4 * g ^ 2 * beta_int ^ 2 * var0v + var0v + beta_int ^ 2 * (var0x - 2)
    var_xx = var_xx .* multiplier .+ 1e-6
    var_xv = var_xv .* multiplier
    var_vv = var_vv .* multiplier .+ 1e-6

    # Cholesky decomposition
    cholesky11 = sqrt.(var_xx)
    cholesky21 = var_xv ./ cholesky11
    cholesky22 = sqrt.(var_vv .- cholesky21 .^ 2)

    batch_noise = randn!(similar(x_start))
    batch_noise_x, batch_noise_v = chunk(batch_noise, 2, dims=d.dim)

    noise_x = cholesky11 .* batch_noise_x
    noise_v = cholesky21 .* batch_noise_x .+ cholesky22 .* batch_noise_v
    noise = cat(noise_x, noise_v, dims=d.dim)

    perturbed_data = mean .+ noise

    return perturbed_data
end

function sample_prior(d::CriticallyDampedDiffusion,  dims::Tuple{Int}; kwargs...)
    x = randn(dims, kwargs...)
    v = randn(dims, kwargs...) .* (d.gamma / d.m_inv)
    return cat(x, v, dims=d.dim)
end
