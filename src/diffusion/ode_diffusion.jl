# ODE Diffusion from
# - "Elucidating the Design Space of Diffusion-Based Generative Models"
# - "Building Normalizing Flows with Stochastic Interpolants"

abstract type AbstractODEDiffusion <: AbstractDiffusion end

"""
    Probability flow diffusion from [1]
        dx = -\dot{\sigma}(t)\sigma(t)\nabla_x \log p(x;\sigma(t)) dt
    where the dot denotes a time derivative.

    [1] Elucidating the Design Space of Diffusion-Based Generative Models
        Karras et al. NeurIPS 2022.
"""
struct EDMDiffusion{T} <: AbstractODEDiffusion
    schedule::NoiseSchedule
    score_fn::T
end