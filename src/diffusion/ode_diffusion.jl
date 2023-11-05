# ODE Diffusion from
# - "Elucidating the Design Space of Diffusion-Based Generative Models"
# - "Building Normalizing Flows with Stochastic Interpolants"

abstract type AbstractODEDiffusion <: AbstractDiffusion end

# """
#     Probability flow diffusion from [1]
#         dx = -\dot{\sigma}(t)\sigma(t)\nabla_x \log p(x;\sigma(t)) dt
#     where the dot denotes a time derivative.

#     [1] Elucidating the Design Space of Diffusion-Based Generative Models
#         Karras et al. NeurIPS 2022.
# """
# struct EDMDiffusion{T} <: AbstractODEDiffusion
#     schedule::NoiseSchedule
#     score_fn::T
# end


# For ease start with a one sided flow (where x_end \sim N(0,I))?
# In which case subclass this with OneSidedODEDiffusion?
struct VPFlow <: AbstractODEDiffusion
    schedule::VPNoiseSchedule
    sigma_min
end

function marginal(
    d::VPFlow, 
    x_start::AbstractArray, 
    t::AbstractVector,
)
    shape = tuple([1 for _ in 1:ndims(x_start)-1]..., length(t))
    t = reshape(t, shape)
    mean = sqrt.(alpha_cumulative(d.schedule, t)) .* x_start
    std = sqrt.(1 .- alpha_cumulative(d.schedule, t))
    std = repeat(std, outer=(size(mean)[1:end-1]..., 1))
    return rand(MdNormal(mean, std))
    # TODO: Also need to add sigma_min * noise here
end

# For constant sigma_min like here, this is just d/dt μ(t)
# where μ(t) is the marginal density above
function conditional_vector_field()

end

# Also possible to define for arbitrary start/end densities. 
# In this case
# function marginal(
#     d::VPFlow, 
#     x_start::AbstractArray, 
#     x_end::AbstractArray, 
#     t::AbstractVector,
# )

# end

