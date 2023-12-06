# __precompile__(false) # Breaks installation. Can only have this turned on for development
module DiffusionModels

    # using Flux

    # include("models/Models.jl")
    # using .Models
    # export Unet

    include("diffusion/Diffusion.jl")
    using .Diffusion
    export AbstractDiffusion, AbstractContinuousTimeDiffusion, AbstractDiscreteTimeDiffusion, AbstractGaussianDiffusion, AbstractCategoricalDiffusion,
           NoiseSchedule, VPNoiseSchedule, CosineSchedule, SigmoidSchedule, LinearSchedule, LinearMutualInfoSchedule,
           VPDiffusion, VEDiffusion, CriticallyDampedDiffusion, BlurringDiffusion, DimensionalJumpDiffusion,
           FrequencySchedule,
           ConstantJumpSchedule, rate, rate_integral,
           get_drift_diffusion, marginal, sample_prior, set_score_fn, get_forward_sde, get_backward_sde, get_ode, 
           get_jump, get_forward_diffeq,
           transition, sample, encode,
           MDNormal,
           alpha_cumulative, beta

end