# __precompile__(false) # Breaks installation. Can only have this turned on for development
module DiffusionModels
    using Distributions
    using Random
    using MLUtils
    using NNlib
    # using Flux # TODO: Remove Flux calls
    # using DifferentialEquations
    import Statistics
    import Statistics.var
    import Base.rand
    using SciMLBase
    using SciMLBase: AbstractSDEAlgorithm
    using JumpProcesses
    using OptimalTransport # TODO: Request bump in NNLib version in OptimalTransport.jl
    using Distances: SqEuclidean
    using StatsBase: ProbabilityWeights
    using RecursiveArrayTools
    using RequiredInterfaces


    export MdNormal
    export AbstractSchedule
    export AbstractNoiseSchedule
    export AbstractGaussianNoiseSchedule
    export VPNoiseSchedule
    export CosineSchedule
    export LinearSchedule
    export LinearMutualInfoSchedule
    #, SigmoidSchedule, LinearMutualInfoSchedule

    export AbstractScoreParameterisation
    export get_target
    export NoiseScoreParameterisation
    export StartScoreParameterisation
    export VPredictScoreParameterisation

    export ScoreFunction

    export AbstractDiffusion
    export AbstractContinuousTimeDiffusion
    export AbstractDiscreteTimeDiffusion
    export AbstractGaussianDiffusion
    export AbstractCategoricalDiffusion

    export GaussianDiffusion
    export CriticallyDampedDiffusion
    export DimensionalJumpDiffusion
    # export VPDiffusion, VEDiffusion, CriticallyDampedDiffusion, BlurringDiffusion,

    export marginal
    export sample_prior
    export get_drift_diffusion
    export get_diffeq_function
    export get_forward_diffeq
    export get_backward_diffeq
    export sample
    export denoising_loss_fn

    # export FrequencySchedule
    export ConstantJumpSchedule, rate, rate_integral
    # export get_drift_diffusion, marginal, sample_prior, set_score_fn, get_diffeq_function, get_forward_diffeq, get_backward_diffeq
    export get_jump
    export sample
    # export transition, sample, encode
    export MDNormal
    # export alpha_cumulative, beta

    export marginal_mean_coeff
    export marginal_std_coeff
    export drift_coeff
    export diffusion_coeff
    export log_snr
    export beta

    include("diffusion/base.jl")
    include("diffusion/noise_schedules.jl")
    include("diffusion/gaussian_diffusion.jl")
    include("diffusion/critically_damped.jl")
    # Removed for now since doesn't work and to remove FFTW dependency.
    # include("blurring_diffusion.jl")
    include("diffusion/jump_diffusion.jl")
    include("diffusion/dists.jl")

end
