module Diffusion

# using ProtoStructs # TODO: Remove protostructs
# TODO: Use RedefStructs?
using Distributions
using Random
using MLUtils
using NNlib
# using Flux # TODO: Remove Flux calls
# using DifferentialEquations
import Statistics.var
import Base.rand
using StochasticDiffEq
using OrdinaryDiffEq
using JumpProcesses
using OptimalTransport # TODO: Request bump in NNLib version in OptimalTransport.jl
using Distances: SqEuclidean
using StatsBase: ProbabilityWeights
using RecursiveArrayTools
using RequiredInterfaces

# TODO: Replace with AbstractFFTs then let the user import a specific library
# Or maybe it needs to go into [extras]??
# TODO: Support DCT on GPU if available
# using FFTW # TODO: Make optional dependency (for blurred diffusion)

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

export AbstractDiffusion
export AbstractContinuousTimeDiffusion
export AbstractDiscreteTimeDiffusion
export AbstractGaussianDiffusion
export AbstractCategoricalDiffusion



export GaussianDiffusion
export CriticallyDampedDiffusion
export DimensionalJumpDiffusion
# export VPDiffusion, VEDiffusion, CriticallyDampedDiffusion, BlurringDiffusion,

# export FrequencySchedule
export ConstantJumpSchedule, rate, rate_integral
export get_drift_diffusion, marginal, sample_prior, set_score_fn, get_diffeq_function, get_forward_diffeq, get_backward_diffeq
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

include("base.jl")
include("noise_schedules.jl")
include("gaussian_diffusion.jl")
include("critically_damped.jl")
# Removed for now since doesn't work and to remove FFTW dependency.
# include("blurring_diffusion.jl")
include("jump_diffusion.jl")
include("dists.jl")

end
