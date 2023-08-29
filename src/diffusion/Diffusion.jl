module Diffusion

using ProtoStructs # TODO: Remove protostructs
using Distributions
using Random
using MLUtils
using NNlib
# using Flux # TODO: Remove Flux calls
using DifferentialEquations
import Statistics.var
import Base.rand
using RecursiveArrayTools

# TODO: Replace with AbstractFFTs then let the user import a specific library
# Or maybe it needs to go into [extras]??
# TODO: Support DCT on GPU if available
using FFTW # TODO: Make optional dependency (for blurred diffusion)

export AbstractDiffusion, AbstractContinuousTimeDiffusion, AbstractDiscreteTimeDiffusion, AbstractGaussianDiffusion, AbstractCategoricalDiffusion

export AbstractSchedule, AbstractNoiseSchedule, VPNoiseSchedule, CosineSchedule, SigmoidSchedule, LinearSchedule, LinearMutualInfoSchedule

export VPDiffusion, VEDiffusion, CriticallyDampedDiffusion, BlurringDiffusion, DimensionalJumpDiffusion
export FrequencySchedule
export ConstantJumpSchedule, rate, rate_integral
export get_drift_diffusion, marginal, sample_prior, set_score_fn, get_forward_sde, get_backward_sde, get_ode
export get_jump, get_forward_diffeq
export transition, sample, encode
export MDNormal

include("base.jl")
include("noise_schedules.jl")
include("gaussian_diffusion.jl")
include("critically_damped.jl")
include("blurring_diffusion.jl")
include("jump_diffusion.jl")
include("dists.jl")

end