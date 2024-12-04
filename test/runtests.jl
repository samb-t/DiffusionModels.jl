using Test
using JET
using RequiredInterfaces: check_interface_implemented
using Random

using DiffusionModels
# TODO: Why is below needed? They're exported
using DiffusionModels: AbstractGaussianNoiseSchedule
using DiffusionModels: marginal_mean_coeff
using DiffusionModels: marginal_std_coeff
using DiffusionModels: drift_coeff
using DiffusionModels: diffusion_coeff
using DiffusionModels: log_snr
using DiffusionModels: beta

using DiffusionModels: AbstractScoreParameterisation
using DiffusionModels: get_target
using DiffusionModels: NoiseScoreParameterisation
using DiffusionModels: StartScoreParameterisation
using DiffusionModels: VPredictScoreParameterisation
using DiffusionModels: ScoreFunction

using JET: JET, JETTestFailure, get_reports, report_call, report_opt
# XXX: In 1.11, JET leads to stack overflows
# global JET_TESTING_ENABLED = v"1.10-" ≤ VERSION < v"1.11-"
global JET_TESTING_ENABLED = true # hope for the best...

@testset verbose = true "DiffusionModels.jl" begin
    include("test_noise_schedules.jl")
    include("test_gaussian_diffusion.jl")
end
