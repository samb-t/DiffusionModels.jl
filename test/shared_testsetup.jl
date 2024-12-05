@testsnippet SharedTestSetup begin

    using Test
    using JET
    using RequiredInterfaces: check_interface_implemented
    using Random
    using Statistics
    using SciMLBase
    using StochasticDiffEq
    using TestItemRunner

    using DiffusionModels
    # TODO: Why is below needed? They're exported
    using DiffusionModels: MdNormal

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

    using DiffusionModels: AbstractGaussianDiffusion
    using DiffusionModels: VPDiffusion
    using DiffusionModels: marginal
    using DiffusionModels: sample_prior
    using DiffusionModels: get_drift_diffusion
    using DiffusionModels: get_diffeq_function
    using DiffusionModels: get_forward_diffeq
    using DiffusionModels: get_backward_diffeq
    using DiffusionModels: sample
    using DiffusionModels: denoising_loss_fn

    using JET: JET, JETTestFailure, get_reports, report_call, report_opt
    # XXX: In 1.11, JET leads to stack overflows
    # global JET_TESTING_ENABLED = v"1.10-" ≤ VERSION < v"1.11-"
    global JET_TESTING_ENABLED = true # hope for the best...

end
