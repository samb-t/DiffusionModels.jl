# Look at Lux.jl tests and copy what they do
# E.g. Run @jet on function calls

# Test ScoreFunction outputs in the correct shape
# 1. Mock the model
# 2. Mock the parameterisation
# 2.1 Mock the schedule

# struct MockScoreParameterisation <: AbstractScoreParameterisation end
# marginal_std_coeff()

@testitem "Test Score Parameterisations" setup=[SharedTestSetup] begin
    @testset "Test $(nameof(ScoreParameterisation))" for ScoreParameterisation in [
        NoiseScoreParameterisation,
        StartScoreParameterisation,
        VPredictScoreParameterisation,
    ]
        @test check_interface_implemented(AbstractScoreParameterisation, ScoreParameterisation)

        # TODO: Probably some good way to do this with a mock
        schedule = CosineSchedule{Float64}()
        parameterisation = ScoreParameterisation(schedule)
        x_start = randn(Xoshiro(0), 10, 3)
        noise = randn(Xoshiro(1), 10, 3)
        t = rand(Xoshiro(2), 3)

        # JET
        if JET_TESTING_ENABLED
            @test_opt target_modules=(DiffusionModels,) get_target(parameterisation, x_start, noise, t)
            @test_call get_target(parameterisation, x_start, noise, t)
        end

        # Test correctness
        @test get_target(parameterisation, x_start, noise, t) isa AbstractArray
        @test size(get_target(parameterisation, x_start, noise, t)) == size(x_start)
    end
end


@testitem "Test ScoreFunction Inference" setup=[SharedTestSetup] begin
    @testset "Test ScoreFunction{F,$(nameof(ScoreParameterisation))}" for ScoreParameterisation in [
        NoiseScoreParameterisation,
        StartScoreParameterisation,
        VPredictScoreParameterisation,
    ]
        # TODO: could use Lux for this
        model = (x,p,t) -> x
        # TODO: Mock the schedule
        schedule = CosineSchedule{Float64}()

        parameterisation = ScoreParameterisation(schedule)
        score_function = ScoreFunction(model, parameterisation)

        x = randn(Xoshiro(0), 10, 3)
        p = nothing
        # TODO: Does this need to be tested with t as a vector?
        t = 0.3

        # JET
        if JET_TESTING_ENABLED
            @test_opt target_modules=(DiffusionModels,) score_function(x, p, t)
            @test_call score_function(x, p, t)
        end

        # Test correctness
        @test score_function(x, p, t) isa AbstractArray
        @test size(score_function(x, p, t)) == size(x)
    end
end

@testsnippet GaussianDiffusionSetup begin
    schedule = CosineSchedule{Float64}()
    diffusion_model = GaussianDiffusion(schedule)
    x_start = randn(Xoshiro(0), 10, 3)
    noise = randn(Xoshiro(1), 10, 3)
    t = rand(Xoshiro(2), 3)
end


@testitem "Test marginal(::GaussianDiffusion, ...)" setup=[SharedTestSetup, GaussianDiffusionSetup] begin
    p_x_t = marginal(diffusion_model, x_start, t)
    @test p_x_t isa MdNormal
    @test size(mean(p_x_t)) == size(x_start)
    @test size(rand(p_x_t)) == size(x_start)
    # JET
    if JET_TESTING_ENABLED
        @test_opt target_modules=(DiffusionModels,) marginal(diffusion_model, x_start, t)
        @test_call marginal(diffusion_model, x_start, t)
    end
end

# TODO: add some for sample sample_prior
# Definitely needs fixing

@testitem "Test get_drift_diffusion(::GaussianDiffusion, ...)" setup=[SharedTestSetup, GaussianDiffusionSetup] begin
    drift_fn, diffusion_fn = get_drift_diffusion(diffusion_model)
    @test drift_fn isa Function
    @test diffusion_fn isa Function
    @test drift_fn(x_start, nothing, 0.3) isa AbstractArray
    @test size(drift_fn(x_start, nothing, 0.3)) == size(x_start)
    @test diffusion_fn(x_start, nothing, 0.3) isa Union{AbstractArray, AbstractFloat}
    # Test broadcastable with?
    if JET_TESTING_ENABLED
        @test_opt target_modules=(DiffusionModels,) drift_fn(x_start, nothing, 0.3)
        @test_call drift_fn(x_start, nothing, 0.3)
        @test_opt target_modules=(DiffusionModels,) diffusion_fn(x_start, nothing, 0.3)
        @test_call diffusion_fn(x_start, nothing, 0.3)
    end
end

@testitem "Test get_diffeq_function(::GaussianDiffusion, ...)" setup=[SharedTestSetup, GaussianDiffusionSetup] begin
    sde_func = get_diffeq_function(diffusion_model)
    @test sde_func isa SDEFunction

    # JET
    if JET_TESTING_ENABLED
        @test_opt target_modules=(DiffusionModels,) get_diffeq_function(diffusion_model)
        @test_call broken=true get_diffeq_function(diffusion_model)
    end
end

@testitem "Test get_forward_diffeq(::GaussianDiffusion, ...)" setup=[SharedTestSetup, GaussianDiffusionSetup] begin
    prob = get_forward_diffeq(diffusion_model, x_start, (0.0, 1.0))
    @test prob isa SDEProblem
    @test prob.u0 == x_start
    @test prob.tspan == (0.0, 1.0)
    # TODO: Test that the prob can be solved

    # JET
    if JET_TESTING_ENABLED
        @test_opt broken=true get_forward_diffeq(diffusion_model, x_start, (0.0, 1.0))
        @test_call broken=true get_forward_diffeq(diffusion_model, x_start, (0.0, 1.0))
    end
end

@testitem "Test get_backward_diffeq(::GaussianDiffusion, ...)" setup=[SharedTestSetup, GaussianDiffusionSetup] begin
    score_fn = ScoreFunction((x,p,t) -> x, NoiseScoreParameterisation(schedule))
    prob = get_backward_diffeq(diffusion_model, score_fn, noise, (1.0, 0.0))
    @test prob isa SDEProblem
    @test prob.u0 == noise
    @test prob.tspan == (1.0, 0.0)
    # TODO: Test that the prob can be solved

    # Test x_0 parameterisation
    pred_x_start = (x,p,t) -> x_start
    score_fn = ScoreFunction(pred_x_start, StartScoreParameterisation(schedule))
    prob = get_backward_diffeq(diffusion_model, score_fn, noise, (1.0, 0.0))
    sol = solve(prob, EM(), dt=1e-4)
    @test sol isa RODESolution
    @test size(sol.u[end]) == size(x_start)
    # There will be an arror depending on dt, so we pick a fairly large atol
    @test sol.u[end] ≈ x_start atol=0.005

    # Run JET on solving
    if JET_TESTING_ENABLED
        @test_opt target_modules=(DiffusionModels,) solve(prob, EM(), dt=1e-3)
        @test_call solve(prob, EM(), dt=1e-3)
    end
end

@testitem "Test sample(::GaussianDiffusion, ...)" setup=[SharedTestSetup, GaussianDiffusionSetup] begin
    alg = EM()
    score_fn = ScoreFunction((x,p,t) -> x, NoiseScoreParameterisation(schedule))
    sol = sample(diffusion_model, score_fn, size(x_start), alg, dt=1e-3)
    @test sol isa RODESolution
    @test size(sol.u[end]) == size(x_start)

    # JET
    if JET_TESTING_ENABLED
        @test_opt target_modules=(DiffusionModels,) sample(diffusion_model, score_fn, size(x_start), alg, dt=1e-3)
        @test_call broken=true sample(diffusion_model, score_fn, size(x_start), alg, dt=1e-3)
    end
end

@testitem "Test denoising_loss_fn(::GaussianDiffusion, ...)" setup=[SharedTestSetup, GaussianDiffusionSetup] begin
    score_fn = ScoreFunction((x,p,t) -> x_start, StartScoreParameterisation(schedule))
    loss = denoising_loss_fn(diffusion_model, noise, score_fn)
    @test loss isa AbstractFloat
    @test loss >= 0.0

    # JET
    if JET_TESTING_ENABLED
        @test_opt target_modules=(DiffusionModels,) denoising_loss_fn(diffusion_model, noise, score_fn)
        @test_call denoising_loss_fn(diffusion_model, noise, score_fn)
    end

end
