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
        schedule = CosineSchedule()
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
        schedule = CosineSchedule()

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


@testitem "Test VPDiffusion" setup=[SharedTestSetup] begin
    schedule = CosineSchedule()

    diffusion_model = VPDiffusion(schedule)
    x_start = randn(Xoshiro(0), 10, 3)
    t = rand(Xoshiro(1), 3)

    @testset "Test marginal(...)" begin
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

    @testset "Test get_drift_diffusion(...)" begin
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

    @testset "Test get_diffeq_function(...)" begin
        sde_func = get_diffeq_function(diffusion_model)
        @test sde_func isa SDEFunction
    end



end
