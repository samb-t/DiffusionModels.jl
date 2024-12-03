# Look at Lux.jl tests and copy what they do
# E.g. Run @jet on function calls

# Test ScoreFunction outputs in the correct shape
# 1. Mock the model
# 2. Mock the parameterisation
# 2.1 Mock the schedule

# struct MockScoreParameterisation <: AbstractScoreParameterisation end
# marginal_std_coeff()

@testset "Test Score Parameterisations" begin
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
