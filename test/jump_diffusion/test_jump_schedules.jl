@testitem "JumpSchedule Tests" setup = [SharedTestSetup] begin
    @testset "Test $(nameof(Schedule)){$(DType)}" for Schedule in [
            ConstantJumpSchedule
        ],
        DType in [Float32, Float64]

        @test check_interface_implemented(AbstractJumpSchedule, Schedule)

        schedule = Schedule{DType}(max_dim=10, minimum_dims=1)
        # also test with vectorisation
        t_vec = rand(Xoshiro(0), DType, 3)
        t = DType(0.3)

        # JET
        if JET_TESTING_ENABLED
            @test_opt target_modules = (DiffusionModels,) jump_rate(schedule, t)
            @test_call jump_rate(schedule, t)

            @test_opt target_modules = (DiffusionModels,) jump_rate_integral(schedule, t)
            @test_call jump_rate_integral(schedule, t)

            @test_opt target_modules = (DiffusionModels,) jump_rate(schedule, t_vec)
            @test_call jump_rate(schedule, t_vec)

            @test_opt target_modules = (DiffusionModels,) jump_rate_integral(schedule, t_vec)
            @test_call jump_rate_integral(schedule, t_vec)
        end

        # Test correctness
        @test jump_rate(schedule, DType(0.0)) isa DType
        @test jump_rate(schedule, DType(0.0)) ≈ jump_rate(schedule, t) atol = 1e-6

        @test jump_rate_integral(schedule, DType(0.0)) isa DType
        @test jump_rate_integral(schedule, DType(0.0)) ≈ 0.0 atol = 1e-6
        # TODO: Check that it really should be >=.
        @test jump_rate_integral(schedule, DType(1.0)) >= DType(schedule.max_dim - schedule.minimum_dims)

        @test size(jump_rate(schedule, t_vec)) == size(t_vec)
        @test size(jump_rate_integral(schedule, t_vec)) == size(t_vec)

        # TODO: test dtype of vector calls too
    end
end
