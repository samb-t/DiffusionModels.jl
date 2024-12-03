
@testset "VP Noise Schedule Tests" begin
    @testset "Test $(nameof(Schedule))" for Schedule in [CosineSchedule, LinearSchedule, LinearMutualInfoSchedule]
        @test check_interface_implemented(AbstractGaussianNoiseSchedule, CosineSchedule)

        schedule = Schedule()
        # also test with vectorisation
        t = rand(Xoshiro(0), 3)

        # JET
        if JET_TESTING_ENABLED
            @test_opt target_modules=(DiffusionModels,) marginal_mean_coeff(schedule, 0.3)
            @test_call marginal_mean_coeff(schedule, 0.3)

            @test_opt target_modules=(DiffusionModels,) marginal_std_coeff(schedule, 0.3)
            @test_call marginal_std_coeff(schedule, 0.3)

            @test_opt target_modules=(DiffusionModels,) marginal_mean_coeff(schedule, t)
            @test_call marginal_mean_coeff(schedule, t)

            @test_opt target_modules=(DiffusionModels,) marginal_std_coeff(schedule, t)
            @test_call marginal_std_coeff(schedule, t)

            @test_opt target_modules=(DiffusionModels,) drift_coeff(schedule, 0.3)
            @test_call drift_coeff(schedule, 0.3)

            @test_opt target_modules=(DiffusionModels,) diffusion_coeff(schedule, 0.3)
            @test_call diffusion_coeff(schedule, 0.3)

        end

        # Test correctness
        @test marginal_mean_coeff(schedule, 0.0) isa AbstractFloat
        @test marginal_mean_coeff(schedule, 0.0) ≈ 1.0 atol=1e-6
        # NOTE: Linear schedule is known to be 0.006353 at time 1.0
        @test marginal_mean_coeff(schedule, 1.0) ≈ 0.0 atol=1e-2

        @test marginal_std_coeff(schedule, 0.0) isa AbstractFloat
        @test drift_coeff(schedule, 0.0) isa AbstractFloat
        @test diffusion_coeff(schedule, 0.0) isa AbstractFloat

        @test size(marginal_mean_coeff(schedule, t)) == size(t)
        @test size(marginal_std_coeff(schedule, t)) == size(t)

    end
end
