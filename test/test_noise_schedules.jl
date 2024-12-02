
@testset "VP Noise Schedule Tests" begin
    @testset "Test $(nameof(Schedule))" for Schedule in [CosineSchedule, LinearSchedule]
        @test check_interface_implemented(AbstractGaussianNoiseSchedule, CosineSchedule)

        schedule = Schedule()

        # JET
        if JET_TESTING_ENABLED
            @test_call marginal_mean_coeff(schedule, 0.3)
            @test_opt target_modules=(DiffusionModels,) marginal_mean_coeff(schedule, 0.3)
        end

        # Test correctness
        @test marginal_mean_coeff(schedule, 0.0) isa AbstractFloat
        @test marginal_mean_coeff(schedule, 0.0) ≈ 1.0 atol=1e-6
        # NOTE: Linear schedule is known to be 0.006353 at time 1.0
        @test marginal_mean_coeff(schedule, 1.0) ≈ 0.0 atol=1e-2
    end
end
