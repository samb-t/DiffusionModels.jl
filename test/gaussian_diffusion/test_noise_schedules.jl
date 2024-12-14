
@testitem "VP Noise Schedule Tests" setup = [SharedTestSetup] begin
    @testset "Test $(nameof(Schedule)){$(DType)}" for Schedule in [
            CosineSchedule, LinearSchedule, LinearMutualInfoSchedule
        ],
        DType in [Float32, Float64]

        @test check_interface_implemented(AbstractGaussianNoiseSchedule, Schedule)

        schedule = Schedule{DType}()
        # also test with vectorisation
        t_vec = rand(Xoshiro(0), DType, 3)
        t = DType(0.3)

        # JET
        if JET_TESTING_ENABLED
            @test_opt target_modules = (DiffusionModels,) marginal_mean_coeff(schedule, t)
            @test_call marginal_mean_coeff(schedule, t)

            @test_opt target_modules = (DiffusionModels,) marginal_std_coeff(schedule, t)
            @test_call marginal_std_coeff(schedule, t)

            @test_opt target_modules = (DiffusionModels,) marginal_mean_coeff(schedule, t_vec)
            @test_call marginal_mean_coeff(schedule, t_vec)

            @test_opt target_modules = (DiffusionModels,) marginal_std_coeff(schedule, t_vec)
            @test_call marginal_std_coeff(schedule, t_vec)

            @test_opt target_modules = (DiffusionModels,) drift_coeff(schedule, t)
            @test_call drift_coeff(schedule, t)

            @test_opt target_modules = (DiffusionModels,) diffusion_coeff(schedule, t)
            @test_call diffusion_coeff(schedule, t)
        end

        # Test correctness
        @test marginal_mean_coeff(schedule, DType(0.0)) isa DType
        @test marginal_mean_coeff(schedule, DType(0.0)) ≈ 1.0 atol = 1e-6
        # NOTE: Linear schedule is known to be 0.006353 at time 1.0
        @test marginal_mean_coeff(schedule, DType(1.0)) ≈ 0.0 atol = 1e-2

        @test marginal_std_coeff(schedule, DType(0.0)) isa DType

        # Test the variance preserving property
        mean = marginal_mean_coeff(schedule, t)
        std = marginal_std_coeff(schedule, t)
        @test mean^2 + std^2 ≈ 1.0 atol = 1e-6

        @test drift_coeff(schedule, DType(0.0)) isa DType
        @test diffusion_coeff(schedule, DType(0.0)) isa DType

        @test size(marginal_mean_coeff(schedule, t_vec)) == size(t_vec)
        @test size(marginal_std_coeff(schedule, t_vec)) == size(t_vec)

        # TODO: test dtype of vector calls too
    end
end
