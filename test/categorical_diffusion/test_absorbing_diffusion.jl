
@testsnippet AbsorbingDiffusionSetup begin
    schedule = LinearMutualInfoSchedule{Float64}()
    diffusion_model = AbsorbingDiffusion(schedule, (1,), 0)
    x_start = rand(Xoshiro(0), 1:10, 10, 3)
    t = rand(Xoshiro(1), 3)
end

@testitem "Test marginal(::AbsorbingDiffusion, ...)" setup=[SharedTestSetup, AbsorbingDiffusionSetup] begin
    p_x_t = marginal(diffusion_model, x_start, t)
    @test p_x_t isa AbstractArray{<:DiscreteNonParametric}
    @test rand.(p_x_t) isa AbstractArray
    @test size(rand.(p_x_t)) == size(x_start)

    t_0 = zeros(size(t))
    p_x_0 = marginal(diffusion_model, x_start, t_0)
    @test all(rand.(p_x_0) .== x_start)
    t_1 = ones(size(t))
    p_x_1 = marginal(diffusion_model, x_start, t_1)
    @test all(rand.(p_x_1) .== diffusion_model.mask_token)
end
