# Look at Lux.jl tests and copy what they do
# E.g. Run @jet on function calls

# Test ScoreFunction outputs in the correct shape
# 1. Mock the model
# 2. Mock the parameterisation
# 2.1 Mock the schedule

# struct MockScoreParameterisation <: AbstractScoreParameterisation end
# marginal_std_coeff()

@testset "Example Test Set" begin
    @test 1 == 1
end
