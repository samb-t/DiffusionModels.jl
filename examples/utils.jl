function sample_checkerboard(batch_size::Integer)
    x1 = rand(batch_size) .* 4 .- 2
    x2_ = rand(batch_size) .- rand(0:1, batch_size) .* 2
    x2 = x2_ .+ (floor.(x1) )
    return vcat(x1', x2') .* 2
end
