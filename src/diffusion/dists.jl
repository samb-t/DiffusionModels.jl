
struct MdNormal{T<:Real, Mean<:AbstractArray, StdDev<:AbstractArray} <: AbstractMvNormal
    mean::Mean
    std::StdDev
end

function MdNormal(mean::AbstractArray{T, N}, std::AbstractArray{T, N}) where {T, N}
    size(mean) == size(std) || throw(DimensionMismatch("The dimensions of mean and std are inconsistent."))
    MdNormal{T, typeof(mean), typeof(std)}(mean, std)
end

function MdNormal(mean::AbstractArray{T, N}, std::AbstractFloat) where {T, N}
    std = ones_like(mean) .* std
    MdNormal{T, typeof(mean), typeof(std)}(mean, std)
end

Statistics.mean(d::MdNormal) = d.mean
Statistics.var(d::MdNormal) = d.std .^ 2
Statistics.std(d::MdNormal) = d.std
Base.rand(d::MdNormal) = d.mean .+ d.std .* randn!(similar(d.std))
