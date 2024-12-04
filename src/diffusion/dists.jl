
struct MdNormal{T<:Real, Mean<:AbstractArray, StdDev<:AbstractArray} <: AbstractMvNormal
    μ::Mean
    σ::StdDev
end

function MdNormal(μ::AbstractArray{T, N}, σ::AbstractArray{T, N}) where {T, N}
    size(μ) == size(σ) || throw(DimensionMismatch("The dimensions of μ and σ are inconsistent."))
    MdNormal{T, typeof(μ), typeof(σ)}(μ, σ)
end

function MdNormal(μ::AbstractArray{T, N}, σ::AbstractFloat) where {T, N}
    σ = ones_like(μ) .* σ
    MdNormal{T, typeof(μ), typeof(σ)}(μ, σ)
end

Statistics.mean(d::MdNormal) = d.μ
Statistics.var(d::MdNormal) = d.σ .^ 2
Statistics.std(d::MdNormal) = d.σ
Base.rand(d::MdNormal) = d.μ .+ d.σ .* randn!(similar(d.σ))
