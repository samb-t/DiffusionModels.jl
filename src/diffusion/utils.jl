# equivalent to tan but stable at π/2
stable_tan(x::AbstractFloat) = sqrt(cos(x)^-2 - 1)

reparameterise(dist::Normal{T}, z::T) where {T} = dist.μ .+ dist.σ .* z
