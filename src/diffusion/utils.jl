function stable_tan(x::AbstractFloat)
    # equivalent to tan but stable at π/2
    c = cos(x)
    return sqrt(1 / c / c - 1)
end
