@doc raw"""
    AbstractJumpSchedule

Abstract type for jump schedules. TODO: write docstring.
"""
abstract type AbstractJumpSchedule <: AbstractSchedule end

@required AbstractJumpSchedule begin
    jump_rate(::AbstractJumpSchedule, ::AbstractFloat)
    jump_rate_integral(::AbstractJumpSchedule, ::AbstractFloat)
end

function jump_rate(s::AbstractJumpSchedule, t::AbstractVector)
    return jump_rate.(Ref(s), t)
end

function jump_rate_integral(s::AbstractJumpSchedule, t::AbstractVector)
    return jump_rate_integral.(Ref(s), t)
end


@doc raw"""
    ConstantJumpSchedule(max_dim::Int; minimum_dims::Int=1, std_mult::AbstractFloat=0.7)

A jump schedule where the rate function is constant with respect to time. [campbell2024trans](@cite)
"""
@kwdef struct ConstantJumpSchedule{T<:AbstractFloat} <: AbstractJumpSchedule
    max_dim::Int
    minimum_dims::Int
    std_mult::T = 0.7
end

@doc raw"""
    jump_rate(schedule::ConstantJumpSchedule, t::AbstractFloat)

Return the jump rate at time `t`.
"""
function jump_rate(s::ConstantJumpSchedule{T}, ::T) where {T<:AbstractFloat}
    c = T(s.max_dim - s.minimum_dims)
    return (2 * c + s.std_mult^2 + sqrt((s.std_mult^2 + 2 * c)^2 - 4 * c^2)) / 2
end

beta(s::ConstantJumpSchedule, t::T) where {T<:AbstractFloat} = jump_rate(s, t)

# Call this `expected_cumulative_jumps`?
@doc raw"""
    jump_rate_integral(schedule::ConstantJumpSchedule, t::AbstractFloat)

Return the expected number of jumps up to time `t`.
"""
function jump_rate_integral(s::ConstantJumpSchedule{T}, t::T) where {T<:AbstractFloat}
    return jump_rate(s, t) * t
end

ScheduleVariabilityTrait(::Type{ConstantJumpSchedule}) = ConstantRateSchedule()
