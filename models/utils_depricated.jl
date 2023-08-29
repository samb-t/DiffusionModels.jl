using Flux
using Base
using Printf
using Functors
using Setfield
using Setfield: setproperties

"""
    FunctionalModel(func, params)
    FunctionalModel(func; exclude=nothing, include=nothing)

Create complex layers functionally, avoiding using structs.

Why? Coming from pytorch I like gradually building up complex models and in Flux this 
would involve repeatedly adding fields to layer structs. This would be fine if Julia's
startup time wasn't so high or structs could be overwritten. This approach allows 
layers to be built in a similar manner but layers are stored as functions not structs. 

Example:
function ResBlock(dim)
    # define all the layers (in pytorch in __init__)
    layers = Chain(Dense(dim => dim, relu), Dense(dim => dim))
    # 
    FunctionalModel(x -> begin
        x + layers(x)
    end)
end
"""
# TODO: Sort out type of T e.g. Vector{Any} to make this function type stable
struct FunctionalModel{T}
    func::Function
    property_names::T
end

(f::FunctionalModel)(x...) = f.func(x...)

# TODO: Add checks here that include are actually correct for given f
# TODO: Decide, should params store getproperty or propertynames
function FunctionalModel(func; include=nothing)
    if !isnothing(include)
        property_names = include
    else
        property_names = propertynames(func)
    end
    FunctionalModel(func, property_names)
end

# Define explicitly to avoid having to call functor (which isn't a big deal tbf)
Flux.trainable(f::FunctionalModel) = (getproperty(f.func, w) for w in f.property_names)

function Functors.functor(::Type{<:FunctionalModel}, f)
    # Step 1: Extract all the properties
    p = [propname => getproperty(f.func, propname) for propname in f.property_names]
    p = (; p...)
    # Step 2: Update all the properties 
    function reconstruct(f_props)
        func_new = setproperties(f.func, f_props)
        return FunctionalModel(func_new, f.property_names)
    end
    return p, reconstruct
end

# TODO: Better printing so it doesn't show as a jumbled mess! 
#   E.g. allow a description to be passed into FunctionalModel
# # What it prints as in IJulia
# function Base.show(io::IO, ::MIME"test/plain", f::FunctionalModel)
#     if isnothing(f.description)
#         # TODO: How to print this properly???
#         return @printf(io, "FunctionalModel(%s, %s)", io.func, io.params)
#     else
#         return @printf(io, f.description)
#     end
# end

# Base.show(io::IO, f::FunctionalModel) = ""
