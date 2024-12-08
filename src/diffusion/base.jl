abstract type AbstractDiffusion end

abstract type AbstractContinuousTimeDiffusion <: AbstractDiffusion end
abstract type AbstractDiscreteTimeDiffusion <: AbstractContinuousTimeDiffusion end

abstract type AbstractCategoricalDiffusion <: AbstractDiffusion end

abstract type AbstractModelParameterisation end
