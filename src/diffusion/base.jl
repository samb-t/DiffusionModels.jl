abstract type AbstractDiffusion end

abstract type AbstractContinuousTimeDiffusion <: AbstractDiffusion end
abstract type AbstractDiscreteTimeDiffusion <: AbstractContinuousTimeDiffusion end

abstract type AbstractGaussianDiffusion <: AbstractDiffusion end
abstract type AbstractCategoricalDiffusion <: AbstractDiffusion end
