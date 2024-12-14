```@meta
CollapsedDocStrings = true
```

# Noise Schedules

## Schedules

```@docs
DiffusionModels.CosineSchedule
DiffusionModels.LinearSchedule
DiffusionModels.LinearMutualInfoSchedule
```

## Functions

Some functions

```@docs
DiffusionModels.marginal_mean_coeff
DiffusionModels.marginal_std_coeff
DiffusionModels.drift_coeff
DiffusionModels.diffusion_coeff
DiffusionModels.log_snr
DiffusionModels.beta
```

## Abstract Types

```@docs
DiffusionModels.AbstractSchedule
DiffusionModels.AbstractNoiseSchedule
DiffusionModels.AbstractGaussianNoiseSchedule
DiffusionModels.VPNoiseSchedule
```
