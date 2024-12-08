```@meta
CollapsedDocStrings = true
```

# Gaussian Diffusion

## Score Parameterisations

### Abstract Types

```@docs
DiffusionModels.AbstractScoreParameterisation
```

### Types

```@docs
DiffusionModels.NoiseScoreParameterisation
DiffusionModels.VPredictScoreParameterisation
DiffusionModels.StartScoreParameterisation
```

### Methods

```@docs
DiffusionModels.get_target
```


## Score Function

```@docs
DiffusionModels.ScoreFunction
```


## Gaussian Diffusion Models

### Abstract Types

```@docs
DiffusionModels.AbstractGaussianDiffusion
```

### Types

```@docs
DiffusionModels.GaussianDiffusion
```

### Methods

```@docs
DiffusionModels.marginal
DiffusionModels.get_drift_diffusion
DiffusionModels.sample_prior
DiffusionModels.get_diffeq_function
DiffusionModels.get_forward_diffeq
DiffusionModels.get_backward_diffeq
DiffusionModels.sample
DiffusionModels.denoising_loss_fn
```
