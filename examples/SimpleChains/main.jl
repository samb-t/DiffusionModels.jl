## TODO: Format to work with Literate.jl in the future

using DiffusionModels
using SimpleChains
# using NNlib # for gelu
include("../toy_datasets.jl")

function sample_epoch_data(d::VPDiffusion, x_start::AbstractArray)
    t = rand(dtype(x_start), size(x_start, 2))
    p_xt = marginal(d, x_start, t)
    x_t = rand(p_xt)
    score = ...
    # TODO: also concat x_t and t ready for network input
    return x_t, score
end


data_fn = sample_checkerboard
dtype = Float32
input_dim = 2
hidden_dim = 32
training_iters = 1000
num_data_points = 10_000
batch_size = 128
epochs = 100


X = data_fn(num_data_points)
model = SimpleChain(
    static(input_dim + 1),
    TurboDense{true}(relu, hidden_dim),
    TurboDense{true}(relu, hidden_dim),
    TurboDense{true}(relu, hidden_dim),
    TurboDense{true}(identity, input_dim),
)
opt = SimpleChains.ADAM()

schedule = CosineSchedule{dtype}()
parameterisation = NoiseScoreParameterisation(schedule)
score_fn = ScoreFunction(model, parameterisation)
diffusion = VPDiffusion(schedule, score_fn)

loss(x,y) = denoising_loss_fn(diffusion, ...)

params = SimpleChains.init_params(model)
gradient_buffer = SimpleChains.alloc_threaded_grad(model)


for _ in 1:epochs
    x_t, score = sample_epoch_data(diffusion, X)
    train_batched!(gradient_buffer, params, model, X, opt, training_iters, batch_size=batch_size)
end


















# For now maybe just fix the reparameterisation to always be to the true score
# TODO: Define type for score parameterisations which will convert an arbitrary function
#       to the score. As well as providing an estimate of the score for training purposes
# Or perhaps just have a function "target" which is for SDEs and Flow matching etc. it
# could take an optional argument of the type of target in case in the future we introduce more
# But that target type should affect inference time too...
# So yeah, stick to a single type for each for now?

# Maybe create a new score_fn struct that stores the model and the parameterisation type? E.g.
abstract type AbstractScoreParameterisation end

struct NoiseScoreParameterisation <: AbstractScoreParameterisation end
struct StartScoreParameterisation <: AbstractScoreParameterisation end

struct ScoreFn{T, P} where {P<:AbstractScoreParameterisation}
    model::T
    parameterisation::P
end

function target(
    ::NoiseScoreParameterisation,
    d::AbstractGaussianDiffusion,
    x_start::AbstractArray,
    x_t::AbstractArray,
    t::AbstractVector,
)
    ...
end

# and the diffusion model can take as score function either this struct or any callable function

# sample
