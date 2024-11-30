using DiffusionModels
using Lux
using Random
using Optimisers
include("../toy_datasets.jl")


function sample_data(d::VPDiffusion, x_start::AbstractArray)
    t = rand(dtype(x_start), size(x_start, 2))
    p_xt = marginal(d, x_start, t)
    x_t = rand(p_xt)
    score = ...
    # TODO: also concat x_t and t ready for network input
    return x_t, score
end

# TODO: Move this somewhere shared + useful
# HOWEVER!! Time doesn't necessarily have to be the same T and N as the input so...
# const ModelInputType = Tuple{AbstractArray{T,N}, AbstractArray{T,N}} where {T, N}
# Also, could time potentially be a float? So maybe it's best to not define the
# type at all
# const ModelInputType = Tuple{AbstractArray, AbstractArray} where {T, N}
const ModelInputType = Tuple{AbstractArray, Union{AbstractArray, Float}}

data_fn = sample_checkerboard
dtype = Float32
input_dim = 2
hidden_dim = 32
training_iters = 1000
num_data_points = 10_000
batch_size = 128
epochs = 100

# Fine, or should be different functions?
ConcatenateImageTime() = WrappedFunction() do inputs#::ModelInputType
    x, t = inputs
    if typeof(t) <: AbstractFloat
        t = ones(eltype(x), 1, size(x, 2))
    end
    return cat(x, t, dims=1)
end

# Cat() = WrappedFunction(x -> cat(x..., dims=1))
model = Chain(
    ConcatenateImageTime(),
    Dense(input_dim + 1 => hidden_dim, relu),
    Dense(hidden_dim => hidden_dim, relu),
    Dense(hidden_dim => hidden_dim, relu),
    Dense(hidden_dim => input_dim),
)
opt = Adam(1.0f-4)

# TODO: Do this in the train function so it's not global
ps, st = Lux.setup(Xoshiro(0), model)
# Option 1
wrapped_model(x, p, t) = model((x, t), p, st)
# Option 2
# stateful_model =  Lux.Experimental.StatefulLuxLayer(model, ps, st)
# wrapped_stateful_model(x, p, t) = stateful_model((x, t))

schedule = CosineSchedule{dtype}()
parameterisation = NoiseScoreParameterisation(schedule)
score_fn = ScoreFunction(wrapped_model, parameterisation)
diffusion = VPDiffusion(schedule, score_fn)

loss(x) = denoising_loss_fn(diffusion, x; p=ps)

function train(model; rng=Xoshiro(0))

    train_state = Lux.Experimental.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()

    for iter in 1:training_iters
        X = data_fn(batch_size)
        x_t, score = sample_data(diffusion, X)

        (gs, _, _, train_state) = Lux.Experimental.compute_gradients(
            vjp_rule,
            loss,
            (x_t, score),
            train_state
        )
        train_state = Lux.Experimental.apply_gradients!(train_state, gs)

    end
end
