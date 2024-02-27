## Formatted to work with Literate.jl in the future

using DiffusionModels
using SimpleChains
include("../utils.jl")


data_fn = sample_checkerboard
dtype = Float32
input_dim = 2
hidden_dim = 32
training_iters = 1000
num_data_points = 10_000
batch_size = 128


score_fn = SimpleChain(
    static(input_dim + 1),
    TurboDense{true}(relu, hidden_dim),
    TurboDense{true}(relu, hidden_dim),
    TurboDense{true}(relu, hidden_dim),
    TurboDense{true}(identity, input_dim),
)
opt = SimpleChains.ADAM()


schedule = CosineSchedule{dtype}()
diffusion = VPDiffusion(schedule, score_fn)


p = SimpleChains.init_params(score_fn)
G = SimpleChains.alloc_threaded_grad(score_fn);

X = sample_checkerboard(num_data_points)
# TODO:
# 1. Sample a load of time steps,
# 2. then calculate the marginals for all of these
# NOTE: It probably makes sense to not allocate this all in advance though?
train_batched!(G, p, model, X, opt, training_iters, batch_size=batch_size)
