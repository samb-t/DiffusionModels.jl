using Images
using Random
using Flux: unsqueeze
using DifferentialEquations

function plot_diffusion_marginal(
    d::AbstractDiffusion, 
    x_start::AbstractArray; 
    num_steps::Int=10,
)
    if ndims(x_start) == 3
        x_start = unsqueeze(x_start, dims=4)
    elseif ndims(x_start) != 4
        throw(ErrorException("Expected an image of either 3 or 4 dims."))
    elseif size(x_start, 4) != 1
        throw(ErrorException("Expected batch dimension to be 1."))
    end

    t = collect(range(0, 1, length=num_steps))
    imgs = marginal(d, x_start, t)
    # imgs = rand(dist)
    imgs = @. (imgs + 1) / 2
    imgs = clamp.(imgs, 0, 1)
    imgs = cat([imgs[:,:,:,i] for i in 1:size(imgs, 4)]..., dims=2)

    colorview(RGB, permutedims(imgs[:,:,1:3,1], (3,1,2)))
end

function plot_diffusion_forward_sde_transition(
    d::AbstractGaussianDiffusion, 
    x_start::AbstractArray; 
    num_steps::Int=10,
    dt=0.1,
)
    if ndims(x_start) == 3
        x_start = unsqueeze(x_start, dims=4)
    elseif ndims(x_start) != 4
        throw(ErrorException("Expected an image of either 3 or 4 dims."))
    elseif size(x_start, 4) != 1
        throw(ErrorException("Expected batch dimension to be 1."))
    end

    t = 0.0
    x_t = x_start
    all_xs = [x_t]
    for _ in 1:(num_steps-1)
        x_t = transition(d, x_t, (t, t+(1/num_steps)), EM(), dt=dt)
        push!(all_xs, x_t)
        t = t + (1/num_steps)
    end
    imgs = cat(all_xs..., dims=2)
    imgs = @. (imgs + 1) / 2
    imgs = clamp.(imgs, 0, 1)
    colorview(RGB, permutedims(imgs[:,:,1:3,1], (3,1,2)))
end

function plot_diffusion_forward_ode_transition(
    d::AbstractGaussianDiffusion, 
    x_start::AbstractArray; 
    num_steps::Int=10,
)
    if ndims(x_start) == 3
        x_start = unsqueeze(x_start, dims=4)
    elseif ndims(x_start) != 4
        throw(ErrorException("Expected an image of either 3 or 4 dims."))
    elseif size(x_start, 4) != 1
        throw(ErrorException("Expected batch dimension to be 1."))
    end

    t = 0.0
    dt = 1 / num_steps
    x_t = x_start
    all_xs = [x_t]
    for _ in 1:num_steps
        x_t = transition(d, x_t, (t, t+dt), Euler(), dt=dt)
        push!(all_xs, x_t)
        t = t + dt
    end
    imgs = cat(all_xs..., dims=2)
    imgs = @. (imgs + 1) / 2
    imgs = clamp.(imgs, 0, 1)
    colorview(RGB, permutedims(imgs[:,:,:,1], (3,1,2)))
end

function plot_diffusion_backward_sde_transition(
    d::AbstractGaussianDiffusion, 
    x_start::AbstractArray; 
    num_steps::Int=10,
)
    if ndims(x_start) == 3
        x_start = unsqueeze(x_start, dims=4)
    elseif ndims(x_start) != 4
        throw(ErrorException("Expected an image of either 3 or 4 dims."))
    elseif size(x_start, 4) != 1
        throw(ErrorException("Expected batch dimension to be 1."))
    end

    t = 1.0
    dt = 1 / num_steps
    x_t = x_start
    all_xs = [x_t]
    for _ in 1:num_steps
        dist = transition(d, x_t, (t, t-dt), EM(), dt=dt)
        x_t = rand(dist)
        push!(all_xs, x_t)
        t = t - dt
    end
    imgs = cat(all_xs..., dims=2)
    imgs = @. (imgs + 1) / 2
    imgs = clamp.(imgs, 0, 1)
    colorview(RGB, permutedims(imgs[:,:,:,1], (3,1,2)))
end

# function plot_diffusion_q_transition(
#     d::AbstractGaussianDiffusion, 
#     x_start::AbstractArray; 
#     num_steps::Int=10,
# )
#     if ndims(x_start) == 3
#         x_start = unsqueeze(x_start, dims=4)
#     elseif ndims(x_start) != 4
#         throw(ErrorException("Expected an image of either 3 or 4 dims."))
#     elseif size(x_start, 4) != 1
#         throw(ErrorException("Expected batch dimension to be 1."))
#     end

#     t = [0.0]
#     dt = [1 / num_steps]
#     x_t = x_start
#     all_xs = [x_t]
#     for _ in 1:num_steps
#         dist = q_transition(d, x_t, t, dt)
#         x_t = rand(dist)
#         push!(all_xs, x_t)
#         t = t .+ dt
#     end
#     imgs = cat(all_xs..., dims=2)
#     imgs = @. (imgs + 1) / 2
#     imgs = clamp.(imgs, 0, 1)
#     colorview(RGB, permutedims(imgs[:,:,:,1], (3,1,2)))
# end

# function plot_diffusion_q_posterior(
#     d::AbstractGaussianDiffusion, 
#     x_start::AbstractArray; 
#     num_steps::Int=10,
# )
#     if ndims(x_start) == 3
#         x_start = unsqueeze(x_start, dims=4)
#     elseif ndims(x_start) != 4
#         throw(ErrorException("Expected an image of either 3 or 4 dims."))
#     elseif size(x_start, 4) != 1
#         throw(ErrorException("Expected batch dimension to be 1."))
#     end

#     x_t = randn!(similar(x_start))
#     t = [1.0]
#     dt = [1 / num_steps]
#     all_xs = [x_t]
#     for _ in 1:num_steps
#         dist = q_posterior(d, x_t, t, x_start, dt)
#         x_t = rand(dist)
#         push!(all_xs, x_t)
#         t .-= dt
#     end
#     imgs = cat(all_xs..., dims=2)
#     imgs = @. (imgs + 1) / 2
#     imgs = clamp.(imgs, 0, 1)
#     colorview(RGB, permutedims(imgs[:,:,:,1], (3,1,2)))
# end

# function plot_diffusion_q_transition_solver(
#     d::AbstractGaussianDiffusion, 
#     x_start::AbstractArray; 
#     num_steps::Int=10,
# )
#     if ndims(x_start) == 3
#         x_start = unsqueeze(x_start, dims=4)
#     elseif ndims(x_start) != 4
#         throw(ErrorException("Expected an image of either 3 or 4 dims."))
#     elseif size(x_start, 4) != 1
#         throw(ErrorException("Expected batch dimension to be 1."))
#     end

#     t = 0.0
#     dt = 1 / num_steps
#     x_t = x_start
#     all_xs = [x_t]
#     for _ in 1:num_steps
#         x_t = q_transition_solver(d, x_t, t, dt)
#         push!(all_xs, x_t)
#         t = t .+ dt
#     end
#     imgs = cat(all_xs..., dims=2)
#     imgs = @. (imgs + 1) / 2
#     imgs = clamp.(imgs, 0, 1)
#     colorview(RGB, permutedims(imgs[:,:,:,1], (3,1,2)))
# end