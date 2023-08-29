# UNet and an Encoder

GELU() = x -> gelu.(x)

Upsample(dim) = ConvTranspose((4,4), dim => dim, stride=2, pad=1)
Downsample(dim) = Conv((4,4), dim => dim, stride=2, pad=1)

function Block(dim, dim_out; groups=8)
    proj = Conv((3,3), dim => dim_out, pad=1)
    norm = GroupNorm(dim_out, groups)
    act = GELU()

    FunctionalModel((x, scale_shift=nothing) -> begin
        x = proj(x)
        x = norm(x)

        if !isnothing(scale_shift)
            scale, shift = scale_shift
            x = x .* (scale .+ 1f0) .+ shift
        end

        x = act(x)
        return x
    end)
end

function ResnetBlock(dim, dim_out; time_emb_dim=nothing, groups=8)
    mlp = !isnothing(time_emb_dim) ? Chain(GELU(), Dense(time_emb_dim => dim_out * 2)) : nothing
    block1 = Block(dim, dim_out, groups=groups)
    block2 = Block(dim_out, dim_out, groups=groups)
    res_conv = dim != dim_out ? Conv((1,1), dim => dim_out) : x -> x
    
    FunctionalModel((x, time_emb=nothing) -> begin
        scale_shift = nothing
        if !isnothing(mlp) && !isnothing(time_emb)
            time_emb = mlp(time_emb)
            time_emb = unsqueeze(unsqueeze(time_emb, dims=1), dims=1)
            scale_shift = MLUtils.chunk(time_emb, 2, dims=3)
        end
        h = block1(x, scale_shift)
        h = block2(h)
        return h + res_conv(x)
    end)
end

# TODO: Do casting/gpu transfer better. 
function SinusoidalPosEmb(dim)
    x -> begin
        half_dim = div(dim, 2) - 1
        epty = convert.(Float32, similar(x, half_dim+1)) # This is so ugly, isn't there a better way?
        epty[:] = 0:half_dim
        emb = log(10000f0) / half_dim
        emb = exp.(epty .* -emb)
        emb = transpose(x) .* emb
        emb = vcat(sin.(emb), cos.(emb))
        emb
    end
end


struct Unet
    init_conv
    time_mlp
    downs
    mids
    ups
    final_conv
end

@functor Unet

function Unet(dim; init_dim=nothing, dim_mults=(1,2,3,4), channels=3, 
    with_time_emb=true, resnet_groups=8)
    init_dim = isnothing(init_dim) ? div(dim, 3) * 2 : init_dim
    init_conv = Conv((7,7), channels => init_dim, pad=3)

    dims = [init_dim, map(m -> dim * m, dim_mults)...]
    in_out = [zip(dims[1:end-1], dims[2:end])...]

    if !isnothing(with_time_emb)
        time_dim = dim * 4
        time_mlp = Chain(
            SinusoidalPosEmb(dim),
            Dense(dim => time_dim, gelu),
            Dense(time_dim => time_dim)
        )
    else
        time_dim = nothing
        time_mlp = nothing
    end

    downs = []
    ups = []
    num_resolutions = length(in_out)

    for (ind, (dim_in, dim_out)) in enumerate(in_out)
        islast = ind >= num_resolutions

        push!(downs, [
            ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_groups),
            ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_groups),
            !islast ? Downsample(dim_out) : x -> x
        ])
    end

    mid_dim = dims[end]
    mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_groups)
    mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_groups)
    mids = [mid_block1, mid_block2]

    for (ind, (dim_in, dim_out)) in enumerate(reverse(in_out[2:end]))
        islast = ind >= num_resolutions
        push!(ups, [
            ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, groups=resnet_groups),
            ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_groups),
            !islast ? Upsample(dim_in) : x -> x
        ])
    end

    final_conv = Chain(
        ResnetBlock(dim, dim, time_emb_dim=time_dim, groups=resnet_groups),
        Conv((1,1), dim => channels)
    )

    Unet(init_conv, time_mlp, downs, mids, ups, final_conv)
end

function (u::Unet)(x, time::AbstractArray; p=nothing)
    x = u.init_conv(x)
    t = !isnothing(u.time_mlp) ? u.time_mlp(time) : nothing

    h = []
    for (block1, block2, downsample) in u.downs
        x = block1(x, t)
        x = block2(x, t)
        push!(h, x)
        x = downsample(x)
    end

    for block in u.mids
        x = block(x, t)
    end

    for (block1, block2, upsample) in u.ups
        x = cat(x, pop!(h), dims=3)
        x = block1(x, t)
        x = block2(x, t)
        x = upsample(x)
    end

    return u.final_conv(x)
end

# TODO: Do casting/gpu transfer better
function (u::Unet)(x, time::AbstractFloat; p=nothing)
    t = fill!(similar(x, size(x, 4)), time)
    return u(x, t; p=p)
end



# TODO: Fix bug "ERROR: too many parameters for type"
# function Unet(dim; init_dim=nothing, dim_mults=(1,2,3,4), channels=3, 
#               with_time_emb=true, resnet_groups=8)
    
#     init_dim = isnothing(init_dim) ? div(dim, 3) * 2 : init_dim
#     init_conv = Conv((7,7), channels => init_dim, pad=3)

#     dims = [init_dim, map(m -> dim * m, dim_mults)...]
#     in_out = [zip(dims[1:end-1], dims[2:end])...]

#     if !isnothing(with_time_emb)
#         time_dim = dim * 4
#         time_mlp = Chain(
#             SinusoidalPosEmb(dim),
#             Dense(dim => time_dim, gelu),
#             Dense(time_dim => time_dim)
#         )
#     else
#         time_dim = nothing
#         time_mlp = nothing
#     end

#     downs = []
#     ups = []
#     num_resolutions = length(in_out)

#     for (ind, (dim_in, dim_out)) in enumerate(in_out)
#         islast = ind >= num_resolutions

#         push!(downs, [
#             ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_groups),
#             ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_groups),
#             !islast ? Downsample(dim_out) : x -> x
#         ])
#     end

#     mid_dim = dims[end]
#     mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_groups)
#     mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_groups)

#     for (ind, (dim_in, dim_out)) in enumerate(reverse(in_out[2:end]))
#         islast = ind >= num_resolutions
#         push!(ups, [
#             ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, groups=resnet_groups),
#             ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_groups),
#             !islast ? Upsample(dim_in) : x -> x
#         ])
#     end

#     final_conv = Chain(
#         ResnetBlock(dim, dim, time_emb_dim=time_dim, groups=resnet_groups),
#         Conv((1,1), dim => channels)
#     )

#     FunctionalModel((x, time) -> begin
#         x = init_conv(x)
#         t = !isnothing(time_mlp) ? time_mlp(time) : nothing

#         h = []
#         for (block1, block2, downsample) in downs
#             x = block1(x, t)
#             x = block2(x, t)
#             push!(h, x)
#             x = downsample(x)
#         end

#         x = mid_block1(x, t)
#         x = mid_block2(x, t)

#         for (block1, block2, upsample) in ups
#             x = cat(x, pop!(h), dims=3)
#             x = block1(x, t)
#             x = block2(x, t)
#             x = upsample(x)
#         end

#         return final_conv(x)
#     end)
# end