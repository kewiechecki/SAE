function outerenc(m=3)
    kern = (3,3)
    s = (2,2)
    θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))

    θ_mlp = Chain(Dense(12 => 6,relu),
                Dense(6 => m,relu))

    θ = Chain(θ_conv,
              x->reshape(x,12,:),
              θ_mlp)
    return θ
end

function outerclassifier(m=3)
    π = Chain(Dense(m => 5,relu),
                    Dense(5 => 10,relu),
                    softmax)
    return π
end

function outerdec(m=3)
    kern = (3,3)
    s = (2,2)
    ϕ_mlp = Chain(Dense(m => 6,relu),
                  Dense(6 => 12,relu))
    ϕ_deconv = Chain(ConvTranspose((2,2),12 => 9,relu),
                   ConvTranspose((4,4),9 => 6,relu,stride=s),
                   ConvTranspose(kern,6 => 3,relu,stride=s),
                   ConvTranspose((4,4),3 => 1,relu,stride=s))
    ϕ = Chain(ϕ_mlp,
              x->reshape(x,1,1,12,:),
              ϕ_deconv)
    return ϕ
end

function outermodel()
    kern = (3,3)
    s = (2,2)
    θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))

    θ_mlp = Chain(Dense(12 => 6,relu),
                Dense(6 => m,relu))

    θ_outer = Chain(θ_conv,
                    x->reshape(x,12,:),
                    θ_mlp)

    π_outer = Chain(Dense(m => 5,relu),
                    Dense(5 => 10,relu),
                    softmax)

    M_outer = Chain(θ_outer,π_outer)
    return M_outer
end

function mnistloader(batchsize)
    dat = MNIST(split=:train)[:]
    target = onehotbatch(dat.targets,0:9)

    m_x,m_y,n = size(dat.features)
    X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)

    loader = Flux.DataLoader((X,target),
                            batchsize=batchsize,
                            shuffle=true)
    return loader
end

function loadouter(m=3,path="data/MNIST/")
    θ = outerenc(m) |> gpu
    π = outerclassifier(m) |> gpu
    ϕ = outerdec(m) |> gpu
    M = Chain(θ,π)

    state_M = JLD2.load(path*"outer/classifier/final.jld2","state")
    Flux.loadmodel!(M,state_M)
    state_ϕ = JLD2.load(path*"outer/encoder/final.jld2","state")
    Flux.loadmodel!(ϕ,state_ϕ)
    return θ,π,ϕ
end

function loadsae(m,d,path="data/MNIST/")
    sae = SAE(m,d) |> gpu
    state = JLD2.load(path*"inner/SAE/final.jld2","state")
    Flux.loadmodel!(sae,state)
    return sae
end

function loadpsae(m,d,path="data/MNIST/")
    sae = SAE(m,d)
    partitioner = Chain(Dense(m => k,relu))

    psae = PSAE(sae,partitioner) |> gpu
    state = JLD2.load(path*"inner/PSAE/final.jld2","state")
    Flux.loadmodel!(psae,state)
    return psae
end

