using MLDatasets, JLD2, CSV

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

function outerclassifier(m=3,k=10)
    π = Chain(Dense(m => 5,relu),
                    Dense(5 => k,relu),
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

function trainouter(m::Integer,
                    loader::Flux.DataLoader,
                    opt::Flux.Optimiser,
                    epochs::Integer,
                    path)
    θ = outerenc(m) |> gpu
    ϕ = outerdec(m) |> gpu
    π = outerclassifier(m) |> gpu
    outerclas = Chain(θ,π)

    L_π = train!(outerclas,
                 loader,opt,epochs,logitcrossentropy;
                 savecheckpts=true,
                 path=path*"/classifier/");
    L_ϕ = train!(ϕ,
                 loader,opt,epochs,Flux.mse;
                 prefn=outerclas[1],
                 ignoreY=true,
                 savecheckpts=true,
                 path=path*"/encoder/");

    return θ,π,ϕ,L_π,L_ϕ
end

function trainsae(m::Integer,
                  d::Integer,
                  θ,π,ϕ,
                  α::AbstractFloat,
                  dat::Flux.DataLoader,
                  opt::Flux.Optimiser,
                  epochs::Integer,
                  path)
    sae = SAE(m,d) |> gpu
    L_classifier = train!(sae,Chain(θ,π),α,dat,opt,epochs,
                          logitcrossentropy;
                          path=path*"/classifier/SAE/L1_L2/")

    sae_L2 = SAE(m,d) |> gpu
    L2_classifier = train!(sae_L2,dat,opt,epochs,
                           logitcrossentropy;
                           prefn=θ,postfn=π,
                           ignoreY=true, savecheckpts=true,
                           path=path*"/classifier/SAE/L2/")

    sae_enc = SAE(m,d) |> gpu
    L_encoder = train!(sae_enc,Chain(θ,ϕ),
                       α,dat,opt,epochs,
                       Flux.mse;
                       path=path*"/encoder/SAE/L1_L2/")

    sae_enc_L2 = SAE(m,d) |> gpu
    L2_encoder = train!(sae_enc_L2,
                        dat,opt,epochs,
                        Flux.mse;
                        prefn=θ,postfn=ϕ,
                        ignoreY=true, savecheckpts=true,
                        path=path*"/encoder/SAE/L2/")
    return sae,sae_L2,sae_enc,sae_enc_L2
end

function trainpsae(m::Integer,
                   d::Integer,
                   k::Integer,
                   θ,π,ϕ,
                   α::AbstractFloat,
                   dat::Flux.DataLoader,
                   opt::Flux.Optimiser,
                   epochs::Integer,
                   path)
    psae = PSAE(m,d,k) |> gpu
    L_classifier = train!(psae,Chain(θ,π),α,dat,opt,epochs,
                          logitcrossentropy;
                          path=path*"/classifier/PSAE/L1_L2/")

    psae_L2 = PSAE(m,d,k) |> gpu
    L2_classifier = train!(psae_L2,dat,opt,epochs,
                           logitcrossentropy;
                           prefn=θ,postfn=π,
                           ignoreY=true, savecheckpts=true,
                           path=path*"/classifier/PSAE/L2/")

    psae_enc = PSAE(m,d,k) |> gpu
    L_encoder = train!(psae_enc,Chain(θ,ϕ),
                       α,dat,opt,epochs,
                       Flux.mse;
                       path=path*"/encoder/PSAE/L1_L2/")

    psae_enc_L2 = PSAE(m,d,k) |> gpu
    L2_encoder = train!(psae_enc_L2,
                        dat,opt,epochs,
                        Flux.mse;
                        prefn=θ,postfn=ϕ,
                        ignoreY=true, savecheckpts=true,
                        path=path*"/encoder/PSAE/L2/")
    return psae,psae_L2,psae_enc,psae_enc_L2
end



function mnistloader(batchsize::Integer)
    dat = MNIST(split=:train)[:]
    target = onehotbatch(dat.targets,0:9)

    m_x,m_y,n = size(dat.features)
    X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)

    loader = Flux.DataLoader((X,target),
                            batchsize=batchsize,
                            shuffle=true)
    return loader
end

function loader(dat,batchsize::Integer)
    X = dat(split=:train)[:]
    target = onehotbatch(X.targets,range(extrema(X.targets)...))
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

    state_M = JLD2.load(path*"/classifier/final.jld2","state")
    Flux.loadmodel!(M,state_M)
    state_ϕ = JLD2.load(path*"/encoder/final.jld2","state")
    Flux.loadmodel!(ϕ,state_ϕ)

    L_π = CSV.File(path*"/classifier/loss.csv").Column1
    L_ϕ = CSV.File(path*"/encoder/loss.csv").Column1
    return θ,π,ϕ,L_π,L_ϕ
end

function loadsae(m,d,path="data/MNIST/inner/classifier/SAE/L1_L2/")
    sae = SAE(m,d) |> gpu
    state = JLD2.load(path*"/final.jld2","state")
    L = CSV.File(path*"/loss.csv").Column1
    Flux.loadmodel!(sae,state)
    return sae,L
end

function loadpsae(m,d,k,path="data/MNIST/inner/classifier/PSAE/L1_L2/")
    psae = PSAE(m,d,k) |> gpu
    partitioner = Chain(Dense(m => k,relu))

    psae = PSAE(sae,partitioner) |> gpu
    state = JLD2.load(path*"/final.jld2","state")
    Flux.loadmodel!(psae,state)
    L = CSV.File(path*"/loss.csv").Column1
    return psae,L
end

