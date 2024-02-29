include("SAE.jl")
include("PSAE.jl")
include("trainingfns.jl")
include("models.jl")
include("auxfns.jl")

using StatsPlots
using Flux: logitcrossentropy

# where to write the toy model
path = "data/MNIST/"

epochs = 100
batchsize=512

η = 0.001
λ = 0.001
α = 0.001

opt = Flux.AdamW(η)
opt_wd = Flux.Optimiser(opt,Flux.WeightDecay(λ))

m = 3
d = 27
# max clusters
k = 12

mnist = mnistloader(batchsize)

θ,ϕ,π,L_π,L_ϕ = trainouter(m,mnist,opt,epochs,
                           path*"outer/nowd/")

outer_wd = trainouter(m,mnist,opt_wd,epochs,
                      path*"outer/wd/")

sae = trainsae(m,d,θ,π,ϕ,α,mnist,opt,epochs,
               path*"inner/nowd/")
sae_wd = trainsae(m,d,θ,π,ϕ,α,mnist,opt_wd,epochs,
                  path*"inner/wd/")

psae = trainpsae(m,d,k,θ,π,ϕ,α,mnist,opt,epochs,
                 path*"inner/nowd/")
psae_wd = trainpsae(m,d,k,θ,π,ϕ,α,mnist,opt_wd,epochs,
                    path*"inner/wd/")
