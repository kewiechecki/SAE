include("SAE.jl")
include("PSAE.jl")
include("trainingfns.jl")
include("models.jl")
include("auxfns.jl")

using MLDatasets, StatsPlots, OneHotArrays
using Flux: logitcrossentropy

using JLD2,Tables,CSV
using StatsPlots

# where to write the toy model
path = "data/MNIST/"

epochs = 100
batchsize=512

η = 0.001
λ = 0.000
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))
α = 0.001

m = 3
d = 27
# max clusters
k = 12

loader = mnistloader(batchsize)
θ,π,ϕ = loadouter(m,path)
outer = Chain(θ,ϕ)

sae = SAE(m,d) |> gpu
L_SAE = []
train!(sae,outer,α,loader,opt,epochs,Flux.mse,L_SAE,path*"inner/SAE/")

partitioner = Chain(Dense(m => k,relu))
psae = PSAE(sae,partitioner) |> gpu
L_PSAE = []
train!(psae,M_outer,α,loader,opt,epochs,Flux.mse,L_PSAE,"inner/PSAE/")

state_SAE = Flux.state(sae) |> cpu;
jldsave(path*"state_SAE.jld2";state_SAE)
Tables.table(L_SAE) |> CSV.write(path*"L_SAE.csv")

sae_linear = SAE(m,d,identity) |> gpu
L_linear = []
train!(sae_linear,M_outer,α,loader,opt,epochs,logitcrossentropy,L_linear)

state_linear = Flux.state(sae_linear) |> cpu;
jldsave(path*"state_linear.jld2";state_linear)
Tables.table(L_linear) |> CSV.write(path*"L_linear.csv")

p = scatter(1:length(L_SAE), L_SAE,
            xlabel="batch",ylabel="loss",
            legend=:none)
savefig(p,"data/MNIST/loss_SAE.pdf")


psae_linear = PSAE(sae_linear,partitioner) |> gpu
L_PSAElinear = []
train!(psae_linear,M_outer,α,loader,opt,epochs,logitcrossentropy,L_PSAElinear)
p = scatter(1:length(L_SAE), L_SAE,
            xlabel="batch",ylabel="loss",
            label="SAE")
scatter!(1:length(L_PSAE), L_PSAE,label="PSAE")
savefig(p,"data/MNIST/loss_relu.svg")

p = scatter(1:length(L_linear), L_linear,
            xlabel="batch",ylabel="loss",
            label="SAE")
scatter!(1:length(L_PSAElinear), L_PSAElinear,label="PSAE")
savefig(p,"data/MNIST/loss_linear.svg")

train!(psae_linear,M_outer,α,loader,opt,1000,logitcrossentropy,L_PSAElinear)

state_PSAE1k = Flux.state(psae_linear) |> cpu;
jldsave(path*"state_PSAE1k.jld2";state_PSAE1k)
Tables.table(L_PSAElinear) |> CSV.write(path*"L_PSAElinear1k.csv")
p = scatter(1:length(L_linear), L_linear,
            xlabel="batch",ylabel="loss",
            label="SAE")
scatter!(1:length(L_PSAElinear), L_PSAElinear,label="PSAE")
savefig(p,"data/MNIST/loss_linear1k.svg")

include("Rmacros.jl")
