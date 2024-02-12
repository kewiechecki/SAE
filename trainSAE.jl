include("SAE.jl")
include("PSAE.jl")
include("trainingfns.jl")

using MLDatasets, StatsPlots, OneHotArrays
using Flux: logitcrossentropy

using JLD2,Tables,CSV
using StatsPlots

# where to write the toy model
path = "data/MNIST/"

epochs = 100
batchsize=512

η = 0.001
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

m = 3
d = 27
α = 0.001

dat = MNIST(split=:train)[:]
target = onehotbatch(dat.targets,0:9)

m_x,m_y,n = size(dat.features)
X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)

loader = Flux.DataLoader((X,target),
                         batchsize=batchsize,
                         shuffle=true)
#load outer model
M_outer = outermodel() |> gpu
state_outer = JLD2.load("data/MNIST/state_outer.jld2","state_outer");
Flux.loadmodel!(M_outer,state_outer)

function loss_SAE(M_outer,α,lossfn,x)
    x = gpu(x)
    yhat = M_outer(x)
    f = M->L1(M,α,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end

sae = SAE(m,d) |> gpu
L_SAE = []
train!(sae,M_outer,α,loader,opt,epochs,logitcrossentropy,L_SAE)

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

# max clusters
k = 12
partitioner = Chain(Dense(m => k,relu))

psae = PSAE(sae,partitioner) |> gpu

L_PSAE = []

train!(psae,M_outer,α,loader,opt,epochs,logitcrossentropy,L_PSAE)

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
