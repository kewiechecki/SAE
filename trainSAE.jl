include("SAE.jl")
include("trainingfns.jl")

using MLDatasets, StatsPlots, OneHotArrays
using Flux: logitcrossentropy

using JLD2,Tables,CSV

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

sae = SAE(m,d) |> gpu

function loss_SAE(M_outer,α,lossfn,x)
    x = gpu(x)
    yhat = M_outer(x)
    f = M->L1(M,α,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end

L_SAE = []
@showprogress map(1:epochs) do _
    map(loader) do (x,y)
        f = loss_SAE(M_outer,α,logitcrossentropy,x)
        state = Flux.setup(opt,sae)
        l,∇ = Flux.withgradient(f,sae)
        Flux.update!(state,sae,∇[1])
        push!(L_SAE,l)
    end
end

state_SAE = Flux.state(sae) |> cpu;
jldsave(path*"state_SAE.jld2";state_SAE)
Tables.table(L_SAE) |> CSV.write(path*"L_SAE.csv")

