include("SAE.jl")
include("PSAE.jl")
include("trainingfns.jl")
include("models.jl")
include("auxfns.jl")

using StatsPlots
using Flux: logitcrossentropy

using JLD2,Tables,CSV

using ImageInTerminal,Images

# where to write the toy model
path = "data/MNIST/outer/"

epochs = 100
batchsize = 512
m = 3

η = 0.001
λ = 0.000
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

loader = mnistloader(batchsize)

θ = outerenc()
ϕ = outerdec() |> gpu

M_outerenc = Chain(θ,ϕ) |> gpu

L_outerenc = []


train!(ϕ,loader,opt,epochs,Flux.mse,L_outerenc;
       prefn=M_outerclas[1],
       ignoreY=true,savecheckpts=true,path=path*"encoder/");

π = outerclassifier()
M_outerclas = Chain(θ,π) |> gpu
L_outerclas = []

train!(M_outerclas,loader,opt,epochs,logitcrossentropy,L_outerclas,
       savecheckpts=true,path=path*"classifier/");

p = scatter(1:length(L_outerclas), L_outerclas,
            xlabel="batch",ylabel="loss",
            label="outer");
savefig(p,path*"classifier/loss.pdf")

ŷ = M_outerclas(x)
labels = string.(unhot(cpu(y)))[1,:]

grD.pdf(path*"classifier/logits.pdf")
CH.Heatmap(ŷ',"P(label)",col=["white","blue"],
           split=labels,border=true)
grD.dev_off()

p = scatter(1:length(L_outerenc), L_outerenc,
            xlabel="batch",ylabel="loss",
            label="outer");
savefig(p,path*"encoder/loss.pdf")

x,y = first(loader) |> gpu
colorview(Gray,x[:,:,1,1:2])
colorview(Gray,M_outerenc(x[:,:,1,1:2]))

