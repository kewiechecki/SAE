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
path = "data/MNIST/"

epochs = 1000
batchsize = 512
m = 3

η = 0.001
λ = 0.001
opt = Flux.AdamW(η)
opt_wd = Flux.Optimiser(opt,Flux.WeightDecay(λ))

mnist = mnistloader(batchsize)

θ,ϕ,π,L_π,L_ϕ = trainouter(m,mnist,opt,epochs;
                           path=path*"outer/nowd/")

outer_wd = trainouter(m,mnist,opt_wd,epochs;
                      path=path*"outer/wd/")

θ = outerenc() |> gpu
ϕ = outerdec() |> gpu
π = outerclassifier() |> gpu
outerclas = Chain(θ,π)

L_clas = train!(outerclas,
                loader,opt,epochs,logitcrossentropy;
                savecheckpts=true,
                path=path*"nowd/classifier/");
L_enc = train!(ϕ,
               loader,opt,epochs,Flux.mse;
               prefn=outerclas[1],
               ignoreY=true,savecheckpts=true,
               path=path*"nowd/encoder/");

θ = outerenc() |> gpu
ϕ = outerdec() |> gpu
π = outerclassifier() |> gpu
outerclas = Chain(θ,π)

L_clas = train!(outerclas,
                loader,opt_wd,epochs,logitcrossentropy;
                savecheckpts=true,
                path=path*"wd/classifier/");
L_enc = train!(ϕ,
               loader,opt_wd,epochs,Flux.mse;
               prefn=outerclas[1],
               ignoreY=true,savecheckpts=true,
               path=path*"wd/encoder/");

M_outerclas = Chain(θ,π) |> gpu
L_outerclas = []

train!(M_outerclas,loader,opt,epochs,logitcrossentropy,L_outerclas,
       savecheckpts=true,path=path*"nowd/classifier/");

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

