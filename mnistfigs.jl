include("SAE.jl")
include("PSAE.jl")
include("models.jl")
include("auxfns.jl")
include("Rmacros.jl")

using MLDatasets, StatsPlots, OneHotArrays

path = "data/MNIST/"
batchsize=128

m = 3
d = 27
k = 12

mnist = mnistloader(batchsize)
x,y = first(mnist) |> gpu
labels = string.(unhot(cpu(y)))[1,:]

θ,ϕ,π,L_π,L_ϕ = loadouter(m, path*"outer/nowd")

sae_clas,L_clas = loadsae(m,d,path*"inner/nowd/classifier/SAE/L1_L2/")
sae_clas_L2,L2_clas = loadsae(m,d,path*"inner/nowd/classifier/SAE/L2/")
sae_clas_wd,L_clas_wd = loadsae(m,d,path*"inner/wd/classifier/SAE/L1_L2/")
sae_clas_L2_wd,L2_clas_wd = loadsae(m,d,path*"inner/wd/classifier/SAE/L2/")

sae_enc,L_enc = loadsae(m,d,path*"inner/nowd/encoder/SAE/L1_L2/")
sae_enc_L2,L2_enc = loadsae(m,d,path*"inner/nowd/encoder/SAE/L2/")
sae_enc_wd,L_enc_wd = loadsae(m,d,path*"inner/wd/encoder/SAE/L1_L2/")
sae_enc_L2_wd,L2_enc_wd = loadsae(m,d,path*"inner/wd/encoder/SAE/L2/")

psae_clas,L_PSAE_clas = loadpsae(m,d,k,path*"inner/nowd/classifier/PSAE/L1_L2/")
psae_clas_L2,L2_PSAE_clas = loadpsae(m,d,k,path*"inner/nowd/classifier/PSAE/L2/")
psae_clas_wd,L_PSAE_clas_wd = loadpsae(m,d,k,path*"inner/wd/classifier/PSAE/L1_L2/")
psae_clas_L2_wd,L2_PSAE_clas_wd = loadpsae(m,d,k,path*"inner/wd/classifier/PSAE/L2/")

psae_enc,L_PSAE_enc = loadpsae(m,d,k,path*"inner/nowd/encoder/PSAE/L1_L2/")
psae_enc_L2,L2_PSAE_enc = loadpsae(m,d,k,path*"inner/nowd/encoder/PSAE/L2/")
psae_enc_wd,L_PSAE_enc_wd = loadpsae(m,d,k,path*"inner/wd/encoder/PSAE/L1_L2/")
psae_enc_L2_wd,L2_PSAE_enc_wd = loadpsae(m,d,k,path*"inner/wd/encoder/PSAE/L2/")

f = (L,lab,ylab)->scatter(1:length(L),L,
                          xlabel="batch", ylabel=ylab,
                          label=lab)

f! = (L,lab)->scatter!(1:length(L),L,label=lab)

p = f(L_clas,"L1+L2, no WD","logitCE");
map(x->f!(x...),[(L2_clas,"L2, no WD"),
        (L_clas_wd,"L1+L2, WD"),
        (L2_clas_wd,"L2, WD")]);
savefig(p,"data/MNIST/L_SAE_clas.pdf")

p = f(L_PSAE_clas,"L1+L2, no WD","logitCE");
map(x->f!(x...),[(L2_clas,"L2, no WD"),
        (L_PSAE_clas_wd,"L1+L2, WD"),
        (L2_PSAE_clas_wd,"L2, WD")]);
savefig(p,"data/MNIST/L_PSAE_clas.pdf");

sp = scatter(1:length(L_clas),L_clas
p = scatter(1:length(L_SAE), L_SAE,
            xlabel="batch",ylabel="loss",
            label="SAE")
scatter!(1:length(L_PSAE), L_PSAE,label="PSAE")
savefig(p,"data/MNIST/loss_relu.svg")
