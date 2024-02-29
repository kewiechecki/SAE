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

psae_clas,L_clas = loadpsae(m,d,k,path*"inner/nowd/classifier/PSAE/L1_L2/")
psae_clas_L2,L2_clas = loadpsae(m,d,k,path*"inner/nowd/classifier/PSAE/L2/")
psae_clas_wd,L_clas_wd = loadpsae(m,d,k,path*"inner/wd/classifier/PSAE/L1_L2/")
psae_clas_L2_wd,L2_clas_wd = loadpsae(m,d,k,path*"inner/wd/classifier/PSAE/L2/")

psae_enc,L_enc = loadpsae(m,d,k,path*"inner/nowd/encoder/PSAE/L1_L2/")
psae_enc_L2,L2_enc = loadpsae(m,d,k,path*"inner/nowd/encoder/PSAE/L2/")
psae_enc_wd,L_enc_wd = loadpsae(m,d,k,path*"inner/wd/encoder/PSAE/L1_L2/")
psae_enc_L2_wd,L2_enc_wd = loadpsae(m,d,k,path*"inner/wd/encoder/PSAE/L2/")
