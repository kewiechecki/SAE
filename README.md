A simple sparse autoencoder implementation in Julia.

# Usage
This example trains an SAE on a toy model of superposition (the "outer model").
The outer model consists of an 
```{bash}
julia mnist.jl
```

# Defining the SAE

```{julia}
include("SAE.jl")

# SAE hyperparameters
# input dimension
m = 3
# number of features
d = 27

sae = SAE(m,d,relu)
```
# Training the SAE
Training hyperparameters
```{julia}
epochs = 100
batchsize=512

# learning rate
η = 0.001

# weight decay rate
λ = 0.001
opt = Flux.Optimiser(Flux.AdamW(η),Flux.WeightDecay(λ))

# sparcity coefficient
α = 0.001
```
Import outer model
```{julia}
include("trainingfns.jl"

M_outer = outermodel()
state_outer = JLD2.load("data/MNIST/state_outer.jld2","state_outer");
Flux.loadmodel!(M_outer,state_outer)
```
Train SAE on bottleneck activations of outer model
```{julia}

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

#
```
Save model
```{julia}
state_SAE = Flux.state(sae) |> cpu;
jldsave(path*"state_SAE.jld2";state_SAE)
Tables.table(L_SAE) |> CSV.write(path*"L_SAE.csv")
```