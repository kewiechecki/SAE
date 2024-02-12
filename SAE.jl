using Flux, Functors
abstract type SparseEncoder
end

struct SAE <: SparseEncoder
    weight::AbstractArray
    bias::AbstractArray
    σ::Function
end
# the magic line that makes everything "just work"
@functor SAE 

# constructor specifying i/o dimensions
function SAE(m,d,σ=relu)
    weight = randn(d,m)
    bias = randn(d)
    return SAE(weight,bias,σ)
end

#
function encode(M::SAE,x)
    return M.σ(M.weight * x .+ M.bias)
end

function decode(M::SAE,c)
    return M.weight' * c
end

struct SED <: SparseEncoder
    encoder::Chain
    decoder::Chain
end
@functor SED

function encode(M::SED,x)
    return M.encoder(x)
end

function decode(M::SED,c)
    return M.decoder(c)
end


function (M::SparseEncoder)(x)
    c = encode(M,x)
    x̂ =  decode(M,c)
    return x̂
end

function L1(M::SparseEncoder,α,x)
    c = encode(M,x)
    return α * sum(abs(c))
end

function L2(M::SparseEncoder,lossfn,x,y)
    return lossfn(M(x),y)
end

function loss(M::SparseEncoder,α,lossfn,x,y)
    x = gpu(x)
    y = gpu(y)
    return L1(M,α,x) + L2(M,lossfn,x,y)
end

function loss_SAE(α,lossfn,x,y)
    return M->loss(M,α,lossfn,x,y)
end

function loss_SAE(M_outer,α,lossfn,x)
    x = gpu(x)
    yhat = M_outer(x)
    f = M->L1(M,α,x) + L2(M,lossfn,x,yhat)
    return m->lossfn((M_outer[2] ∘ m ∘ M_outer[1])(x),yhat)
end
