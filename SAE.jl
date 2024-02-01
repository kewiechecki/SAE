using Flux, Functors

struct SAE
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

function (M::SAE)(x)
    c = encode(M,x)
    x̂ =  decode(M,c)
    return x̂
end

function L1(M::SAE,α,x)
    c = encode(M,x)
    return α * sum(abs(c))
end

function L2(M::SAE,lossfn,x,y)
    return lossfn(M(x),y)
end

function loss(M::SAE,α,lossfn,x,y)
    x = gpu(x)
    y = gpu(y)
    return L1(M,α,x) + L2(M,lossfn,x,y)
end

function loss_SAE(α,lossfn,x,y)
    return M->loss(M,α,lossfn,x,y)
end
