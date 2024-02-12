# SAE implementation using a partitoning submodel.
using Flux, Functors
using CUDA, LinearAlgebra

struct PSAE
    sae::SparseEncoder
    partitioner::Chain
end
@functor PSAE

function encode(M::PSAE,X)
    return encode(M.sae,X)
end

function cluster(M::PSAE,X)
    return M.partitioner(X)
end

function partition(M::PSAE,X)
    return (pwak ∘ cluster)(M,X)
end

function encodepred(M::PSAE,X)
    E = encode(M,X)
    P = partition(M,X)
    return (P * E')'
end

function decode(M::PSAE,X)
    return decode(M.sae,X)
end


function (M::PSAE)(X::AbstractMatrix)
    Ehat = encodepred(M,X)
    return decode(M,Ehat)
end

    
function L1(M::PSAE,α,x)
    c = encodepred(M,x)
    return α * sum(abs.(c))
end

function L2(M::PSAE,lossfn,x,y)
    return lossfn(M(x),y)
end

function loss_PSAE(M::PSAE,α,lossfn,x,y)
    x = gpu(x)
    y = gpu(y)
    return L1(M,α,x) + L2(M,lossfn,x,y)
end

function zerodiag(G::AbstractArray)
    m, n = size(G)
    G = G .* (1 .- I(n))
    return G
end

# [CuArray] -> [CuArray]
# workaround for moving identity matrix to GPU 
function zerodiag(G::CuArray)
    m, n = size(G)
    G = G .* (1 .- I(n) |> gpu)
    return G
end

# constructs weighted affinity kernel from adjacency matrix
# sets diagonal to 0
# normalizes rows/columns to sum to 1 (default := columns)
function wak(G::AbstractArray; dims=1)
    G = zerodiag(G)
    G = G ./ (sum(G,dims=dims) .+ eps(eltype(G)))
    return G
end

function pwak(K::AbstractMatrix; dims=1)
    P = K' * K
    return wak(P)
end
