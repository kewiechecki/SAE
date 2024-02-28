using Flux, Functors, Zygote

struct SparseDict
    dict
end
@functor SparseDict

function SparseDict(d::Integer,k::Integer)
    dict = randn(Float32,d,k)
    return SparseDict(dict)
end

function (M::SparseDict)(K)
    return (K' * M.dict')'
end

function Zygote.pullback(M::SparseDict, K::AbstractArray)
    E = (K' * M.dict')'  # Forward pass

    function B(Δ)
        # Transpose Δ since E was transposed in the forward pass
        Δ = Δ'
        # Gradient w.r.t M.dict, which is Δ * K
        ∇_dict = Δ * K
        # Gradient w.r.t. K, which is M.dict * Δ
        ∇_K = M.dict * Δ
        # Return the gradients in the expected structure by Zygote
        return (dict=∇_dict,), ∇_K
    end
    
    return E, B
end

struct SparseClassifier
    dict::SparseDict
    classifier
end
@functor SparseClassifier

function SparseClassifier(m::Integer,d::Integer,k::Integer,σ=relu)
    dict = SparseDict(d,k)
    classifier = Dense(m => k,σ)
    return SparseClassifier(dict,classifier)
end

function cluster(M::SparseClassifier,X)
    return (softmax ∘ M.classifier)(X)
end

function (M::SparseClassifier)(X)
    K = cluster(M,X)
    return M.dict(K)
end
