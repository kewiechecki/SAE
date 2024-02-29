using ProgressMeter, MLDatasets, OneHotArrays
using JLD2,Tables,CSV

function update!(M,x,y,
                 loss::Function,
                 opt::Flux.Optimiser)
    #x = gpu(x)
    #y = gpu(y)
    f = m->loss(m(x),y)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(f,M)
    Flux.update!(state,M,∇[1])
    return l
end

function update!(M,
                 loss::Function,
                 opt::Flux.Optimiser)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    Flux.update!(state,M,∇[1])
    return l
end

function savemodel(M,path)
    state = Flux.state(M) |> cpu;
    jldsave(path*".jld2";state)
end

function train!(M,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                loss::Function,
                log;
                prefn = identity,
                postfn=identity,
                ignoreY=false,
                savecheckpts=false,
                path="")
    if length(path) > 0
        mkpath(path)
    end
    f = (E,y)->loss(postfn(E),y)
    @showprogress map(1:epochs) do i
        map(loader) do (x,y)
            x = gpu(x)
            y = gpu(y)
            if ignoreY
                y = (postfn ∘ prefn)(x)
            end
            E = prefn(x)
            l = update!(M,E,y,f,opt)
            push!(log,l)
        end
        if savecheckpts
            savemodel(M,path*string(i))
        end
    end
    if length(path) > 0
        savemodel(M,path*"/final")
        Tables.table(log) |> CSV.write(path*"/loss.csv")
    end
end

function train!(M,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer,
                loss::Function;
                kwargs...)
    log = []
    train!(M,loader,opt,epochs,loss,log;kwargs...)
    return log
end
