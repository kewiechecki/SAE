using ProgressMeter, MLDatasets, OneHotArrays
using JLD2,Tables,CSV

function update!(M,x,y,loss::Function,opt)
    #x = gpu(x)
    #y = gpu(y)
    f = m->loss(m(x),y)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(f,M)
    Flux.update!(state,M,∇[1])
    return l
end

function update!(M,loss::Function,opt)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    Flux.update!(state,M,∇[1])
    return l
end

function savemodel(M,path)
    state = Flux.state(M) |> cpu;
    jldsave(path*".jld2";state)
end

function train!(M,loader,opt,epochs,loss,log;
                prefn = identity,postfn=identity,
                ignoreY=false,savecheckpts=false,
                path="")
    if length(path) > 0
        mkpath(path)
    end
    f = (x,y)->loss(postfn(x),y)
    @showprogress map(1:epochs) do i
        map(loader) do (x,y)
            x = gpu(x)
            y = gpu(y)
            if ignoreY
                y = x
            end
            x = prefn(x)
            l = update!(M,x,y,f,opt)
            push!(log,l)
        end
        if savecheckpts
            savemodel(M,path*string(i))
        end
    end
    if length(path) > 0
        savemodel(M,path*"final")
        Tables.table(log) |> CSV.write(path*"loss.csv")
    end
end

function train!(sae::Union{SAE,PSAE},M_outer,α,loader,opt,epochs,lossfn,log)
    @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            f = loss_SAE(M_outer,α,lossfn,x)
            state = Flux.setup(opt,sae)
            l,∇ = Flux.withgradient(f,sae)
            Flux.update!(state,sae,∇[1])
            push!(log,l)
        end
    end
end

