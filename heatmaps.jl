include("Rmacros.jl")

function Kheatmap(M,x,y)
    ŷ = M(x)
    labels = string.(unhot(cpu(y)))[1,:]
    return CH.Heatmap(ŷ',"P(label)",col=["white","blue"],
                      split=labels,border=true)
end

function Eheatmap(M,x,y)
    ŷ = M(x)
    labels = string.(unhot(cpu(y)))[1,:]
    return CH.Heatmap(ŷ',"P(label)",col=["white","red"],
                      split=labels,border=true)
end

function drawheatmap(H,out)
    grD.pdf(out)
    CH.draw(H)
    grD.dev_off()
end

function plotlogits(M,x,y,out)
    ŷ = M(x)
    labels = string.(unhot(cpu(y)))[1,:]

    grD.pdf(out)
    CH.Heatmap(ŷ',"P(label)",col=["white","blue"],
            split=labels,border=true)
    grD.dev_off()
end
