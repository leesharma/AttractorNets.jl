module AttractorNets
module Utils

using Plots


function random_discrete_patterns(N, M)
    rand((-1,1), N, M)
end
export random_discrete_patterns

function hamming(a1, a2)
    if length(a1) != length(a2)
        error("Arrays a1 and a2 must be the same length")
    end
    sum([a1[i]!=a2[i] for i in 1:length(a1)])
end

function animate_weights(filename, A; title="Post-training weights", fps=3, cmap=:balance, clim=0.25, display=:box)
    N, M = size(A)

    pattern_cmap=:grayC

    anim = @animate for i in [1:M... repeat([M],fps)...]
        W = learn(N, A[:,1:i])
        if display==:box
            l = @layout [ a ; grid(1, 9){0.15h} ]
        elseif display==:line
            l = @layout [ a ; b{0.2h} ]
        else
            l = 1
        end
        plot(layout=l)
        # main heatmap
        plot!(W, st=:heatmap, subplot=1, yflip=true, c=cmap, clims=(-clim,clim))

        # number patterns
        if display==:box
            if sqrt(N)%1!=0
                error("Box display only works for square data")
            end

            side=Int(sqrt(N))
            for ai in 1:M
                plot!(subplot=ai+1, yflip=true, c=pattern_cmap, colorbar=false, ticks=nothing, clim=(-1,1))
                if ai <= i
                    plot!(reshape(A[:,ai],(side,side)), subplot=ai+1, st=:heatmap, c=pattern_cmap)
                else
                    plot!(subplot=ai+1, axis=false)
                end
            end
            xlabel!(" ", subplot=1)
        elseif display==:line
            plot!(subplot=2, yflip=true, c=pattern_cmap, colorbar=false, ticks=nothing)
            plot!([A[:,1:i] -ones(N,max(M-i,N))], subplot=2, st=:heatmap, c=pattern_cmap)
            xlabel!(" ", subplot=1)
        end
        title!("$(title) (M=$(i))", subplot=1)
        n_stable = Hopnet.num_stable(W,A[:,1:i])
        if n_stable != i
            plot!(foreground_color=:red, background_color=:lightgray)
           title!("$(title) (M=$(i))", subplot=1)
           xlabel!("CATASTROPHIC FORGETTING (num stable=$(n_stable))", subplot=1)
           if display==:line
               plot!([A[:,1:i] -0.7*ones(N,max(M-i,N))], subplot=2, st=:heatmap, c=pattern_cmap, clim=(-1,1))
           end
        end
    end
    gif(anim, filename, fps=fps)
end

end # Utils
end # AttractorNets
