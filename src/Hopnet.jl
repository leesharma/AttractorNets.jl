"""
Construct, train, and query Hopfield networks.
"""
module Hopnet

using LinearAlgebra: Symmetric, I
using Random: randperm
using Plots

include("utils.jl")
include("learning.jl")


# Training the net

"""
    learn(N, A)

Returns the weights for a N-node Hopfield network after training on patterns A.

Uses one-step Hebbian learning.
"""
function learn(N, A)
    hebbian_learning(N, A)
end
export learn

# Running the net

"""
    run_to_fixedpoint(W, a_init[; node_order_fn=f])

Runs the Hopfield network, returning the required epochs and attractor pattern.

By default, odes are updated asynchronously in a random order. Update order
can also be specified by the `node_order_fn`, which should be a lambda that
takes an index and returns a list of indices for that epoch.
"""
function run_to_fixedpoint(W, a_init; node_order_fn=_->randperm(length(a_init)), sync=false, max_epochs=1000, do_plot=false)
    current_epoch = 0
    a = copy(a_init)  # non-mutating
    if do_plot
        history = a_init
    end

    while !is_fixedpoint(W,a)
        if current_epoch >= max_epochs
            error("Likely limit cycle detected ($(current_epoch)/$(max_epochs) epochs)")
        end
        current_epoch += 1
        a = run_epoch(W,a; node_order=node_order_fn(current_epoch), sync=sync)
        if do_plot
            history = [history a]
        end
    end

    if do_plot
        fps=3
        anim = @animate for a in eachcol([history repeat(history[:,end], inner=(1,fps*2))])
            plot(reshape(a,(8,8)), st=:heatmap, c=:binary, colorbar=false, yflip=true)
        end
        gif(anim, "training.gif", fps = fps)
    end

    (current_epoch, a)
end
export run_to_fixedpoint

# visualization

function random_capacity_plot(N, M)
    mean(x) = sum(x)/length(x)

    A = random_discrete_patterns(N,M)
    W = learn(N, A)
    stable = [Hopnet.num_stable(learn(N, A[:,1:i]), A[:,1:i]) for i in 1:M]
    dists = [mean(Hopnet.stable_distances(learn(N, A[:,1:i]), A[:,1:i])) for i in 1:M]

    plot(1:M, stable)
    plot!(1:M, dists)
    title!("Random memory capacity of a Hopfield network (N=$(N))")
    xlabel!("num patterns")
end
export random_capacity_plot

# Query the net

"""
    energy(W, a)

Calculates the energy of an activation pattern within the given network.
"""
function energy(W, a)::Float64
    -1/2 * a' * W * a
end
export energy

"""
    num_stable(W, A)

Returns the number of patterns in A stored as fixedpoints in Hopfield net W.
"""
function num_stable(W, A)::Int
    count([is_fixedpoint(W,a) for a in eachcol(A)])
end
export num_stable

"""
    stable_distances(W, A[, node_order_fn=f])

Returns the hamming distance between each input pattern and its attractor.

By default, odes are updated asynchronously in a random order. Update order
can also be specified by the `node_order_fn`, which should be a lambda that
takes an index and returns a list of indices for that epoch.
"""
function stable_distances(W, A; node_order_fn=_->randperm(size(W)[1]))
    function stable_distance(W, a; node_order_fn=_->randperm(length(a)))
        _, a_stable = run_to_fixedpoint(W,a;node_order_fn=node_order_fn)
        hamming(a, a_stable)
    end

    [stable_distance(W,a; node_order_fn=node_order_fn) for a in eachcol(A)]
end
export stable_distances

###

# Private Functions

# integration rule
in_i(W, a, i) = W[1:end.!==i,i]' * a[1:end.!==i]
# activation rule
a_i(in_i, a0)::Int = in_i==0 ? a0 : Int(sign(in_i))
a_i(W, a, i)::Int = a_i(in_i(W,a,i), a[i])

is_fixedpoint(W, a)::Bool = all(a.==run_epoch(W, a))

# runs one epoch for a pattern, updating nodes randomly and async by default
function run_epoch(W, a_old; node_order=randperm(length(a_old)), sync=false)
    a_new = copy(a_old)  # non-mutating
    a_in = sync ? a_old : a_new

    # async update
    for i in node_order
        a_new[i] = a_i(W,a_in,i)
    end
    a_new
end

end # module Hopnet
