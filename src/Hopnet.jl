module Hopnet

using LinearAlgebra: Symmetric, I
using Random: randperm


# Hebbian learning

function learn(N, A)
    M = size(A)[2]
    correlations = sum(map(a->a*a', eachcol(A)))

    W = Symmetric(1/N*correlations - M/N*I)
end
export learn

function random_discrete_patterns(N, M)
    rand((-1,1), N, M)
end
export random_patterns

# Query the net

energy(W, a) = -1/2 * a' * W * a
export energy

# Running the net

in_i(W, a, i) = W[1:end.!==i,i]' * a[1:end.!==i]
a_i(in_i, a0)::Int = in_i==0 ? a0 : Int(sign(in_i))
a_i(W, a, i)::Int = a_i(in_i(W,a,i), a[i])
export in_i, a_i

function is_fixedpoint(W, a_old)::Bool
    in_i(i) = Hopnet.in_i(W, a_old, i)
    N = length(a_old)

    in_all = in_i.(1:N)
    all(a_old .== sign.(in_all))
end
export is_fixedpoint

function run_epoch(W, a; node_order=randperm(length(a)))
    ac = copy(a)  # non-mutating
    for i in node_order
        ac[i] = a_i(W,ac,i)
    end
    ac
end
export run_epoch

end # module
