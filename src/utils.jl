function random_discrete_patterns(N, M)
    rand((-1,1), N, M)
end
export random_patterns

function hamming(a1, a2)
    if length(a1) != length(a2)
        error("Arrays a1 and a2 must be the same length")
    end
    sum([a1[i]!=a2[i] for i in 1:length(a1)])
end
