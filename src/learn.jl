module AttractorNets
module Learn

using LinearAlgebra: Symmetric, I


function hebbian_learning(N, A, η=1/N)
    M = size(A)[2]
    correlations = sum(map(a->a*a', eachcol(A)))

    W = Symmetric(η*correlations - M/N*I)
end

end # Learn
end # AttractorNets
