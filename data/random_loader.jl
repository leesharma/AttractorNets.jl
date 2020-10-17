include("../src/utils.jl")
using .AttractorNets.Utils: random_discrete_patterns


function dataset(M, N)
    random_discrete_patterns(M, N)
end
