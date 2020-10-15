using Pkg; Pkg.activate("AttractorNets"); Pkg.instantiate()

using Test
using SafeTestsets

@time @safetestset "Hopfield Networks" begin include("test_Hopnet.jl") end
