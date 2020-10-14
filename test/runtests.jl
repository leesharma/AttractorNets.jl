using Pkg; Pkg.activate("Hopnet"); Pkg.instantiate()

using Test
using SafeTestsets

@time @safetestset "Basic Hopnet functions" begin include("test_Hopnet.jl") end
