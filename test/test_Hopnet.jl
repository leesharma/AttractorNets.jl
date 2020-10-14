include("../src/Hopnet.jl")
using .Hopnet
using LinearAlgebra: Symmetric


@testset "learning" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = Hopnet.learn(4, A)
    W_expected = [
         0     0.25  0.25 -0.25
         0.25  0    -0.25  0.25
         0.25 -0.25  0     0.25
        -0.25  0.25  0.25  0
    ]

    @test W[1,1] == W[2,2] == W[3,3] == W[4,4] == 0
    @test isequal(W, Symmetric(W))
    @test isequal(W, W_expected)
end

@testset "integration rule" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    @test in_i(W, a2, 1) ==  0.25
    @test in_i(W, a2, 2) == -0.25
    @test in_i(W, a2, 3) ==  0.25
    @test in_i(W, a2, 4) == -0.25

    @test in_i(W, [-1;1;1;1], 1) == 0.25
end

@testset "activation rule" begin
    @test a_i(1.1,1)  ==  1
    @test a_i(-1.1,1) == -1
    @test a_i(0,1)    ==  1
    @test a_i(0,-1)   == -1
end

@testset "energy" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    @test isequal(-0.5, energy(W, A[:,2]))
end

@testset "fixedpoint" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    @test is_fixedpoint(W, a1)
    @test is_fixedpoint(W, a2)
    @test is_fixedpoint(W, a3)

    @test !is_fixedpoint(W, [-1 1 1 1]')

    # complements
    complement(a) = (ai -> ai==1 ? -1 : 1).(a)
    @test is_fixedpoint(W, complement(a1))
    @test is_fixedpoint(W, complement(a2))
    @test is_fixedpoint(W, complement(a3))
end
