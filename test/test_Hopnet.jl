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

@testset "next activation" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    @test a_i(W, a2, 1) ==  1
    @test a_i(W, a2, 2) == -1
    @test a_i(W, a2, 3) ==  1
    @test a_i(W, a2, 4) == -1

    @test a_i(W, [-1;1;1;1], 1) == 1
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

@testset "run epoch" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    # fixedpoint
    @test all(run_epoch(W, a1) .== a1)

    # non-fixedpoint (1 epoch to stable)
    a = [-1 1 1 1]'
    a_new1 = run_epoch(W, a; node_order=[1 2 3 4])
    a_new2 = run_epoch(W, a_new1; node_order=[1 2 3 4])

    @test any(a_new1 .!= a)
    @test isequal(a_new1, [1 1 1 1]')
    @test isequal(a_new2, a_new1)
end

@testset "run to fixedpoint" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    # fixedpoint
    steps, a1_new = run_to_fixedpoint(W, a1; node_order_fn=_->[1 2 3 4])
    @test steps == 0
    @test isequal(a1_new, a1)

    # non-fixedpoint
    a = [-1 1 1 1]'
    steps, a_new = run_to_fixedpoint(W, a; node_order_fn=_->[1 2 3 4])
    @test steps == 1
    @test !isequal(a_new, a)
    @test isequal(a_new, [1 1 1 1]')

    # order can matter
    steps, a_new = run_to_fixedpoint(W, a; node_order_fn=_->[4 1 2 3])
    @test steps == 1
    @test isequal(a_new, [1 1 1 1]')
end
