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

@testset "energy" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    @test isequal(-0.5, energy(W, A[:,2]))
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

    # order can matter with async
    steps, a_new = run_to_fixedpoint(W, a; node_order_fn=_->[4 2 3 1])
    @test steps == 1
    @test isequal(a_new, [-1 -1 1 1]')

    # detects limit cycles
    @test_throws ErrorException run_to_fixedpoint(W, a; sync=true)
end

@testset "count stable points" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    a = [-1 1 1 1]'  # non-fixedpoint

    @test num_stable(W, A) == 3
    @test num_stable(W, [A a]) == 3
end

@testset "measure distances to stable points" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    a = [-1 1 1 1]'  # non-fixedpoint

    @test isequal(Hopnet.stable_distances(W, A), [0, 0, 0])
    @test isequal(Hopnet.stable_distances(W, [A a]; node_order_fn=_->[1 2 3 4]), [0, 0, 0, 1])
end

###

@testset "integration rule (priv)" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    @test Hopnet.in_i(W, a2, 1) ==  0.25
    @test Hopnet.in_i(W, a2, 2) == -0.25
    @test Hopnet.in_i(W, a2, 3) ==  0.25
    @test Hopnet.in_i(W, a2, 4) == -0.25

    @test Hopnet.in_i(W, [-1;1;1;1], 1) == 0.25
end

@testset "activation rule (priv)" begin
    @test Hopnet.a_i(1.1,1)  ==  1
    @test Hopnet.a_i(-1.1,1) == -1
    @test Hopnet.a_i(0,1)    ==  1
    @test Hopnet.a_i(0,-1)   == -1
end

@testset "next activation (priv)" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    @test Hopnet.a_i(W, a2, 1) ==  1
    @test Hopnet.a_i(W, a2, 2) == -1
    @test Hopnet.a_i(W, a2, 3) ==  1
    @test Hopnet.a_i(W, a2, 4) == -1

    @test Hopnet.a_i(W, [-1;1;1;1], 1) == 1
end

@testset "fixedpoint (priv)" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    @test Hopnet.is_fixedpoint(W, a1)
    @test Hopnet.is_fixedpoint(W, a2)
    @test Hopnet.is_fixedpoint(W, a3)

    @test !Hopnet.is_fixedpoint(W, [-1 1 1 1]')

    # complements
    complement(a) = (ai -> ai==1 ? -1 : 1).(a)
    @test Hopnet.is_fixedpoint(W, complement(a1))
    @test Hopnet.is_fixedpoint(W, complement(a2))
    @test Hopnet.is_fixedpoint(W, complement(a3))
end

@testset "run epoch (priv)" begin
    a1 = [1 1 1 1]'
    a2 = [1 -1 1 -1]'
    a3 = [1 1 -1 -1]'
    A = [a1 a2 a3]
    W = learn(4,A)

    # fixedpoint
    @test all(Hopnet.run_epoch(W, a1) .== a1)

    # non-fixedpoint (1 epoch to stable)
    a = [-1 1 1 1]'
    a_new1 = Hopnet.run_epoch(W, a; node_order=[1 2 3 4])
    a_new2 = Hopnet.run_epoch(W, a_new1; node_order=[1 2 3 4])

    @test any(a_new1 .!= a)
    @test isequal(a_new1, [1 1 1 1]')
    @test isequal(a_new2, a_new1)
end
