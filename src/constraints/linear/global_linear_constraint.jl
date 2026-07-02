export GlobalLinearConstraint

using SparseArrays

"""
    struct GlobalLinearConstraint <: AbstractLinearConstraint

A general linear constraint on a global variable block: `lb ≤ A·g ≤ ub`, where
`g` is the global block named `name` (length = number of columns of `A`). Each
row of `A` becomes one affine row over the global variables; rows with
`lb[r] == ub[r]` are emitted as equalities (`MOI.EqualTo`), otherwise as the
finite-sided bounds (`GreaterThan`/`LessThan`, skipping `±Inf`).

This is the primitive for relations and bounds on *subcomponents* of a global
block — which the per-variable `BoundsConstraint`/`GlobalEqualityConstraint`
cannot express (they pin the whole block to values, not linear combinations of
its slots). Typical uses: pin a B-spline endpoint *derivative*
(`c₁ − c₀ = 0`, an equality relation) or bound a B-spline slew
(`|c_{i+1} − c_i| ≤ vₘₐₓ`, a two-sided inequality).

# Fields
- `name::Symbol`: global block to constrain (a key of `traj.global_components`)
- `A::SparseMatrixCSC{Float64,Int}`: `(n_rows × global_dim)` coefficient matrix
- `lb::Vector{Float64}`, `ub::Vector{Float64}`: per-row bounds (length `n_rows`)
- `label::String`
"""
struct GlobalLinearConstraint <: AbstractLinearConstraint
    name::Symbol
    A::SparseMatrixCSC{Float64,Int}
    lb::Vector{Float64}
    ub::Vector{Float64}
    label::String
end

"""
    GlobalLinearConstraint(name, A, lb, ub; label=...)

Inequality form `lb ≤ A·g ≤ ub`. `A` may be any `AbstractMatrix` (stored sparse).
"""
function GlobalLinearConstraint(
    name::Symbol,
    A::AbstractMatrix,
    lb::AbstractVector,
    ub::AbstractVector;
    label::String = "global linear constraint on :$name",
)
    # Preserve sparsity (don't densify a banded slew-bound A): convert straight
    # to CSC. SparseArrays.sparse on a dense Matrix would be O(rows×cols).
    As = convert(SparseMatrixCSC{Float64,Int}, A)
    size(As, 1) == length(lb) == length(ub) || throw(
        ArgumentError(
            "row count mismatch: A has $(size(As,1)) rows, lb has $(length(lb)), ub has $(length(ub))",
        ),
    )
    all(lb .<= ub) || throw(ArgumentError("lb must be elementwise ≤ ub"))
    return GlobalLinearConstraint(name, As, Vector{Float64}(lb), Vector{Float64}(ub), label)
end

"""
    GlobalLinearConstraint(name, A, b; label=...)

Equality form `A·g = b`.
"""
function GlobalLinearConstraint(
    name::Symbol,
    A::AbstractMatrix,
    b::AbstractVector;
    label::String = "global linear equality on :$name",
)
    return GlobalLinearConstraint(name, A, b, b; label = label)
end

function Base.show(io::IO, c::GlobalLinearConstraint)
    print(io, "GlobalLinearConstraint: \"$(c.label)\" ($(size(c.A,1)) rows on :$(c.name))")
end

# =========================================================================== #

@testitem "GlobalLinearConstraint - equality relation g[1] = g[2]" begin
    using SparseArrays
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    gdim = length(traj.global_components[:g])
    # one row: g[1] - g[2] = 0
    A = spzeros(1, gdim);
    A[1, 1] = 1.0;
    A[1, 2] = -1.0
    con = GlobalLinearConstraint(:g, A, [0.0])

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 100)

    g = prob.trajectory.global_data[traj.global_components[:g]]
    @test abs(g[1] - g[2]) < 1e-7
end

@testitem "GlobalLinearConstraint - two-sided inequality bound on a difference" begin
    using SparseArrays
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    gdim = length(traj.global_components[:g])
    # -0.1 ≤ g[1] - g[2] ≤ 0.1
    A = spzeros(1, gdim);
    A[1, 1] = 1.0;
    A[1, 2] = -1.0
    con = GlobalLinearConstraint(:g, A, [-0.1], [0.1])

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 100)

    g = prob.trajectory.global_data[traj.global_components[:g]]
    @test g[1] - g[2] <= 0.1 + 1e-7
    @test g[1] - g[2] >= -0.1 - 1e-7
end

@testitem "GlobalLinearConstraint - infeasible all-zero row is surfaced" begin
    using SparseArrays
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    gdim = length(traj.global_components[:g])
    # All-zero row reduces to 1 ≤ 0·g ≤ 2, i.e. 1 ≤ 0 ≤ 2 — infeasible.
    A = spzeros(1, gdim)
    con = GlobalLinearConstraint(:g, A, [1.0], [2.0])

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    @test_throws ErrorException solve!(prob; max_iter = 1)
end

@testitem "GlobalLinearConstraint - multi-row banded slew bound" begin
    using SparseArrays
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    gdim = length(traj.global_components[:g])
    # Banded first-difference operator: |g[i+1] - g[i]| ≤ vmax for all i.
    nrows = gdim - 1
    A = spzeros(nrows, gdim)
    for i = 1:nrows
        A[i, i] = -1.0
        A[i, i+1] = 1.0
    end
    vmax = 0.5
    con = GlobalLinearConstraint(:g, A, fill(-vmax, nrows), fill(vmax, nrows))
    # The conversion must keep A sparse (not densify it).
    @test con.A isa SparseMatrixCSC{Float64,Int}
    @test nnz(con.A) == 2nrows

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 100)

    g = prob.trajectory.global_data[traj.global_components[:g]]
    @test all(abs.(diff(g)) .<= vmax + 1e-7)
end

@testitem "GlobalLinearConstraint - one-sided (Inf) bound" begin
    using SparseArrays
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    gdim = length(traj.global_components[:g])
    # Only a lower bound: g[1] - g[2] ≥ 0.2, upper side is +Inf (must be skipped,
    # emitting a lone GreaterThan and no LessThan).
    A = spzeros(1, gdim);
    A[1, 1] = 1.0;
    A[1, 2] = -1.0
    con = GlobalLinearConstraint(:g, A, [0.2], [Inf])

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 100)

    g = prob.trajectory.global_data[traj.global_components[:g]]
    @test g[1] - g[2] >= 0.2 - 1e-7
end

@testitem "GlobalLinearConstraint - equality overload A·g = b" begin
    using SparseArrays
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    gdim = length(traj.global_components[:g])
    # Two-arg equality form: g[1] + g[2] = 1.0.
    A = spzeros(1, gdim);
    A[1, 1] = 1.0;
    A[1, 2] = 1.0
    con = GlobalLinearConstraint(:g, A, [1.0])
    @test con.lb == con.ub == [1.0]

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 100)

    g = prob.trajectory.global_data[traj.global_components[:g]]
    @test abs(g[1] + g[2] - 1.0) < 1e-7
end
