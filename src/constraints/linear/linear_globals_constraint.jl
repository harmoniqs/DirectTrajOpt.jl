export LinearGlobalsConstraint

"""
    struct LinearGlobalsConstraint <: AbstractLinearConstraint

Sparse linear constraint over the trajectory's concatenated globals vector,
with per-row upper and lower bounds.

The constraint expresses `lb .≤ A * globals .≤ ub`, where `A` is a sparse
matrix whose columns span the entire `traj.global_data` vector
(`size(A, 2) == traj.global_dim`). Bound vectors may contain `±Inf` to encode
one-sided rows; per-row dispatch selects the appropriate MOI set at
emission time:

| `lb[i]` finite | `ub[i]` finite | `lb[i] == ub[i]` | MOI set                |
|:--------------:|:--------------:|:----------------:|:-----------------------|
| ✓              | ✓              | ✓                | `MOI.EqualTo(lb[i])`   |
| ✓              | ✓              | ✗                | `MOI.Interval(lb[i], ub[i])` |
| ✓              | ✗              | —                | `MOI.GreaterThan(lb[i])` |
| ✗              | ✓              | —                | `MOI.LessThan(ub[i])`  |
| ✗              | ✗              | —                | error                  |

# Fields
- `A::SparseMatrixCSC{Float64,Int}`: constraint matrix, size `(n_rows, traj.global_dim)`
- `lb::Vector{Float64}`: lower bounds per row; `-Inf` allowed
- `ub::Vector{Float64}`: upper bounds per row; `+Inf` allowed
- `label::String`: constraint label
"""
struct LinearGlobalsConstraint <: AbstractLinearConstraint
    A::SparseMatrixCSC{Float64,Int}
    lb::Vector{Float64}
    ub::Vector{Float64}
    label::String
end

"""
    LinearGlobalsConstraint(
        A::SparseMatrixCSC{Float64,Int},
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real};
        label::String = "linear globals constraint",
    )

Two-sided linear constraint `lb .≤ A * globals .≤ ub`. Pass `-Inf` / `+Inf`
in `lb` / `ub` to encode one-sided rows.
"""
function LinearGlobalsConstraint(
    A::SparseMatrixCSC{Float64,Int},
    lb::AbstractVector{<:Real},
    ub::AbstractVector{<:Real};
    label::String = "linear globals constraint",
)
    @assert length(lb) == length(ub) == size(A, 1) (
        "lb and ub length must equal size(A, 1) = $(size(A, 1))"
    )
    @assert all(lb .<= ub) "lb must be ≤ ub elementwise"
    return LinearGlobalsConstraint(A, collect(Float64, lb), collect(Float64, ub), label)
end

"""
    LinearGlobalsConstraint(A, b, <=; label = "…")

One-sided constraint `A * globals .≤ b`.
"""
function LinearGlobalsConstraint(
    A::SparseMatrixCSC{Float64,Int},
    b::AbstractVector{<:Real},
    ::typeof(<=);
    label::String = "linear globals constraint (≤)",
)
    return LinearGlobalsConstraint(
        A,
        fill(-Inf, length(b)),
        collect(Float64, b);
        label = label,
    )
end

"""
    LinearGlobalsConstraint(A, b, >=; label = "…")

One-sided constraint `A * globals .≥ b`.
"""
function LinearGlobalsConstraint(
    A::SparseMatrixCSC{Float64,Int},
    b::AbstractVector{<:Real},
    ::typeof(>=);
    label::String = "linear globals constraint (≥)",
)
    return LinearGlobalsConstraint(
        A,
        collect(Float64, b),
        fill(+Inf, length(b));
        label = label,
    )
end

"""
    LinearGlobalsConstraint(A, b, ==; label = "…")

Linear equality constraint `A * globals .== b`.
"""
function LinearGlobalsConstraint(
    A::SparseMatrixCSC{Float64,Int},
    b::AbstractVector{<:Real},
    ::typeof(==);
    label::String = "linear globals constraint (=)",
)
    bvec = collect(Float64, b)
    return LinearGlobalsConstraint(A, copy(bvec), copy(bvec); label = label)
end

function Base.show(io::IO, c::LinearGlobalsConstraint)
    print(
        io,
        "LinearGlobalsConstraint: \"$(c.label)\" ",
        "($(size(c.A, 1)) rows, $(SparseArrays.nnz(c.A)) nnz, ",
        "$(size(c.A, 2)) globals)",
    )
end

# =========================================================================== #

@testitem "LinearGlobalsConstraint - equality row" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    # Single equality row: g[1] + g[2] + g[end] = 1.5
    g_dim = traj.global_dim
    A = sparse([1, 1, 1], [1, 2, g_dim], [1.0, 1.0, 1.0], 1, g_dim)
    target = 1.5
    con = LinearGlobalsConstraint(A, [target], [target]; label = "equality row")

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 200)

    g_vals = prob.trajectory.global_data
    @test abs(g_vals[1] + g_vals[2] + g_vals[g_dim] - target) < 1e-6
end

@testitem "LinearGlobalsConstraint - two-sided Interval row" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    # Push globals toward zero — without constraint the sum would be 0.
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    # Single interval row: 0.5 ≤ g[1] + g[2] ≤ 1.5  (lower bound is active)
    g_dim = traj.global_dim
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, g_dim)
    con = LinearGlobalsConstraint(A, [0.5], [1.5]; label = "two-sided")

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 200)

    g_vals = prob.trajectory.global_data
    s = g_vals[1] + g_vals[2]
    @test 0.5 - 1e-6 ≤ s ≤ 1.5 + 1e-6
    # Lower bound should be active to within Ipopt's interior-point tolerance
    # (objective alone would drive s to 0, well below the [0.5, 1.5] interval).
    @test abs(s - 0.5) < 1e-3
end

@testitem "LinearGlobalsConstraint - one-sided ≤ row" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    # Pull globals toward +1 elementwise — unconstrained sum > 0.5.
    g_dim = traj.global_dim
    target = fill(1.0, g_dim)
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g - target)^2, :g, traj; Q = 1.0)

    # Single ≤ row: g[1] + g[2] ≤ 0.5
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, g_dim)
    con = LinearGlobalsConstraint(A, [0.5], <=; label = "one-sided ≤")

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 200)

    g_vals = prob.trajectory.global_data
    s = g_vals[1] + g_vals[2]
    @test s ≤ 0.5 + 1e-6
    @test abs(s - 0.5) < 1e-4  # constraint should be active
end

@testitem "LinearGlobalsConstraint - one-sided ≥ row" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    # Pull globals toward -1 elementwise — unconstrained sum < -0.5.
    g_dim = traj.global_dim
    target = fill(-1.0, g_dim)
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g - target)^2, :g, traj; Q = 1.0)

    # Single ≥ row: g[1] + g[2] ≥ -0.5
    A = sparse([1, 1], [1, 2], [1.0, 1.0], 1, g_dim)
    con = LinearGlobalsConstraint(A, [-0.5], >=; label = "one-sided ≥")

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 200)

    g_vals = prob.trajectory.global_data
    s = g_vals[1] + g_vals[2]
    @test s ≥ -0.5 - 1e-6
    @test abs(s - (-0.5)) < 1e-4  # constraint should be active
end

@testitem "LinearGlobalsConstraint - cross-named-global coupling" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    # Build a trajectory with TWO named globals :g and :h.
    G, traj0 = bilinear_dynamics_and_trajectory(add_global = true)
    h_data = randn(5)
    traj = add_component(traj0, :h, h_data, type = :global)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)
    J += GlobalObjective(h -> norm(h)^2, :h, traj; Q = 1.0)

    g_comps = traj.global_components[:g]
    h_comps = traj.global_components[:h]
    g1 = g_comps[1]                # column index of :g[1] in global_data
    h1 = h_comps[1]                # column index of :h[1] in global_data

    # Row 1: g[1] - h[1] == 0.4   (couples the two named globals)
    A = sparse([1, 1], [g1, h1], [1.0, -1.0], 1, traj.global_dim)
    con = LinearGlobalsConstraint(A, [0.4], ==; label = "cross-global coupling")

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 200)

    g_vals = prob.trajectory.global_data
    @test abs(g_vals[g1] - g_vals[h1] - 0.4) < 1e-6
end

@testitem "LinearGlobalsConstraint - multi-row mixed senses" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    g_dim = traj.global_dim
    # Row 1 (equality):   g[1] = 0.5
    # Row 2 (interval):   0.0 ≤ g[2] + g[3] ≤ 0.3
    # Row 3 (≤ only):     g[4] ≤ 0.1
    # Row 4 (≥ only):     g[5] ≥ -0.2
    rows = [1, 2, 2, 3, 4]
    cols = [1, 2, 3, 4, 5]
    vals = [1.0, 1.0, 1.0, 1.0, 1.0]
    A = sparse(rows, cols, vals, 4, g_dim)
    lb = [0.5, 0.0, -Inf, -0.2]
    ub = [0.5, 0.3, 0.1, +Inf]
    con = LinearGlobalsConstraint(A, lb, ub; label = "mixed senses")

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    solve!(prob; max_iter = 200)

    g_vals = prob.trajectory.global_data
    @test abs(g_vals[1] - 0.5) < 1e-6
    @test 0.0 - 1e-6 ≤ g_vals[2] + g_vals[3] ≤ 0.3 + 1e-6
    @test g_vals[4] ≤ 0.1 + 1e-6
    @test g_vals[5] ≥ -0.2 - 1e-6
end

@testitem "LinearGlobalsConstraint - cross-solver agreement" setup=[DTOTestHelpers] begin
    using Random
    include("../../../test/test_utils.jl")

    function build_seeded_problem(seed::UInt64)
        Random.seed!(seed)
        G, traj = bilinear_dynamics_and_trajectory(add_global = true)
        integrators = [
            BilinearIntegrator(G, :x, :u, traj),
            DerivativeIntegrator(:u, :du, traj),
            DerivativeIntegrator(:du, :ddu, traj),
        ]
        J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
        J += QuadraticRegularizer(:u, traj, 1.0)
        J += MinimumTimeObjective(traj)
        J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

        g_dim = traj.global_dim
        # Single equality row pinning a 3-term combination.
        A = sparse([1, 1, 1], [1, 2, 3], [1.0, 1.0, 1.0], 1, g_dim)
        con = LinearGlobalsConstraint(A, [0.4], [0.4]; label = "xsolver")
        return DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    end

    seed = UInt64(0x1234abcd)

    prob_ipopt = build_seeded_problem(seed)
    solve!(
        prob_ipopt;
        options = IpoptSolverExt.IpoptOptions(; max_iter = 200, print_level = 0),
    )

    prob_madnlp = build_seeded_problem(seed)
    solve!(prob_madnlp; options = DirectTrajOpt.MadNLPOptions(; max_iter = 200))

    g_ipopt = prob_ipopt.trajectory.global_data
    g_madnlp = prob_madnlp.trajectory.global_data

    # Both solvers must satisfy the constraint to high precision.
    @test abs(g_ipopt[1] + g_ipopt[2] + g_ipopt[3] - 0.4) < 1e-6
    @test abs(g_madnlp[1] + g_madnlp[2] + g_madnlp[3] - 0.4) < 1e-6

    # And the recovered globals should agree.
    rmse = sqrt(sum((g_ipopt .- g_madnlp) .^ 2)) / length(g_ipopt)
    @test rmse < 1e-4
end

@testitem "LinearGlobalsConstraint - vacuous row errors" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    g_dim = traj.global_dim
    # Single row with -Inf / +Inf bounds — construction is permitted, but
    # the MOI emission must reject it.
    A = sparse([1], [1], [1.0], 1, g_dim)
    con = LinearGlobalsConstraint(A, [-Inf], [+Inf]; label = "vacuous")

    @test_throws ErrorException DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = [con],
    ) |> p -> solve!(p; max_iter = 5)
end

@testitem "LinearGlobalsConstraint - empty row errors" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    # Row 1 has non-zeros; row 2 is empty — should error at application time.
    g_dim = traj.global_dim
    A = sparse([1], [1], [1.0], 2, g_dim)
    con = LinearGlobalsConstraint(A, [0.0, 0.0], [0.0, 0.0]; label = "empty row")

    @test_throws ErrorException DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = [con],
    ) |> p -> solve!(p; max_iter = 5)
end

@testitem "LinearGlobalsConstraint - dimension mismatch errors" begin
    include("../../../test/test_utils.jl")
    using SparseArrays

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    # A has the wrong column count vs traj.global_dim.
    A = sparse([1], [1], [1.0], 1, traj.global_dim + 3)
    con = LinearGlobalsConstraint(A, [0.0], [0.0]; label = "dim mismatch")

    @test_throws AssertionError DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints = [con],
    ) |> p -> solve!(p; max_iter = 5)
end

@testitem "LinearGlobalsConstraint - lb > ub errors at construction" begin
    using SparseArrays

    A = sparse([1], [1], [1.0], 1, 3)
    @test_throws AssertionError LinearGlobalsConstraint(A, [1.0], [0.0])
end

@testitem "LinearGlobalsConstraint - verbose smoke test" setup=[DTOTestHelpers] begin
    using SparseArrays
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory(add_global = true)
    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]
    J = QuadraticRegularizer(:u, traj, 1.0)
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)
    A = sparse([1], [1], [1.0], 1, traj.global_dim)
    con = LinearGlobalsConstraint(A, [0.2], [0.2]; label = "lg-verbose-test")

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [con])
    evaluator = Solvers.Evaluator(prob; eval_hessian = true, verbose = false)
    nl_cons = Solvers.get_nonlinear_constraints(prob)
    block_data = MOI.NLPBlockData(nl_cons, evaluator, true)

    optimizer = Ipopt.Optimizer()
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    data_dim = traj.dim * traj.N
    variables = MOI.add_variables(optimizer, data_dim + traj.global_dim)
    MOI.set(
        optimizer,
        MOI.VariablePrimalStart(),
        variables[1:data_dim],
        collect(traj.datavec),
    )

    linear_constraints = AbstractLinearConstraint[filter(
        c -> c isa AbstractLinearConstraint,
        prob.constraints,
    )...]

    output = capture_stdout() do
        Solvers.constrain!(
            optimizer,
            variables,
            linear_constraints,
            prob.trajectory;
            verbose = true,
        )
    end
    @test contains(output, "lg-verbose-test")
end
