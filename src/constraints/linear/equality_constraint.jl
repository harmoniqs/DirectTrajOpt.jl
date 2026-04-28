export EqualityConstraint
export GlobalEqualityConstraint

### 
### EqualityConstraint
###

"""
    struct EqualityConstraint <: AbstractLinearConstraint

Represents a linear equality constraint defined by variable names.
Indices are computed when constraint is applied in constrain!.

# Fields
- `var_names::Union{Symbol, Vector{Symbol}}`: Variable name(s) to constrain
- `times::Union{Nothing, Vector{Int}}`: Time indices (nothing for global variables)
- `values::Union{Vector{Float64}, Matrix{Float64}}`: Constraint values (Vector for uniform, Matrix for per-timestep)
- `is_global::Bool`: Whether this is a global variable constraint
- `label::String`: Constraint label
"""
struct EqualityConstraint <: AbstractLinearConstraint
    var_names::Union{Symbol,Vector{Symbol}}
    times::Union{Nothing,Vector{Int}}
    values::Union{Vector{Float64},Matrix{Float64}}
    is_global::Bool
    label::String
end

"""
    EqualityConstraint(
        name::Symbol,
        ts::Vector{Int},
        val::Union{Float64, Vector{Float64}};
        label="equality constraint on trajectory variable \$name"
    )

Constructs equality constraint for trajectory variable.
Indices are computed when applied to a trajectory.
"""
function EqualityConstraint(
    name::Symbol,
    ts::AbstractVector{Int},
    val::Union{Float64,Vector{Float64}};
    label = "equality constraint on trajectory variable $name",
)
    # Convert scalar to vector (will be repeated per time step)
    values = val isa Float64 ? [val] : val

    return EqualityConstraint(
        name,
        collect(ts),
        values,
        false,  # not global
        label,
    )
end

"""
    GlobalEqualityConstraint(
        name::Symbol,
        val::Union{Float64, Vector{Float64}};
        label="equality constraint on global variable \$name"
    )::EqualityConstraint

Constructs equality constraint for global variable.
Indices are computed when applied to a trajectory.
"""
function GlobalEqualityConstraint(
    name::Symbol,
    val::Union{Float64,Vector{Float64}};
    label = "equality constraint on global variable $name",
)
    # Convert scalar to vector
    values = val isa Float64 ? [val] : val

    return EqualityConstraint(
        name,
        nothing,  # no times for global
        values,
        true,  # is global
        label,
    )
end

"""
    EqualityConstraint(
        name::Symbol,
        ts::AbstractVector{Int},
        val::Matrix{Float64};
        label="per-timestep equality constraint on trajectory variable \$name"
    )

Constructs a per-timestep equality constraint for a trajectory variable.
`val` must have size `(var_dim, length(ts))` — column `k` pins the variable
at timestep `ts[k]`.
"""
function EqualityConstraint(
    name::Symbol,
    ts::AbstractVector{Int},
    val::AbstractMatrix{Float64};
    label = "per-timestep equality constraint on trajectory variable $name",
)
    @assert size(val, 2) == length(ts) (
        "Matrix columns ($(size(val, 2))) must match number of timesteps ($(length(ts)))"
    )
    return EqualityConstraint(name, collect(ts), Matrix(val), false, label)
end

export fix_trajectory_variable!

"""
    fix_trajectory_variable!(constraints, name, values; times)

Pin a trajectory variable to per-timestep values using an `EqualityConstraint`.
Removes any existing `BoundsConstraint` on `name` to avoid MOI conflicts.

# Arguments
- `constraints::Vector{<:AbstractConstraint}`: mutable constraint list
- `name::Symbol`: trajectory variable name to pin
- `values::AbstractMatrix{Float64}`: size `(var_dim, N)` — column `k` pins timestep `k`

# Keyword Arguments
- `times::AbstractVector{Int}`: timesteps to pin (default: `1:size(values, 2)`)
"""
function fix_trajectory_variable!(
    constraints::Vector{<:AbstractConstraint},
    name::Symbol,
    values::AbstractMatrix{Float64};
    times::AbstractVector{Int} = 1:size(values, 2),
)
    # Remove existing BoundsConstraint and EqualityConstraint on this trajectory variable
    # to avoid MOI conflicts (per-timestep pinning supersedes initial/final/bounds)
    filter!(constraints) do c
        if c isa BoundsConstraint && c.var_names == name && !c.is_global
            return false
        elseif c isa EqualityConstraint && c.var_names == name && !c.is_global
            return false
        end
        return true
    end
    # Add per-timestep equality constraint
    push!(constraints, EqualityConstraint(name, times, values))
    return constraints
end

export fix_global_variable!

"""
    fix_global_variable!(constraints, name, value)

Pin a global (time-invariant) variable to `value` using a `GlobalEqualityConstraint`.
Removes any existing `BoundsConstraint` or `EqualityConstraint` on the same global
variable to avoid MOI conflicts. Companion to [`fix_trajectory_variable!`](@ref) for
the global-variable case.

This is the integrator-agnostic mechanism for pinning a calibrated parameter
(e.g. learned `θ` in QILC alternating calibration) into the control NLP — any
globals-aware integrator (`HermitianExponentialIntegrator`, `SplineIntegrator`,
`NonHermitianExponentialIntegrator`, ...) will read the pinned value through the
NLP variable rather than requiring an integrator rebuild.

# Arguments
- `constraints::Vector{<:AbstractConstraint}`: mutable constraint list
- `name::Symbol`: global variable name to pin
- `value::AbstractVector{Float64}`: pin value (length must equal the global variable dim)
"""
function fix_global_variable!(
    constraints::Vector{<:AbstractConstraint},
    name::Symbol,
    value::AbstractVector{Float64},
)
    # Remove any existing BoundsConstraint and EqualityConstraint on this global variable
    # (per-timestep dedup at line ~134 only filters non-global constraints; this is the
    # globals counterpart). Pinning supersedes any prior bounds or pin on the same global.
    filter!(constraints) do c
        if c isa BoundsConstraint && c.var_names == name && c.is_global
            return false
        elseif c isa EqualityConstraint && c.var_names == name && c.is_global
            return false
        end
        return true
    end
    push!(constraints, GlobalEqualityConstraint(name, collect(Float64, value)))
    return constraints
end

function Base.show(io::IO, c::EqualityConstraint)
    print(io, "EqualityConstraint: \"$(c.label)\"")
end

# =========================================================================== #

@testitem "EqualityConstraint - trajectory variable" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Test that trajectory constraints from traj.initial are applied correctly
    prob = DirectTrajOptProblem(traj, J, integrators)
    solve!(prob; max_iter = 100)

    # Verify initial constraint from traj.initial is satisfied
    @test prob.trajectory[1][:x] ≈ [1.0, 0.0, 0.0, 0.0] atol=1e-6
end

@testitem "EqualityConstraint - scalar value" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Test equality constraint with scalar value at a middle timestep
    # (avoiding conflict with traj.initial and traj.final)
    du_mid = 0.0
    mid_con = EqualityConstraint(:du, [5], du_mid)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [mid_con])
    solve!(prob; max_iter = 100)

    # Verify constraint is satisfied (scalar applied to all components)
    @test all(abs.(prob.trajectory[5][:du]) .< 1e-6)
end

@testitem "GlobalEqualityConstraint" begin
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

    # Test global equality constraint
    # g has dimension N (10 by default in test_utils)
    g_value = fill(0.5, traj.N)
    global_con = GlobalEqualityConstraint(:g, g_value)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [global_con])
    solve!(prob; max_iter = 100)

    # Verify global constraint is satisfied
    g_components = traj.global_components[:g]
    @test prob.trajectory.global_data[g_components] ≈ g_value atol=1e-6
end

@testitem "EqualityConstraint - per-timestep matrix values" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Pin du at timesteps 3:7 to per-timestep values
    du_dim = traj.dims[:du]
    pin_times = 3:7
    du_ref = randn(du_dim, length(pin_times))
    pin_con = EqualityConstraint(:du, collect(pin_times), du_ref)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [pin_con])
    solve!(prob; max_iter = 100)

    # Verify: pinned values are exactly recovered
    for (k, t) in enumerate(pin_times)
        @test prob.trajectory[t][:du] ≈ du_ref[:, k] atol=1e-8
    end
end

@testitem "fix_trajectory_variable! removes bounds and pins values" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:ddu, traj, 1.0)
    J += MinimumTimeObjective(traj)

    # Build a normal problem first (generates initial/final/bounds constraints)
    prob_orig = DirectTrajOptProblem(traj, J, integrators)

    # Verify bounds and initial equality exist on :u before fixing
    @test any(c -> c isa BoundsConstraint && c.var_names == :u, prob_orig.constraints)
    @test any(c -> c isa EqualityConstraint && c.var_names == :u, prob_orig.constraints)

    # Fix u to per-timestep values (should remove bounds AND initial/final equality)
    u_ref = copy(traj[:u])
    constraints = deepcopy(prob_orig.constraints)
    fix_trajectory_variable!(constraints, :u, u_ref)

    # Old bounds and equality constraints on :u should be removed
    @test !any(c -> c isa BoundsConstraint && c.var_names == :u, constraints)
    # Only the new per-timestep equality should remain
    u_eq_cons = filter(c -> c isa EqualityConstraint && c.var_names == :u, constraints)
    @test length(u_eq_cons) == 1
    @test u_eq_cons[1].values isa Matrix{Float64}

    # Solve with 4-arg constructor (no double-adding trajectory constraints)
    prob = DirectTrajOptProblem(traj, J, integrators, constraints)
    solve!(prob; max_iter = 100)

    # Pinned values should have zero drift
    for t = 1:traj.N
        @test prob.trajectory[t][:u] ≈ u_ref[:, t] atol=1e-10
    end
end

@testitem "fix_global_variable! pins global to supplied value" begin
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

    # Build a normal problem first (so bounds/equality on globals may exist)
    prob_orig = DirectTrajOptProblem(traj, J, integrators)

    # Pin global :g to a specific value via fix_global_variable!
    g_dim = length(traj.global_components[:g])
    g_value = fill(0.42, g_dim)
    constraints = deepcopy(prob_orig.constraints)
    fix_global_variable!(constraints, :g, g_value)

    # After fixing, no remaining BoundsConstraint or EqualityConstraint on :g
    # other than our new GlobalEqualityConstraint
    g_eq_cons = filter(
        c -> c isa EqualityConstraint && c.var_names == :g && c.is_global,
        constraints,
    )
    @test length(g_eq_cons) == 1
    @test !any(c -> c isa BoundsConstraint && c.var_names == :g && c.is_global, constraints)

    # Re-applying should still leave exactly one equality constraint (dedup works)
    fix_global_variable!(constraints, :g, fill(0.5, g_dim))
    g_eq_cons2 = filter(
        c -> c isa EqualityConstraint && c.var_names == :g && c.is_global,
        constraints,
    )
    @test length(g_eq_cons2) == 1
    @test g_eq_cons2[1].values ≈ fill(0.5, g_dim)

    # Solve and confirm the global is pinned at the supplied value
    fix_global_variable!(constraints, :g, g_value)  # pin at 0.42 again
    prob = DirectTrajOptProblem(traj, J, integrators, constraints)
    solve!(prob; max_iter = 100)

    g_components = traj.global_components[:g]
    @test prob.trajectory.global_data[g_components] ≈ g_value atol=1e-6
end
