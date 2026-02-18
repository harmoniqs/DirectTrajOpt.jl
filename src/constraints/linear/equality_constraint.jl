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
- `values::Vector{Float64}`: Constraint values
- `is_global::Bool`: Whether this is a global variable constraint
- `label::String`: Constraint label
"""
struct EqualityConstraint <: AbstractLinearConstraint
    var_names::Union{Symbol,Vector{Symbol}}
    times::Union{Nothing,Vector{Int}}
    values::Vector{Float64}
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
