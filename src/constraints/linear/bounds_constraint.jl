export BoundsConstraint
export GlobalBoundsConstraint

### 
### BoundsConstraint
###

"""
    struct BoundsConstraint <: AbstractLinearConstraint

Represents a box constraint defined by variable names.
Indices and concrete bounds are computed in constrain!.

# Fields
- `var_names::Union{Symbol, Vector{Symbol}}`: Variable name(s) to constrain
- `times::Union{Nothing, Vector{Int}}`: Time indices (nothing for global variables)
- `bounds_values::Union{Float64, Vector{Float64}, Tuple{Vector{Float64}, Vector{Float64}}}`: Bound specification
- `is_global::Bool`: Whether this is a global variable constraint
- `subcomponents::Union{Nothing, UnitRange{Int}}`: Optional subcomponent selection
- `label::String`: Constraint label
"""
struct BoundsConstraint <: AbstractLinearConstraint
    var_names::Union{Symbol,Vector{Symbol}}
    times::Union{Nothing,Vector{Int}}
    bounds_values::Union{Float64,Vector{Float64},Tuple{Vector{Float64},Vector{Float64}}}
    is_global::Bool
    subcomponents::Union{Nothing,UnitRange{Int}}
    label::String
end

"""
    BoundsConstraint(
        name::Symbol,
        ts::Vector{Int},
        bounds::Union{Float64, Vector{Float64}, Tuple{Vector{Float64}, Vector{Float64}}};
        subcomponents=nothing,
        label="bounds constraint on trajectory variable \$name"
    )

Constructs box constraint for trajectory variable.
Indices are computed when applied to a trajectory.

# Arguments
- `name`: Variable name
- `ts`: Time indices
- `bounds`: Can be:
  - Scalar: symmetric bounds [-bounds, bounds]
  - Vector: symmetric bounds [-bounds, bounds] element-wise
  - Tuple: (lower_bounds, upper_bounds)
"""
function BoundsConstraint(
    name::Symbol,
    ts::AbstractVector{Int},
    bounds::Union{Float64,Vector{Float64},Tuple{Vector{Float64},Vector{Float64}}};
    subcomponents::Union{Nothing,UnitRange{Int}} = nothing,
    label = "bounds constraint on trajectory variable $name",
)
    return BoundsConstraint(
        name,
        collect(ts),
        bounds,
        false,  # not global
        subcomponents,
        label,
    )
end

"""
    GlobalBoundsConstraint(
        name::Symbol,
        bounds::Union{Float64, Vector{Float64}, Tuple{Vector{Float64}, Vector{Float64}}};
        label="bounds constraint on global variable \$name"
    )

Constructs box constraint for global variable.
Indices are computed when applied to a trajectory.
"""
function GlobalBoundsConstraint(
    name::Symbol,
    bounds::Union{Float64,Vector{Float64},Tuple{Vector{Float64},Vector{Float64}}};
    label = "bounds constraint on global variable $name",
)
    return BoundsConstraint(
        name,
        nothing,  # no times for global
        bounds,
        true,  # is global
        nothing,  # no subcomponents for global
        label,
    )
end

# =========================================================================== #

@testitem "BoundsConstraint - symmetric scalar bounds" begin
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

    # Test symmetric scalar bounds (use ddu which has no automatic constraints)
    ddu_bound = 0.5
    bounds_con = BoundsConstraint(:ddu, 1:traj.N, ddu_bound)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [bounds_con])
    solve!(prob; max_iter = 100)

    # Verify bounds are satisfied
    for t = 1:traj.N
        ddu = prob.trajectory[t][:ddu]
        @test all(ddu .>= -ddu_bound - 1e-6)
        @test all(ddu .<= ddu_bound + 1e-6)
    end
end

@testitem "BoundsConstraint - asymmetric tuple bounds" begin
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

    # Test asymmetric bounds with different lower and upper
    # Use :du which doesn't have automatic bounds from traj.bounds
    du_dim = traj.dims[:du]
    lb = fill(-0.3, du_dim)
    ub = fill(0.7, du_dim)
    bounds_con = BoundsConstraint(:du, 1:traj.N, (lb, ub))

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [bounds_con])
    solve!(prob; max_iter = 100)

    # Verify bounds are satisfied
    for t = 1:traj.N
        du = prob.trajectory[t][:du]
        @test all(du .>= lb .- 1e-6)
        @test all(du .<= ub .+ 1e-6)
    end
end

@testitem "BoundsConstraint - with subcomponents" begin
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

    # Test bounds on only first component of state
    # Avoid timestep 1 where x has an initial constraint from traj.initial
    x_bound = 0.8
    bounds_con = BoundsConstraint(:x, 2:traj.N, x_bound; subcomponents = 1:1)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [bounds_con])
    solve!(prob; max_iter = 100)

    # Verify bounds are satisfied on first component only
    for t = 2:traj.N
        x = prob.trajectory[t][:x]
        @test x[1] >= -x_bound - 1e-6
        @test x[1] <= x_bound + 1e-6
        # Second component should not be constrained
    end
end

@testitem "GlobalBoundsConstraint" begin
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
    J += GlobalObjective(g -> norm(g)^2, :g, traj; Q = 1.0)

    # Test global bounds constraint
    g_bound = 0.5
    global_bounds_con = GlobalBoundsConstraint(:g, g_bound)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [global_bounds_con])
    solve!(prob; max_iter = 100)

    # Verify global bounds are satisfied
    g_components = traj.global_components[:g]
    g_values = prob.trajectory.global_data[g_components]
    @test all(g_values .>= -g_bound - 1e-6)
    @test all(g_values .<= g_bound + 1e-6)
end
