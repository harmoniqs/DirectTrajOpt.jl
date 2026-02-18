export L1SlackConstraint

###
### L1SlackConstraint
###

"""
    struct L1SlackConstraint <: AbstractLinearConstraint

Linear constraint tying a slack variable to the absolute value of another variable.

For each timestep `k` and component `i`, enforces:
```math
v_{k,i} \\leq s_{k,i}, \\quad -v_{k,i} \\leq s_{k,i}
\\quad \\Longleftrightarrow \\quad |v_{k,i}| \\leq s_{k,i}
```

The bound `s ≥ 0` is expected to come from the trajectory's bounds on the slack
component. When combined with a [`LinearRegularizer`](@ref) on the slack variable,
this yields an exact L1 penalty on `v`.

# Fields
- `var_name::Symbol`: Variable to penalize (e.g. `:du`)
- `slack_name::Symbol`: Slack variable name (e.g. `:s_du`)
- `times::Vector{Int}`: Time indices where constraint is applied
- `label::String`: Constraint label
"""
struct L1SlackConstraint <: AbstractLinearConstraint
    var_name::Symbol
    slack_name::Symbol
    times::Vector{Int}
    label::String
end

"""
    L1SlackConstraint(
        var_name::Symbol,
        slack_name::Symbol,
        traj::NamedTrajectory;
        times::AbstractVector{Int}=1:traj.N,
        label="L1 slack constraint: |var_name| ≤ slack_name"
    )

Construct an L1 slack constraint tying `|var_name|` to `slack_name`.
"""
function L1SlackConstraint(
    var_name::Symbol,
    slack_name::Symbol,
    traj::NamedTrajectory;
    times::AbstractVector{Int} = 1:traj.N,
    label = "L1 slack constraint: |$var_name| ≤ $slack_name",
)
    @assert var_name ∈ traj.names "Variable $var_name not found in trajectory"
    @assert slack_name ∈ traj.names "Slack variable $slack_name not found in trajectory"
    @assert traj.dims[var_name] == traj.dims[slack_name] "Dimension mismatch: $(var_name) ($(traj.dims[var_name])) vs $(slack_name) ($(traj.dims[slack_name]))"
    return L1SlackConstraint(var_name, slack_name, Vector{Int}(times), label)
end

# =========================================================================== #

@testitem "L1SlackConstraint" begin
    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    # Add a slack variable for du
    n_du = traj.dims[:du]
    s_du_data = abs.(traj[:du])
    traj = add_component(
        traj,
        :s_du,
        s_du_data;
        type = :control,
        bounds = (s_du = (zeros(n_du), fill(Inf, n_du)),),
    )

    integrators = [
        BilinearIntegrator(G, :x, :u, traj),
        DerivativeIntegrator(:u, :du, traj),
        DerivativeIntegrator(:du, :ddu, traj),
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0)
    J += LinearRegularizer(:s_du, traj, 1e-2)
    J += MinimumTimeObjective(traj)

    l1_con = L1SlackConstraint(:du, :s_du, traj)

    prob = DirectTrajOptProblem(traj, J, integrators; constraints = [l1_con])
    solve!(prob; max_iter = 100)

    # Verify slack constraint: s_du ≥ |du| at solution
    for t = 1:traj.N
        du = prob.trajectory[t][:du]
        s_du = prob.trajectory[t][:s_du]
        @test all(s_du .>= abs.(du) .- 1e-6)
    end
end
