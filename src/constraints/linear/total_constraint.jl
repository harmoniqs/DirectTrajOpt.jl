export TotalConstraint
export DurationConstraint

struct TotalConstraint <: AbstractLinearConstraint
    indices::Vector{Int}
    value::Float64
    label::String
end


function DurationConstraint(
    traj::NamedTrajectory,
    value::Float64;
    label="duration constraint of $value"
)
    @assert traj.timestep isa Symbol
    indices = [index(k, traj.components[traj.timestep][1], traj.dim) for k ∈ 1:traj.N]
    return TotalConstraint(indices, value ,label)
end

# =========================================================================== #

@testitem "testing duration constraint" begin

    include("../../../test/test_utils.jl")

    G, traj = bilinear_dynamics_and_trajectory()

    integrators = [
        BilinearIntegrator(G, :x, :u),
        DerivativeIntegrator(:u, :du),
        DerivativeIntegrator(:du, :ddu)
    ]

    J = TerminalObjective(x -> norm(x - traj.goal.x)^2, :x, traj)
    J += QuadraticRegularizer(:u, traj, 1.0) 
    J += QuadraticRegularizer(:du, traj, 1.0)
    J += MinimumTimeObjective(traj)

    prob = DirectTrajOptProblem(traj, J, integrators;)

    dur_constraint = DurationConstraint(
        prob.trajectory,
        10.0;
    )
    push!(prob.constraints, dur_constraint);

    solve!(prob; max_iter=10)
end
