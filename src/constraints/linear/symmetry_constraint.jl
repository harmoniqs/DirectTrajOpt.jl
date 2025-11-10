export SymmetryConstraint
export SymmetricControlConstraint

struct SymmetryConstraint <: AbstractLinearConstraint
    even_index_pairs::Vector{Tuple{Int64,Int64}}
    odd_index_pairs::Vector{Tuple{Int64,Int64}}
    label::String 
end

function SymmetricControlConstraint(
    traj::NamedTrajectory,
    name::Symbol,
    idx::Vector{Int64};
    even = true,
    label = "symmetry constraint on $name"
)
    even_pairs = Vector{Tuple{Int64,Int64}}()
    odd_pairs = Vector{Tuple{Int64,Int64}}()

    component_indicies = [slice(t, traj.components[name], traj.dim)[idx] for t ∈ 1:traj.N]
    if(even)
        even_pairs = vcat(even_pairs,reduce(vcat,[collect(zip(component_indicies[[idx,traj.N - idx+1]]...)) for idx in 1:traj.N ÷ 2]))
    else 
        odd_pairs = vcat(odd_pairs,reduce(vcat,[collect(zip(component_indicies[[idx,traj.N - idx+1]]...)) for idx in 1:traj.N ÷ 2]))
    end 

    if traj.timestep isa Symbol
        time_indices = [index(k, traj.components[traj.timestep][1], traj.dim) for k ∈ 1:traj.N]
        even_pairs = vcat(even_pairs,[(time_indices[idx],time_indices[traj.N + 1 - idx]) for idx ∈ 1:traj.N÷2]) 
    end 

    return SymmetryConstraint(
        even_pairs,
        odd_pairs,
        label
    )

end

# =========================================================================== #

@testitem "testing symmetry constraint" begin

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

    sym_constraint = SymmetricControlConstraint(
        prob.trajectory, 
        :u,
        [1];
        even = true
    );
    push!(prob.constraints, sym_constraint);
    
    solve!(prob; max_iter=10)
end
